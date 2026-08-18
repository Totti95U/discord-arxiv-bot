from google import genai
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple, Union
import arxiv
import time
import os
import datetime
import requests
import json
import argparse
import io
import uuid
from urllib.parse import quote
from zoneinfo import ZoneInfo

# set up clients for arXiv, GenAI, and Discord
client_arxiv = arxiv.Client()
client_genai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

STATE_FILE_PATH = os.getenv("PENDING_JOBS_FILE", "state/pending_jobs.json")
STATE_SCHEMA_VERSION = 1
INTEREST_MODEL = os.getenv("INTEREST_MODEL", "gemini-3.5-flash-lite")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gemini-3.6-flash")
READING_MODEL = os.getenv("READING_MODEL", SUMMARY_MODEL)
DISCORD_API_BASE_URL = "https://discord.com/api/v10"
READ_EMOJI = "📖"
MAX_PDF_BYTES = 50 * 1024 * 1024
COMPLETED_BATCH_STATUS = (
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
)

try:
    BATCH_TIMEOUT_HOURS = int(os.getenv("BATCH_TIMEOUT_HOURS", "48"))
except ValueError:
    BATCH_TIMEOUT_HOURS = 48


def read_positive_number_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


DISCORD_CONNECT_TIMEOUT_SECONDS = read_positive_number_env("DISCORD_CONNECT_TIMEOUT_SECONDS", 5.0)
DISCORD_READ_TIMEOUT_SECONDS = read_positive_number_env("DISCORD_READ_TIMEOUT_SECONDS", 15.0)
DISCORD_RETRY_BACKOFF_SECONDS = read_positive_number_env("DISCORD_RETRY_BACKOFF_SECONDS", 1.0)
try:
    DISCORD_MAX_ATTEMPTS = max(1, int(os.getenv("DISCORD_MAX_ATTEMPTS", "3")))
except ValueError:
    DISCORD_MAX_ATTEMPTS = 3

DISCORD_CONTENT_LIMIT = 2000
DISCORD_EMBED_TITLE_LIMIT = 256
DISCORD_EMBED_FIELD_NAME_LIMIT = 256
DISCORD_EMBED_FIELD_VALUE_LIMIT = 1024
DISCORD_EMBED_AUTHOR_NAME_LIMIT = 256
DISCORD_EMBED_FOOTER_TEXT_LIMIT = 2048
DISCORD_EMBED_TOTAL_LIMIT = 6000


class InterestCheck(BaseModel):
    interested_in: bool = Field(..., description="興味がありそうな内容かどうか")


class Summary(BaseModel):
    title: str = Field(..., description="論文のタイトル")
    summary: str = Field(..., description="論文の概要")
    keywords: List[str] = Field(..., description="論文のキーワード")
    appendix: Optional[str] = Field(None, description="補足情報")


class ReadingMemo(BaseModel):
    conclusion: str = Field(..., description="30秒で分かる結論")
    main_claims: str = Field(..., description="主定理・主張")
    method_outline: str = Field(..., description="証明・手法の骨格")
    research_connection: str = Field(..., description="興味分野との接点")
    reading_guide: str = Field(..., description="読むならここ")
    follow_up_questions: List[str] = Field(
        ..., min_length=3, max_length=3, description="次に尋ねるとよい質問"
    )


prompt_check_interest = ""
with open("src/prompt_check_interest.txt", "r", encoding="utf-8") as f:
    prompt_check_interest = f.read()

prompt_summarize = ""
with open("src/prompt_summarize.txt", "r", encoding="utf-8") as f:
    prompt_summarize = f.read()

prompt_reading_memo = ""
with open("src/prompt_reading_memo.txt", "r", encoding="utf-8") as f:
    prompt_reading_memo = f.read()


def now_iso_utc() -> str:
    return datetime.datetime.now(ZoneInfo("UTC")).isoformat()


def parse_iso_datetime(value: str) -> Optional[datetime.datetime]:
    if not value:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=ZoneInfo("UTC"))
    return parsed


def is_older_than_hours(value: str, threshold_hours: int) -> bool:
    parsed = parse_iso_datetime(value)
    if parsed is None:
        return False
    now = datetime.datetime.now(ZoneInfo("UTC"))
    return (now - parsed) >= datetime.timedelta(hours=threshold_hours)


def mark_job_updated(job: dict) -> None:
    job["updated_at"] = now_iso_utc()


def ensure_state_file() -> None:
    parent_dir = os.path.dirname(STATE_FILE_PATH)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    if not os.path.exists(STATE_FILE_PATH):
        with open(STATE_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump({"schema_version": STATE_SCHEMA_VERSION, "jobs": []}, f, ensure_ascii=False, indent=2)


def load_state() -> dict:
    ensure_state_file()
    with open(STATE_FILE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)

    if not isinstance(state, dict):
        return {"schema_version": STATE_SCHEMA_VERSION, "jobs": []}
    if "schema_version" not in state:
        state["schema_version"] = STATE_SCHEMA_VERSION
    if "jobs" not in state or not isinstance(state["jobs"], list):
        state["jobs"] = []
    return state


def save_state(state: dict) -> None:
    with open(STATE_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def search_papers():
    # search for papers submitted yesterday
    yesterday = datetime.datetime.now(ZoneInfo("America/New_York")) - datetime.timedelta(days=3)
    search_start = yesterday.strftime("%Y%m%d0000")
    search_end = yesterday.strftime("%Y%m%d2359")
    print(f"Searching papers from {search_start} to {search_end}")

    search = arxiv.Search(
        query=f"(cat:math.DS OR cat:math.CO OR cat:math.GR OR cat:cs.LO OR cat:cs.FL OR cat:cs.DM) AND submittedDate:[{search_start} TO {search_end}]",
        max_results=None,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    results = client_arxiv.results(search)
    return results


def serialize_paper(result: arxiv.Result) -> dict:
    published = result.published.isoformat() if result.published else None
    return {
        "paper_id": result.entry_id,
        "entry_id": result.entry_id,
        "pdf_url": result.pdf_url,
        "title": result.title,
        "summary": result.summary,
        "authors": [str(author) for author in result.authors],
        "published": published,
    }


def submit_interest_batch(papers: List[dict]) -> str:
    if len(papers) == 0:
        return ""

    inline_request: List[dict] = []
    for paper in papers:
        title = f"\nTitle: {paper['title']}\n"
        abstract = f"\nAbstract: {paper['summary']}\n"
        request_item = {
            "contents": [{"parts": [{"text": title + abstract + prompt_check_interest}]}],
            "config": {
                "response_mime_type": "application/json",
                "response_schema": InterestCheck,
            },
        }
        inline_request.append(request_item)

    batch_job = client_genai.batches.create(
        model=INTEREST_MODEL,
        src=inline_request,
        config={"display_name": "Interest Check Batch Job"},
    )
    print(f"Interest batch job created: {batch_job.name}")
    print(f"Number of papers in batch: {len(inline_request)}")
    return batch_job.name


def submit_summary_batch(papers: List[dict]) -> str:
    if len(papers) == 0:
        return ""

    inline_request: List[dict] = []
    for paper in papers:
        title = f"\nTitle: {paper['title']}\n"
        abstract = f"\nAbstract: {paper['summary']}\n"
        request_item = {
            "contents": [{"parts": [{"text": title + abstract + prompt_summarize}]}],
            "config": {
                "response_mime_type": "application/json",
                "response_schema": Summary,
                "thinking_config": {"thinking_level": "low"},
            },
        }
        inline_request.append(request_item)

    batch_job = client_genai.batches.create(
        model=SUMMARY_MODEL,
        src=inline_request,
        config={"display_name": "Summarize Paper Batch Job"},
    )
    print(f"Summary batch job created: {batch_job.name}")
    print(f"Number of papers in batch: {len(inline_request)}")
    return batch_job.name


def poll_batch_once(batch_name: str):
    if not batch_name:
        return None
    batch_job = client_genai.batches.get(name=batch_name)
    print(f"Batch {batch_name}: {batch_job.state.name}")
    return batch_job


def cancel_batch_safely(batch_name: str) -> bool:
    if not batch_name:
        return False
    try:
        cancel_method = getattr(client_genai.batches, "cancel", None)
        if not callable(cancel_method):
            print("Batch cancel API is not available in current SDK.")
            return False
        cancel_method(name=batch_name)
        print(f"Batch cancel requested: {batch_name}")
        return True
    except Exception as exc:
        print(f"Batch cancel failed for {batch_name}: {exc}")
        return False


def short_error(error: object, limit: int = 300) -> str:
    message = str(error) if error is not None else "unknown error"
    return message if len(message) <= limit else message[: limit - 1] + "…"


def format_item_errors(prefix: str, errors: Dict[str, str], limit: int = 3) -> str:
    examples = list(errors.items())[:limit]
    details = "; ".join(f"{paper_id}: {message}" for paper_id, message in examples)
    remaining = len(errors) - len(examples)
    if remaining > 0:
        details += f"; and {remaining} more"
    return f"{prefix} ({len(errors)} item(s)): {details}"


def check_interest_sequential_papers(
    papers: List[dict], existing_results: Optional[Dict[str, bool]] = None
) -> Tuple[Dict[str, bool], Dict[str, str]]:
    print("Checking interest sequentially...")
    interest_results = dict(existing_results or {})
    errors: Dict[str, str] = {}
    for i, paper in enumerate(papers):
        paper_id = paper["paper_id"]
        if paper_id in interest_results:
            continue

        title = f"\nTitle: {paper['title']}\n"
        abstract = f"\nAbstract: {paper['summary']}\n"
        try:
            response = client_genai.models.generate_content(
                model=INTEREST_MODEL,
                contents=title + abstract + prompt_check_interest,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": InterestCheck,
                },
            )
            is_interest = InterestCheck.model_validate_json(response.text)
            interest_results[paper_id] = is_interest.interested_in
            print(f"Result for paper {i + 1}: Interested: {is_interest.interested_in}")
        except Exception as exc:
            errors[paper_id] = short_error(exc)
            print(f"Interest retry failed for {paper_id}: {errors[paper_id]}")
    return interest_results, errors


def summarize_sequential_papers(
    papers: List[dict], existing_summaries: dict
) -> Tuple[dict, Dict[str, str]]:
    print("Summarizing papers sequentially...")
    summaries = dict(existing_summaries)
    errors: Dict[str, str] = {}
    for i, paper in enumerate(papers):
        paper_id = paper["paper_id"]
        if paper_id in summaries:
            continue

        title = f"\nTitle: {paper['title']}\n"
        abstract = f"\nAbstract: {paper['summary']}\n"
        try:
            response = client_genai.models.generate_content(
                model=SUMMARY_MODEL,
                contents=title + abstract + prompt_summarize,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": Summary,
                    "thinking_config": {"thinking_level": "low"},
                },
            )
            summary = Summary.model_validate_json(response.text)
            summaries[paper_id] = {
                "title": summary.title,
                "summary": summary.summary,
                "keywords": summary.keywords,
                "appendix": summary.appendix,
            }
            print(f"Result for paper {i + 1}: summarized {paper_id}")
        except Exception as exc:
            errors[paper_id] = short_error(exc)
            print(f"Summary retry failed for {paper_id}: {errors[paper_id]}")
    return summaries, errors


def pdf_url_for_paper(paper: dict) -> str:
    pdf_url = paper.get("pdf_url") or paper.get("entry_id", "").replace("/abs/", "/pdf/")
    return pdf_url.replace("http://arxiv.org/", "https://arxiv.org/", 1)


def download_pdf(paper: dict) -> bytes:
    pdf_url = pdf_url_for_paper(paper)
    if not pdf_url:
        raise ValueError("paper PDF URL is missing")

    response = requests.get(
        pdf_url,
        headers={"User-Agent": "discord-arxiv-bot/reading-memo"},
        stream=True,
        timeout=(DISCORD_CONNECT_TIMEOUT_SECONDS, 60),
    )
    response.raise_for_status()
    content_length_header = response.headers.get("content-length", "")
    content_length = int(content_length_header) if content_length_header.isdigit() else 0
    if content_length > MAX_PDF_BYTES:
        raise ValueError(f"PDF is larger than {MAX_PDF_BYTES // (1024 * 1024)} MiB")

    chunks = []
    size = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        size += len(chunk)
        if size > MAX_PDF_BYTES:
            raise ValueError(f"PDF is larger than {MAX_PDF_BYTES // (1024 * 1024)} MiB")
        chunks.append(chunk)
    pdf_data = b"".join(chunks)
    if not pdf_data.lstrip().startswith(b"%PDF"):
        raise ValueError("arXiv response is not a PDF")
    return pdf_data


def generate_reading_memo(paper: dict) -> dict:
    print(f"Reading full PDF: {paper['paper_id']}")
    uploaded_file = client_genai.files.upload(
        file=io.BytesIO(download_pdf(paper)),
        config={"mime_type": "application/pdf"},
    )
    try:
        response = client_genai.models.generate_content(
            model=READING_MODEL,
            contents=[
                uploaded_file,
                f"Title: {paper['title']}\nURL: {paper['entry_id']}\n\n{prompt_reading_memo}",
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": ReadingMemo,
                "thinking_config": {"thinking_level": "medium"},
            },
        )
        return ReadingMemo.model_validate_json(response.text).model_dump()
    finally:
        try:
            client_genai.files.delete(name=uploaded_file.name)
        except Exception as exc:
            print(f"Failed to delete temporary Gemini file: {short_error(exc)}")


def extract_interest_check(batch_job, papers: List[dict]) -> Tuple[Dict[str, bool], Dict[str, str]]:
    interest_results: Dict[str, bool] = {}
    errors: Dict[str, str] = {}
    inline_responses = getattr(getattr(batch_job, "dest", None), "inlined_responses", None) or []
    for i, paper in enumerate(papers):
        paper_id = paper["paper_id"]
        if i >= len(inline_responses):
            errors[paper_id] = "missing batch response"
            continue

        inline_response = inline_responses[i]
        response = getattr(inline_response, "response", None)
        if response is None:
            errors[paper_id] = short_error(getattr(inline_response, "error", "missing batch response"))
            continue

        try:
            is_interest = InterestCheck.model_validate_json(response.text)
            interest_results[paper_id] = is_interest.interested_in
        except Exception as exc:
            errors[paper_id] = short_error(exc)
    return interest_results, errors


def extract_summaries(batch_job, papers: List[dict]) -> Tuple[dict, Dict[str, str]]:
    summaries = {}
    errors: Dict[str, str] = {}
    inline_responses = getattr(getattr(batch_job, "dest", None), "inlined_responses", None) or []
    for i, paper in enumerate(papers):
        paper_id = paper["paper_id"]
        if i >= len(inline_responses):
            errors[paper_id] = "missing batch response"
            continue

        inline_response = inline_responses[i]
        response = getattr(inline_response, "response", None)
        if response is None:
            errors[paper_id] = short_error(getattr(inline_response, "error", "missing batch response"))
            continue

        try:
            summary = Summary.model_validate_json(response.text)
            summaries[paper_id] = {
                "title": summary.title,
                "summary": summary.summary,
                "keywords": summary.keywords,
                "appendix": summary.appendix,
            }
        except Exception as exc:
            errors[paper_id] = short_error(exc)
    return summaries, errors


def truncate_discord_text(value: object, limit: int, fallback: str = "（なし）") -> str:
    text = str(value) if value is not None else ""
    if not text:
        text = fallback
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def discord_embed_text_length(embed: dict) -> int:
    total = len(str(embed.get("title", ""))) + len(str(embed.get("description", "")))
    total += len(str(embed.get("author", {}).get("name", "")))
    total += len(str(embed.get("footer", {}).get("text", "")))
    for field in embed.get("fields", []):
        total += len(str(field.get("name", ""))) + len(str(field.get("value", "")))
    return total


def fit_discord_embed_total_limit(embed: dict) -> None:
    overflow = discord_embed_text_length(embed) - DISCORD_EMBED_TOTAL_LIMIT
    if overflow <= 0:
        return

    for field in reversed(embed.get("fields", [])):
        value = str(field.get("value", ""))
        removable = max(0, len(value) - 1)
        if removable == 0:
            continue
        remove_count = min(removable, overflow)
        field["value"] = truncate_discord_text(value, len(value) - remove_count)
        overflow -= remove_count
        if overflow <= 0:
            return


def discord_retry_delay(response, attempt: int) -> float:
    if response is not None and response.status_code == 429:
        try:
            retry_after = float(response.json().get("retry_after", 0))
            if retry_after > 0:
                return min(retry_after, 30.0)
        except (TypeError, ValueError, AttributeError, json.JSONDecodeError):
            pass
    return DISCORD_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))


def post_discord_payload(
    webhook_url: str, payload: dict, description: str, wait: bool = False
) -> Union[bool, dict]:
    for attempt in range(1, DISCORD_MAX_ATTEMPTS + 1):
        response = None
        try:
            response = requests.post(
                webhook_url,
                json=payload,
                params={"wait": "true"} if wait else None,
                timeout=(DISCORD_CONNECT_TIMEOUT_SECONDS, DISCORD_READ_TIMEOUT_SECONDS),
            )
            if response.status_code in (200, 204):
                if wait and response.status_code == 200:
                    return response.json()
                return True
            retryable = response.status_code == 429 or 500 <= response.status_code < 600
            body = truncate_discord_text(response.text, 300, fallback="")
            print(
                f"Failed to send {description} to Discord "
                f"(attempt {attempt}/{DISCORD_MAX_ATTEMPTS}). "
                f"Status: {response.status_code}, Body: {body}"
            )
            if not retryable:
                return False
        except requests.RequestException as exc:
            print(
                f"Failed to send {description} to Discord "
                f"(attempt {attempt}/{DISCORD_MAX_ATTEMPTS}): {short_error(exc)}"
            )

        if attempt < DISCORD_MAX_ATTEMPTS:
            time.sleep(discord_retry_delay(response, attempt))
    return False


def post_summary_to_discord(webhook_url: str, paper: dict, summary: dict) -> Optional[dict]:
    authors = truncate_discord_text(", ".join(paper["authors"]), DISCORD_EMBED_FIELD_VALUE_LIMIT)
    embed = {
        "author": {
            "name": truncate_discord_text("arXiv", DISCORD_EMBED_AUTHOR_NAME_LIMIT),
            "url": "https://arxiv.org/",
            "icon_url": "https://shuyaojiang.github.io/assets/images/badges/arXiv.png",
        },
        "title": truncate_discord_text(summary["title"], DISCORD_EMBED_TITLE_LIMIT),
        "url": paper["entry_id"],
        "color": 0xE12D2D,
        "timestamp": datetime.datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        "fields": [
            {
                "name": truncate_discord_text("著者", DISCORD_EMBED_FIELD_NAME_LIMIT),
                "value": authors,
                "inline": False,
            },
            {
                "name": truncate_discord_text("概要", DISCORD_EMBED_FIELD_NAME_LIMIT),
                "value": truncate_discord_text(summary["summary"], DISCORD_EMBED_FIELD_VALUE_LIMIT),
                "inline": False,
            },
        ],
        "thumbnail": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/7/7a/ArXiv_logo_2022.png"
        },
        "footer": {
            "text": truncate_discord_text("arXiv Summarizer", DISCORD_EMBED_FOOTER_TEXT_LIMIT),
            "icon_url": "https://cdn.discordapp.com/embed/avatars/4.png",
        },
    }

    if summary.get("appendix"):
        embed["fields"].append(
            {
                "name": truncate_discord_text("補足情報", DISCORD_EMBED_FIELD_NAME_LIMIT),
                "value": truncate_discord_text(summary["appendix"], DISCORD_EMBED_FIELD_VALUE_LIMIT),
                "inline": False,
            }
        )

    keywords = summary.get("keywords", [])
    embed["fields"].append(
        {
            "name": truncate_discord_text("keywords", DISCORD_EMBED_FIELD_NAME_LIMIT),
            "value": truncate_discord_text(", ".join(keywords), DISCORD_EMBED_FIELD_VALUE_LIMIT),
            "inline": False,
        }
    )
    fit_discord_embed_total_limit(embed)

    message = {"embeds": [embed]}
    result = post_discord_payload(webhook_url, message, f"paper {paper['paper_id']}", wait=True)
    if isinstance(result, dict) and result.get("id") and result.get("channel_id"):
        print(f"Sent paper: {paper['title']}")
        return result
    print(f"Discord did not return the message ID for {paper['paper_id']}")
    return None


def discord_bot_request(
    method: str,
    path: str,
    bot_token: str,
    description: str,
    payload: Optional[dict] = None,
    params: Optional[dict] = None,
    success_codes: Tuple[int, ...] = (200, 204),
    max_attempts: int = DISCORD_MAX_ATTEMPTS,
) -> Optional[object]:
    token = bot_token.removeprefix("Bot ").strip()
    if not token:
        print(f"DISCORD_BOT_TOKEN is missing; cannot {description}.")
        return None

    for attempt in range(1, max_attempts + 1):
        response = None
        try:
            response = requests.request(
                method,
                f"{DISCORD_API_BASE_URL}{path}",
                headers={"Authorization": f"Bot {token}"},
                json=payload,
                params=params,
                timeout=(DISCORD_CONNECT_TIMEOUT_SECONDS, DISCORD_READ_TIMEOUT_SECONDS),
            )
            if response.status_code in success_codes:
                return True if response.status_code == 204 else response.json()
            retryable = response.status_code == 429 or 500 <= response.status_code < 600
            body = truncate_discord_text(response.text, 300, fallback="")
            print(
                f"Failed to {description} (attempt {attempt}/{max_attempts}). "
                f"Status: {response.status_code}, Body: {body}"
            )
            if not retryable:
                return None
        except requests.RequestException as exc:
            print(
                f"Failed to {description} (attempt {attempt}/{max_attempts}): "
                f"{short_error(exc)}"
            )

        if attempt < max_attempts:
            time.sleep(discord_retry_delay(response, attempt))
    return None


def add_read_reaction(bot_token: str, channel_id: str, message_id: str) -> bool:
    result = discord_bot_request(
        "PUT",
        f"/channels/{channel_id}/messages/{message_id}/reactions/{quote(READ_EMOJI)}/@me",
        bot_token,
        f"add {READ_EMOJI} reaction to message {message_id}",
    )
    return result is not None


def reaction_users_include_request(users: List[dict], discord_user_id: str = "") -> bool:
    people = [user for user in users if not user.get("bot", False)]
    if discord_user_id:
        return any(str(user.get("id")) == discord_user_id for user in people)
    return bool(people)


def has_read_request(
    bot_token: str, channel_id: str, message_id: str, discord_user_id: str = ""
) -> bool:
    users = discord_bot_request(
        "GET",
        f"/channels/{channel_id}/messages/{message_id}/reactions/{quote(READ_EMOJI)}",
        bot_token,
        f"get {READ_EMOJI} reactions for message {message_id}",
        params={"limit": 100},
    )
    return isinstance(users, list) and reaction_users_include_request(users, discord_user_id)


def build_reading_memo_embed(paper: dict, memo: dict) -> dict:
    questions = "\n".join(f"- {question}" for question in memo["follow_up_questions"])
    embed = {
        "title": truncate_discord_text(paper["title"], DISCORD_EMBED_TITLE_LIMIT),
        "url": paper["entry_id"],
        "color": 0x5865F2,
        "fields": [
            {"name": "30秒で分かる結論", "value": memo["conclusion"], "inline": False},
            {"name": "主定理・主張", "value": memo["main_claims"], "inline": False},
            {"name": "証明・手法の骨格", "value": memo["method_outline"], "inline": False},
            {"name": "研究との接点", "value": memo["research_connection"], "inline": False},
            {"name": "読むならここ", "value": memo["reading_guide"], "inline": False},
            {"name": "次に尋ねるとよい質問", "value": questions, "inline": False},
        ],
        "footer": {"text": "arXiv full-paper reading memo"},
        "timestamp": datetime.datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
    }
    for field in embed["fields"]:
        field["value"] = truncate_discord_text(field["value"], 880)
    fit_discord_embed_total_limit(embed)
    return embed


def post_reading_memo_to_forum(
    bot_token: str, forum_channel_id: str, paper: dict, memo: dict
) -> Optional[dict]:
    result = discord_bot_request(
        "POST",
        f"/channels/{forum_channel_id}/threads",
        bot_token,
        f"create Forum post for {paper['paper_id']}",
        payload={
            "name": truncate_discord_text(paper["title"], 100),
            "message": {
                "embeds": [build_reading_memo_embed(paper, memo)],
                "allowed_mentions": {"parse": []},
            },
        },
        success_codes=(200, 201),
        max_attempts=1,
    )
    return result if isinstance(result, dict) and result.get("id") else None


def run_stage_enqueue_interest() -> int:
    search_results = list(search_papers())
    if len(search_results) == 0:
        print("No papers found, exiting.")
        return 0

    papers = [serialize_paper(paper) for paper in search_results]
    interest_job_name = submit_interest_batch(papers)
    if not interest_job_name:
        print("Failed to create interest batch job.")
        return 1

    state = load_state()
    now = now_iso_utc()
    pipeline_id = f"{datetime.datetime.now(ZoneInfo('UTC')).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    state["jobs"].append(
        {
            "pipeline_id": pipeline_id,
            "status": "interest_submitted",
            "interest_job_name": interest_job_name,
            "summarize_job_name": None,
            "papers": papers,
            "interest_results": {},
            "interested_paper_ids": [],
            "summaries": {},
            "sent_paper_ids": [],
            "discord_messages": {},
            "reading_memos": {},
            "notification_sent": False,
            "retry_count": 0,
            "last_error": None,
            "created_at": now,
            "updated_at": now,
            "finalized_at": None,
        }
    )
    save_state(state)
    print(f"Queued pipeline: {pipeline_id}")
    return 0


def run_stage_poll_interest_submit_summary() -> int:
    state = load_state()
    updated = False

    for job in state["jobs"]:
        if job.get("status") not in ("interest_submitted", "interest_running", "interest_fallback_running"):
            continue

        papers = job.get("papers", [])
        interest_results = dict(job.get("interest_results", {}))
        job["interest_results"] = interest_results
        is_timeout = is_older_than_hours(job.get("created_at", ""), BATCH_TIMEOUT_HOURS)

        if job.get("status") != "interest_fallback_running":
            batch_job = poll_batch_once(job.get("interest_job_name", ""))
            if not batch_job:
                continue

            batch_state = batch_job.state.name
            if batch_state not in COMPLETED_BATCH_STATUS:
                if is_timeout:
                    cancel_ok = cancel_batch_safely(job.get("interest_job_name", ""))
                    job["status"] = "interest_fallback_running"
                    job["last_error"] = None if cancel_ok else "interest timeout reached, cancel request failed"
                    mark_job_updated(job)
                    updated = True
                elif job.get("status") != "interest_running":
                    job["status"] = "interest_running"
                    mark_job_updated(job)
                    updated = True
                if not is_timeout:
                    continue

            elif batch_state != "JOB_STATE_SUCCEEDED":
                job["status"] = "interest_fallback_running"
                job["last_error"] = f"interest batch ended with {batch_state}"
                mark_job_updated(job)
                updated = True
            else:
                extracted, batch_errors = extract_interest_check(batch_job, papers)
                interest_results.update(extracted)
                job["interest_results"] = interest_results
                if batch_errors:
                    job["status"] = "interest_fallback_running"
                    job["last_error"] = format_item_errors("interest batch item failed", batch_errors)
                mark_job_updated(job)
                updated = True

        missing_papers = [paper for paper in papers if paper["paper_id"] not in interest_results]
        if missing_papers:
            interest_results, retry_errors = check_interest_sequential_papers(
                missing_papers, interest_results
            )
            job["interest_results"] = interest_results
            updated = True
            still_missing = [
                paper["paper_id"] for paper in papers if paper["paper_id"] not in interest_results
            ]
            if still_missing:
                job["status"] = "interest_fallback_running"
                job["retry_count"] = int(job.get("retry_count", 0)) + 1
                job["last_error"] = format_item_errors("interest retry failed", retry_errors)
                mark_job_updated(job)
                continue

        interested_ids = [
            paper["paper_id"] for paper in papers if interest_results.get(paper["paper_id"]) is True
        ]
        job["interested_paper_ids"] = interested_ids
        updated = True

        if len(interested_ids) == 0:
            job["status"] = "completed_no_interests"
            job["finalized_at"] = now_iso_utc()
            job["last_error"] = None
            mark_job_updated(job)
            continue

        interested_set = set(interested_ids)
        interested_papers = [paper for paper in papers if paper["paper_id"] in interested_set]
        try:
            summarize_job_name = submit_summary_batch(interested_papers)
            if not summarize_job_name:
                raise RuntimeError("summary batch creation returned an empty job name")
            job["summarize_job_name"] = summarize_job_name
            job["status"] = "summarize_submitted"
            job["last_error"] = None
            mark_job_updated(job)
            updated = True
        except Exception as exc:
            job["status"] = "interest_fallback_running"
            job["retry_count"] = int(job.get("retry_count", 0)) + 1
            job["last_error"] = f"summary batch submission failed: {short_error(exc)}"
            mark_job_updated(job)
            updated = True

    if updated:
        save_state(state)
    else:
        print("No interest jobs updated.")
    return 0


def run_stage_poll_summary_send() -> int:
    discord_webhook_url = os.getenv("ARXIV_RECOMMENDER_WEBHOOK_URL")
    discord_bot_token = os.getenv("DISCORD_BOT_TOKEN", "")
    if not discord_webhook_url:
        print("ARXIV_RECOMMENDER_WEBHOOK_URL is not set.")
        return 1
    if not discord_bot_token:
        print("DISCORD_BOT_TOKEN is not set.")
        return 1

    state = load_state()
    updated = False

    for job in state["jobs"]:
        if job.get("status") not in (
            "summarize_submitted",
            "summarize_running",
            "summary_fallback_running",
            "send_failed",
        ):
            continue

        job["summaries"] = dict(job.get("summaries", {}))
        job["sent_paper_ids"] = list(job.get("sent_paper_ids", []))
        job["discord_messages"] = dict(job.get("discord_messages", {}))

        if job.get("status") in ("summarize_submitted", "summarize_running"):
            timeout_anchor = job.get("updated_at") or job.get("created_at", "")
            is_timeout = is_older_than_hours(timeout_anchor, BATCH_TIMEOUT_HOURS)

            batch_job = poll_batch_once(job.get("summarize_job_name", ""))
            if not batch_job:
                continue

            batch_state = batch_job.state.name
            if batch_state not in COMPLETED_BATCH_STATUS:
                if is_timeout:
                    cancel_ok = cancel_batch_safely(job.get("summarize_job_name", ""))
                    job["status"] = "summary_fallback_running"
                    job["last_error"] = None if cancel_ok else "summary timeout reached, cancel request failed"
                    mark_job_updated(job)
                    updated = True
                elif job.get("status") != "summarize_running":
                    job["status"] = "summarize_running"
                    mark_job_updated(job)
                    updated = True
                if not is_timeout:
                    continue

            elif batch_state != "JOB_STATE_SUCCEEDED":
                job["status"] = "summary_fallback_running"
                job["last_error"] = f"summary batch ended with {batch_state}"
                mark_job_updated(job)
                updated = True
            else:
                interested_set = set(job.get("interested_paper_ids", []))
                interested_papers = [
                    paper
                    for paper in job.get("papers", [])
                    if paper["paper_id"] in interested_set
                ]
                extracted, batch_errors = extract_summaries(batch_job, interested_papers)
                job["summaries"].update(extracted)
                if batch_errors:
                    job["status"] = "summary_fallback_running"
                    job["last_error"] = format_item_errors("summary batch item failed", batch_errors)
                mark_job_updated(job)
                updated = True

        interested_ids = job.get("interested_paper_ids", [])
        papers_by_id = {paper["paper_id"]: paper for paper in job.get("papers", [])}
        missing_ids = [paper_id for paper_id in interested_ids if paper_id not in job["summaries"]]
        if missing_ids:
            job["status"] = "summary_fallback_running"
            retry_errors = {
                paper_id: "paper metadata is missing"
                for paper_id in missing_ids
                if paper_id not in papers_by_id
            }
            missing_papers = [papers_by_id[paper_id] for paper_id in missing_ids if paper_id in papers_by_id]
            summaries, generated_errors = summarize_sequential_papers(
                missing_papers, job["summaries"]
            )
            retry_errors.update(generated_errors)
            job["summaries"] = summaries
            updated = True

            still_missing = [paper_id for paper_id in interested_ids if paper_id not in summaries]
            if still_missing:
                for paper_id in still_missing:
                    retry_errors.setdefault(paper_id, "summary was not generated")
                job["retry_count"] = int(job.get("retry_count", 0)) + 1
                job["last_error"] = format_item_errors("summary retry failed", retry_errors)
                mark_job_updated(job)
                continue
            else:
                job["last_error"] = None
                mark_job_updated(job)

        pending_ids = [
            paper_id
            for paper_id in interested_ids
            if paper_id not in set(job["sent_paper_ids"])
        ]

        if len(pending_ids) == 0:
            if len(interested_ids) == len(job["sent_paper_ids"]):
                job["status"] = "completed"
                job["finalized_at"] = now_iso_utc()
                job["last_error"] = None
                mark_job_updated(job)
                updated = True
            continue

        if not job.get("notification_sent", False):
            content = truncate_discord_text(
                f"新しい論文が見つかったぞ。目は通せよ（{len(pending_ids)}件）",
                DISCORD_CONTENT_LIMIT,
            )
            if post_discord_payload(discord_webhook_url, {"content": content}, "notification"):
                print("Notification sent successfully to Discord.")
                job["notification_sent"] = True
                job["last_error"] = None
                mark_job_updated(job)
                updated = True
            else:
                job["status"] = "send_failed"
                job["last_error"] = "failed to send notification message"
                mark_job_updated(job)
                updated = True
                continue

        all_success = True
        for paper_id in pending_ids:
            paper = papers_by_id.get(paper_id)
            summary = job["summaries"].get(paper_id)
            if paper is None or summary is None:
                all_success = False
                continue

            message_state = job["discord_messages"].get(paper_id)
            if not message_state:
                message = post_summary_to_discord(discord_webhook_url, paper, summary)
                if message:
                    message_state = {
                        "message_id": message["id"],
                        "channel_id": message["channel_id"],
                        "reaction_added": False,
                        "read_requested": False,
                        "reading_memo_sent": False,
                        "paper_thread_id": None,
                    }
                    job["discord_messages"][paper_id] = message_state
                    updated = True

            if message_state and (
                message_state.get("reaction_added")
                or add_read_reaction(
                    discord_bot_token,
                    message_state["channel_id"],
                    message_state["message_id"],
                )
            ):
                message_state["reaction_added"] = True
                if paper_id not in job["sent_paper_ids"]:
                    job["sent_paper_ids"].append(paper_id)
                    updated = True
            else:
                all_success = False
            time.sleep(1.5)

        if all_success and len(job["sent_paper_ids"]) == len(interested_ids):
            job["status"] = "completed"
            job["finalized_at"] = now_iso_utc()
            job["last_error"] = None
            mark_job_updated(job)
            updated = True
        elif not all_success:
            job["status"] = "send_failed"
            job["retry_count"] = int(job.get("retry_count", 0)) + 1
            job["last_error"] = "failed to send one or more paper summaries"
            mark_job_updated(job)
            updated = True

    if updated:
        save_state(state)
    else:
        print("No summary jobs updated.")
    return 0


def run_stage_poll_reading_requests() -> int:
    discord_bot_token = os.getenv("DISCORD_BOT_TOKEN", "")
    forum_channel_id = os.getenv("DISCORD_FORUM_CHANNEL_ID", "")
    discord_user_id = os.getenv("DISCORD_USER_ID", "").strip()
    if not discord_bot_token:
        print("DISCORD_BOT_TOKEN is not set.")
        return 1
    if not forum_channel_id:
        print("DISCORD_FORUM_CHANNEL_ID is not set.")
        return 1

    state = load_state()
    updated = False
    for job in state["jobs"]:
        messages = job.get("discord_messages", {})
        if not messages:
            continue

        papers_by_id = {paper["paper_id"]: paper for paper in job.get("papers", [])}
        job["reading_memos"] = dict(job.get("reading_memos", {}))
        for paper_id, message_state in messages.items():
            if message_state.get("reading_memo_sent"):
                continue

            if not message_state.get("read_requested"):
                requested = has_read_request(
                    discord_bot_token,
                    message_state["channel_id"],
                    message_state["message_id"],
                    discord_user_id,
                )
                if not requested:
                    continue
                message_state["read_requested"] = True
                message_state["read_requested_at"] = now_iso_utc()
                message_state["reading_last_error"] = None
                updated = True

            paper = papers_by_id.get(paper_id)
            if paper is None:
                message_state["reading_last_error"] = "paper metadata is missing"
                updated = True
                continue

            if paper_id not in job["reading_memos"]:
                try:
                    job["reading_memos"][paper_id] = generate_reading_memo(paper)
                    message_state["reading_last_error"] = None
                    updated = True
                except Exception as exc:
                    message_state["reading_retry_count"] = int(
                        message_state.get("reading_retry_count", 0)
                    ) + 1
                    message_state["reading_last_error"] = short_error(exc)
                    print(f"Reading memo generation failed for {paper_id}: {short_error(exc)}")
                    updated = True
                    continue

            forum_post = post_reading_memo_to_forum(
                discord_bot_token,
                forum_channel_id,
                paper,
                job["reading_memos"][paper_id],
            )
            if forum_post:
                message_state["reading_memo_sent"] = True
                message_state["paper_thread_id"] = forum_post["id"]
                message_state["reading_memo_sent_at"] = now_iso_utc()
                message_state["reading_last_error"] = None
                print(f"Created Forum post for {paper_id}: {forum_post['id']}")
            else:
                message_state["reading_retry_count"] = int(
                    message_state.get("reading_retry_count", 0)
                ) + 1
                message_state["reading_last_error"] = "failed to create Forum post"
            updated = True

    if updated:
        save_state(state)
    else:
        print("No reading requests updated.")
    return 0


def run_self_check() -> int:
    assert pdf_url_for_paper({"entry_id": "http://arxiv.org/abs/2608.12345"}) == (
        "https://arxiv.org/pdf/2608.12345"
    )
    assert not reaction_users_include_request([{"id": "bot", "bot": True}])
    assert reaction_users_include_request([{"id": "user", "bot": False}], "user")
    assert not reaction_users_include_request([{"id": "other", "bot": False}], "user")
    memo = {
        "conclusion": "a" * 2000,
        "main_claims": "b" * 2000,
        "method_outline": "c" * 2000,
        "research_connection": "d" * 2000,
        "reading_guide": "e" * 2000,
        "follow_up_questions": ["f" * 1000] * 3,
    }
    embed = build_reading_memo_embed(
        {"title": "title", "entry_id": "https://arxiv.org/abs/2608.12345"}, memo
    )
    assert discord_embed_text_length(embed) <= DISCORD_EMBED_TOTAL_LIMIT
    assert all(len(field["value"]) <= DISCORD_EMBED_FIELD_VALUE_LIMIT for field in embed["fields"])
    print("Self-check passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="arXiv summarizer pipeline")
    parser.add_argument(
        "--stage",
        choices=[
            "enqueue_interest",
            "poll_interest_submit_summary",
            "poll_summary_send",
            "poll_reading_requests",
            "self_check",
        ],
        default=os.getenv("PIPELINE_STAGE", "enqueue_interest"),
        help="Pipeline stage to execute",
    )
    args = parser.parse_args()

    if args.stage == "enqueue_interest":
        return run_stage_enqueue_interest()
    if args.stage == "poll_interest_submit_summary":
        return run_stage_poll_interest_submit_summary()
    if args.stage == "poll_summary_send":
        return run_stage_poll_summary_send()
    if args.stage == "poll_reading_requests":
        return run_stage_poll_reading_requests()
    if args.stage == "self_check":
        return run_self_check()

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
