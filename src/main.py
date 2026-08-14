from google import genai
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple
import arxiv
import time
import os
import datetime
import requests
import json
import argparse
import uuid
from zoneinfo import ZoneInfo

# set up clients for arXiv, GenAI, and Discord
client_arxiv = arxiv.Client()
client_genai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

STATE_FILE_PATH = os.getenv("PENDING_JOBS_FILE", "state/pending_jobs.json")
STATE_SCHEMA_VERSION = 1
INTEREST_MODEL = os.getenv("INTEREST_MODEL", "gemini-3.5-flash-lite")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gemini-3.6-flash")
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


prompt_check_interest = ""
with open("src/prompt_check_interest.txt", "r", encoding="utf-8") as f:
    prompt_check_interest = f.read()

prompt_summarize = ""
with open("src/prompt_summarize.txt", "r", encoding="utf-8") as f:
    prompt_summarize = f.read()


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


def post_discord_payload(webhook_url: str, payload: dict, description: str) -> bool:
    for attempt in range(1, DISCORD_MAX_ATTEMPTS + 1):
        response = None
        try:
            response = requests.post(
                webhook_url,
                json=payload,
                timeout=(DISCORD_CONNECT_TIMEOUT_SECONDS, DISCORD_READ_TIMEOUT_SECONDS),
            )
            if response.status_code == 204:
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


def post_summary_to_discord(webhook_url: str, paper: dict, summary: dict) -> bool:
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
    if post_discord_payload(webhook_url, message, f"paper {paper['paper_id']}"):
        print(f"Sent paper: {paper['title']}")
        return True
    return False


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
    if not discord_webhook_url:
        print("ARXIV_RECOMMENDER_WEBHOOK_URL is not set.")
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
            is_sent = post_summary_to_discord(discord_webhook_url, paper, summary)
            if is_sent:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="arXiv summarizer pipeline")
    parser.add_argument(
        "--stage",
        choices=[
            "enqueue_interest",
            "poll_interest_submit_summary",
            "poll_summary_send",
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

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
