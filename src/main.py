"""arXiv recommender: 毎日プレプリントを検索し、興味判定・要約して Discord へ投稿し、
Forum のリーディングメモスレッドで対話形式の質問対応も行うポーリングパイプライン。

LLM 呼び出しは OpenAI 互換 API (OpenRouter, OpenAI, Gemini OpenAI 互換など) に限定し、
役割 (interest / summary / reading / qa) ごとに base_url・API キー・モデルを指定できる。
"""
import argparse
import datetime
import io
import json
import os
import time
import uuid
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple
from urllib.parse import quote
from zoneinfo import ZoneInfo

import arxiv
import requests
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

STATE_FILE_PATH = os.getenv("PENDING_JOBS_FILE", "").strip() or "state/pending_jobs.json"
STATE_SCHEMA_VERSION = 2
PROMPT_DIR = os.path.dirname(os.path.abspath(__file__))
DISCORD_API_BASE_URL = (os.getenv("DISCORD_API_BASE_URL") or "").strip() or "https://discord.com/api/v10"
DISCORD_WEBHOOK_URL = os.getenv("ARXIV_RECOMMENDER_WEBHOOK_URL", "")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
DISCORD_FORUM_CHANNEL_ID = os.getenv("DISCORD_FORUM_CHANNEL_ID", "")
DISCORD_USER_ID = os.getenv("DISCORD_USER_ID", "").strip()
READ_EMOJI = "📖"
MAX_PDF_BYTES = 50 * 1024 * 1024
READING_REQUEST_WINDOW_DAYS = int(os.getenv("READING_REQUEST_WINDOW_DAYS", "30"))
QA_MAX_REPLIES_PER_POLL = max(1, int(os.getenv("QA_MAX_REPLIES_PER_POLL", "3")))
QA_HISTORY_MESSAGES = max(1, int(os.getenv("QA_HISTORY_MESSAGES", "15")))
LLM_CALL_INTERVAL_SECONDS = float(os.getenv("LLM_CALL_INTERVAL_SECONDS", "0.5"))
SEARCH_CATEGORIES = os.getenv(
    "ARXIV_QUERY_CATEGORIES",
    "cat:math.DS OR cat:math.CO OR cat:math.GR OR cat:cs.LO OR cat:cs.FL OR cat:cs.DM",
)
try:
    SEARCH_LOOKBACK_DAYS = max(0, int(os.getenv("ARXIV_LOOKBACK_DAYS", "3")))
except ValueError:
    SEARCH_LOOKBACK_DAYS = 3
try:
    SEARCH_MAX_RESULTS = max(0, int(os.getenv("ARXIV_MAX_RESULTS", "0")))
except ValueError:
    SEARCH_MAX_RESULTS = 0


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
DISCORD_REPLY_LIMIT = 1900
DISCORD_EMBED_TITLE_LIMIT = 256
DISCORD_EMBED_FIELD_NAME_LIMIT = 256
DISCORD_EMBED_FIELD_VALUE_LIMIT = 1024
DISCORD_EMBED_AUTHOR_NAME_LIMIT = 256
DISCORD_EMBED_FOOTER_TEXT_LIMIT = 2048
DISCORD_EMBED_TOTAL_LIMIT = 6000

READING_MAX_PAPER_CHARS = int(os.getenv("READING_MAX_PAPER_CHARS", "200000"))
QA_MAX_PAPER_CHARS = int(os.getenv("QA_MAX_PAPER_CHARS", "200000"))


class InterestCheck(BaseModel):
    interested_in: bool = Field(..., description="興味がありそうな内容かどうか")
    reason: Optional[str] = Field(None, description="その判断の簡潔な理由")


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


def read_prompt(name: str) -> str:
    with io.open(os.path.join(PROMPT_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


PROMPT_CHECK_INTEREST = read_prompt("prompt_check_interest.txt")
PROMPT_SUMMARIZE = read_prompt("prompt_summarize.txt")
PROMPT_READING_MEMO = read_prompt("prompt_reading_memo.txt")
PROMPT_PAPER_QA = read_prompt("prompt_paper_qa.txt")

client_arxiv = arxiv.Client()


class LLMError(Exception):
    pass


def env_value(name: str) -> Optional[str]:
    value = (os.getenv(name) or "").strip()
    return value or None


def role_config(role: str) -> Dict[str, str]:
    """役割ごとの LLM 接続先。未設定の部分は LLM_API_BASE / LLM_API_KEY を使う。
    モデルのフォールバックは summary -> interest、reading -> summary、qa -> reading の順
    (interest は INTEREST_MODEL が必須)。"""
    base_url = env_value(f"{role.upper()}_API_BASE") or env_value("LLM_API_BASE")
    api_key = env_value(f"{role.upper()}_API_KEY") or env_value("LLM_API_KEY")
    fallback = {
        "interest": "INTEREST_MODEL",
        "summary": "SUMMARY_MODEL",
        "reading": "READING_MODEL",
        "qa": "QA_MODEL",
    }
    model_source = {"interest": None, "summary": "interest", "reading": "summary", "qa": "reading"}
    model = env_value(fallback[role])
    source = model_source[role]
    while not model and source:
        model = env_value(fallback[source])
        source = model_source[source]
    if not model:
        raise RuntimeError(f"{fallback[role]} is not set")
    if not base_url or not api_key:
        raise RuntimeError(f"LLM_API_BASE and LLM_API_KEY (or {role.upper()}_*) must be set for role {role}")
    return {"model": model, "base_url": base_url, "api_key": api_key}


_clients: Dict[str, OpenAI] = {}


def llm_client(role: str) -> OpenAI:
    if role not in _clients:
        cfg = role_config(role)
        _clients[role] = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    return _clients[role]


def extract_json_object(content: str) -> Optional[dict]:
    text = content.strip()
    if text.startswith("```"):
        inner = text.split("```")
        if len(inner) >= 3:
            text = inner[1]
            if text[:4].lower() == "json":
                text = text[4:]
            text = text.strip()
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except (ValueError, TypeError):
            return None
    return None


def json_instruction(model_cls: type) -> str:
    schema = json.dumps(model_cls.model_json_schema(), ensure_ascii=False)
    return f"出力は次の JSON スキーマに一致する単一の JSON オブジェクトのみとし、他の文章を一切書かないこと:\n{schema}"


def llm_structured(role: str, system: str, user: str, model_cls: type) -> BaseModel:
    client = llm_client(role)
    model = role_config(role)["model"]
    attempt_user = user
    last_error: Exception = ValueError("no response")
    for _ in range(2):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"{system}\n\n{json_instruction(model_cls)}".strip()},
                {"role": "user", "content": attempt_user},
            ],
        )
        try:
            if not response.choices:
                raise ValueError("response contains no choices")
            content = response.choices[0].message.content or ""
            data = extract_json_object(content)
            if data is None:
                raise ValueError("response contains no JSON object")
            return model_cls.model_validate(data)
        except (ValueError, ValidationError) as exc:
            last_error = exc
            attempt_user = (
                f"{user}\n\n前回の応答は無効でした: {exc}\n"
                "スキーマに一致する JSON オブジェクトのみを出力してください。"
            )
    raise LLMError(f"structured output failed for role {role}: {last_error}")


def llm_text(role: str, system: str, user: str) -> str:
    client = llm_client(role)
    model = role_config(role)["model"]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    if not response.choices:
        raise LLMError(f"model {model} returned no choices for role {role}")
    return (response.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# state


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


def is_older_than_days(value: str, threshold_days: int) -> bool:
    parsed = parse_iso_datetime(value)
    if parsed is None:
        return False
    now = datetime.datetime.now(ZoneInfo("UTC"))
    return (now - parsed) >= datetime.timedelta(days=threshold_days)


def mark_job_updated(job: dict) -> None:
    job["updated_at"] = now_iso_utc()


def ensure_state_file() -> None:
    parent_dir = os.path.dirname(STATE_FILE_PATH)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    if not os.path.exists(STATE_FILE_PATH):
        with io.open(STATE_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump({"schema_version": STATE_SCHEMA_VERSION, "jobs": []}, f, ensure_ascii=False, indent=2)


def load_state() -> dict:
    ensure_state_file()
    with io.open(STATE_FILE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)
    if not isinstance(state, dict):
        return {"schema_version": STATE_SCHEMA_VERSION, "jobs": []}
    state.setdefault("schema_version", STATE_SCHEMA_VERSION)
    if not isinstance(state.get("jobs"), list):
        state["jobs"] = []
    return state


def save_state(state: dict) -> None:
    tmp_path = f"{STATE_FILE_PATH}.tmp"
    with io.open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, STATE_FILE_PATH)


def search_window() -> str:
    ny_now = datetime.datetime.now(ZoneInfo("America/New_York")) - datetime.timedelta(
        days=SEARCH_LOOKBACK_DAYS
    )
    return ny_now.strftime("%Y%m%d")


def search_papers() -> List[dict]:
    window = search_window()
    search_start = f"{window}0000"
    search_end = f"{window}2359"
    print(f"Searching papers submitted on {window}")
    search = arxiv.Search(
        query=f"({SEARCH_CATEGORIES}) AND submittedDate:[{search_start} TO {search_end}]",
        max_results=SEARCH_MAX_RESULTS or None,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    return [serialize_paper(result) for result in client_arxiv.results(search)]


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


def paper_prompt(paper: dict) -> str:
    return f"Title: {paper['title']}\n\nAbstract: {paper['summary']}"


def new_pipeline(window: str, papers: List[dict]) -> dict:
    now = now_iso_utc()
    pipeline_id = f"{datetime.datetime.now(ZoneInfo('UTC')).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    return {
        "pipeline_id": pipeline_id,
        "window": window,
        "status": "processing",
        "papers": papers,
        "interest_results": {},
        "interest_reasons": {},
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


def interested_papers(job: dict) -> List[dict]:
    return [
        paper
        for paper in job["papers"]
        if job.get("interest_results", {}).get(paper["paper_id"]) is True
    ]


def pdf_url_for_paper(paper: dict) -> str:
    pdf_url = paper.get("pdf_url") or paper.get("entry_id", "").replace("/abs/", "/pdf/")
    return pdf_url.replace("http://arxiv.org/", "https://arxiv.org/", 1)


def download_pdf(paper: dict) -> bytes:
    pdf_url = pdf_url_for_paper(paper)
    if not pdf_url:
        raise ValueError("paper PDF URL is missing")
    with requests.get(
        pdf_url,
        headers={"User-Agent": "discord-arxiv-bot/reading-memo"},
        stream=True,
        timeout=(DISCORD_CONNECT_TIMEOUT_SECONDS, 60),
    ) as response:
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


def cap_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n…(省略)"


def paper_text_from_pdf_bytes(pdf_data: bytes, max_chars: int) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_data))
    parts: List[str] = []
    size = 0
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
        size += len(text) + 1
        # 上限に達したら以降のページ抽出を打ち切る
        if size > max_chars:
            break
    return cap_text("\n".join(parts), max_chars)


def extract_paper_text(paper: dict, max_chars: int) -> str:
    return paper_text_from_pdf_bytes(download_pdf(paper), max_chars)


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
) -> object:
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
                return response.json() if (wait and response.status_code == 200) else True
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
                f"(attempt {attempt}/{DISCORD_MAX_ATTEMPTS}): {exc}"
            )
        if attempt < DISCORD_MAX_ATTEMPTS:
            time.sleep(discord_retry_delay(response, attempt))
    return False


def short_error(error: object, limit: int = 300) -> str:
    message = str(error) if error is not None else "unknown error"
    return message if len(message) <= limit else message[: limit - 1] + "…"


def discord_bot_request(
    method: str,
    path: str,
    description: str,
    payload: Optional[dict] = None,
    params: Optional[dict] = None,
    success_codes: Tuple[int, ...] = (200, 204),
    max_attempts: int = DISCORD_MAX_ATTEMPTS,
) -> Optional[object]:
    token = DISCORD_BOT_TOKEN.removeprefix("Bot ").strip()
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
            print(f"Failed to {description} (attempt {attempt}/{max_attempts}): {exc}")
        if attempt < max_attempts:
            time.sleep(discord_retry_delay(response, attempt))
    return None


def add_read_reaction(channel_id: str, message_id: str) -> bool:
    result = discord_bot_request(
        "PUT",
        f"/channels/{channel_id}/messages/{message_id}/reactions/{quote(READ_EMOJI)}/@me",
        f"add {READ_EMOJI} reaction to message {message_id}",
    )
    return result is not None


def reaction_users_include_request(users: List[dict], discord_user_id: str = "") -> bool:
    people = [user for user in users if not user.get("bot", False)]
    if discord_user_id:
        return any(str(user.get("id")) == discord_user_id for user in people)
    return bool(people)


def has_read_request(channel_id: str, message_id: str) -> bool:
    users = discord_bot_request(
        "GET",
        f"/channels/{channel_id}/messages/{message_id}/reactions/{quote(READ_EMOJI)}",
        f"get {READ_EMOJI} reactions for message {message_id}",
        params={"limit": 100},
    )
    return isinstance(users, list) and reaction_users_include_request(users, DISCORD_USER_ID)


def post_summary_to_discord(webhook_url: str, paper: dict, summary: dict) -> Optional[dict]:
    authors = truncate_discord_text(", ".join(paper["authors"]), DISCORD_EMBED_FIELD_VALUE_LIMIT)
    embed = {
        "author": {
            "name": "arXiv",
            "url": "https://arxiv.org/",
            "icon_url": "https://shuyaojiang.github.io/assets/images/badges/arXiv.png",
        },
        "title": truncate_discord_text(summary["title"], DISCORD_EMBED_TITLE_LIMIT),
        "url": paper["entry_id"],
        "color": 0xE12D2D,
        "timestamp": datetime.datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        "fields": [
            {"name": "著者", "value": authors, "inline": False},
            {
                "name": "概要",
                "value": truncate_discord_text(summary["summary"], DISCORD_EMBED_FIELD_VALUE_LIMIT),
                "inline": False,
            },
        ],
        "thumbnail": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/7/7a/ArXiv_logo_2022.png"
        },
        "footer": {
            "text": "arXiv Summarizer",
            "icon_url": "https://cdn.discordapp.com/embed/avatars/4.png",
        },
    }
    if summary.get("appendix"):
        embed["fields"].append(
            {
                "name": "補足情報",
                "value": truncate_discord_text(summary["appendix"], DISCORD_EMBED_FIELD_VALUE_LIMIT),
                "inline": False,
            }
        )
    keywords = ", ".join(summary.get("keywords", []))
    if keywords:
        embed["fields"].append(
            {"name": "keywords", "value": truncate_discord_text(keywords, DISCORD_EMBED_FIELD_VALUE_LIMIT), "inline": False}
        )
    fit_discord_embed_total_limit(embed)

    result = post_discord_payload(
        webhook_url, {"embeds": [embed]}, f"paper {paper['paper_id']}", wait=True
    )
    if isinstance(result, dict) and result.get("id") and result.get("channel_id"):
        print(f"Sent paper: {paper['title']}")
        return result
    print(f"Discord did not return the message ID for {paper['paper_id']}")
    return None


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
        "footer": {"text": "arXiv full-paper reading memo — このスレッドに質問を書くと論文について対話できます"},
        "timestamp": datetime.datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
    }
    for field in embed["fields"]:
        field["value"] = truncate_discord_text(field["value"], 880)
    fit_discord_embed_total_limit(embed)
    return embed


def post_reading_memo_to_forum(forum_channel_id: str, paper: dict, memo: dict) -> Optional[dict]:
    result = discord_bot_request(
        "POST",
        f"/channels/{forum_channel_id}/threads",
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


def get_channel(channel_id: str) -> Optional[dict]:
    return discord_bot_request(
        "GET", f"/channels/{channel_id}", f"get channel {channel_id}"
    )


def list_active_forum_threads(forum_channel_id: str) -> List[dict]:
    channel = get_channel(forum_channel_id)
    guild_id = (channel or {}).get("guild_id", "")
    if not guild_id:
        print(f"Could not resolve guild for forum channel {forum_channel_id}")
        return []
    result = discord_bot_request(
        "GET",
        f"/guilds/{guild_id}/threads/active",
        "list active guild threads",
    )
    if not isinstance(result, dict):
        return []
    # ponytail: この API は guild 全体の active スレッドをページングなしで一括返す
    # (公式 docs に pagination の定義なし)。guild 内の active スレッドが極端に多い
    # 場合はフォーラム分の取りこぼし・遅延がありうる。問題になったら
    # channel ごとの列挙に切り替える。
    return [
        thread
        for thread in result.get("threads") or []
        if str(thread.get("parent_id")) == str(forum_channel_id)
    ]


def fetch_thread_messages(
    thread_id: str, after_message_id: Optional[str] = "0", limit: int = 50
) -> List[dict]:
    # after_message_id=None は引数なしの最新ページ取得 (履歴文脈や seed 用)
    params = {"after": after_message_id, "limit": limit}
    if after_message_id is None:
        params = {"limit": limit}
    result = discord_bot_request(
        "GET",
        f"/channels/{thread_id}/messages",
        f"fetch messages of thread {thread_id}",
        params=params,
    )
    return result if isinstance(result, list) else []


def post_thread_reply(thread_id: str, reply_text: str, description: str) -> bool:
    # ponytail: 分割投稿の後半だけが失敗した場合、再試行で前半が重複送信されうる。
    # 問題になったら質問単位の進捗記録を足す。
    ok = True
    for chunk in split_discord_message(reply_text, DISCORD_REPLY_LIMIT):
        result = discord_bot_request(
            "POST",
            f"/channels/{thread_id}/messages",
            f"{description} to thread {thread_id}",
            payload={"content": chunk, "allowed_mentions": {"parse": []}},
            success_codes=(200, 201),
        )
        ok = ok and result is not None
        if result is not None:
            time.sleep(0.5)
    return ok


def split_discord_message(text: str, limit: int) -> List[str]:
    chunks = []
    remaining = (text or "").strip()
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = window.rfind("\n")
        if cut < limit // 2:
            cut = limit - 1
        chunks.append(remaining[: cut + 1].rstrip())
        remaining = remaining[cut + 1 :].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def is_human_message(message: dict, allowed_user_id: str = "") -> bool:
    author = message.get("author") or {}
    if message.get("webhook_id"):
        return False
    if author.get("bot") or author.get("system"):
        return False
    if message.get("type") not in (0, 19):
        return False
    if not (message.get("content") or "").strip():
        return False
    if allowed_user_id and str(author.get("id")) != allowed_user_id:
        return False
    return True


def qa_transcript(messages: List[dict], allowed_user_id: str = "") -> List[dict]:
    transcript = []
    for message in messages:
        content = (message.get("content") or "").strip()
        if not content or message.get("type") not in (0, 19):
            continue
        author = message.get("author") or {}
        if author.get("bot"):
            transcript.append({"role": "assistant", "content": content})
        elif is_human_message(message, allowed_user_id):
            transcript.append({"role": "user", "content": content})
    return transcript


class ProcessResult(NamedTuple):
    replies_sent: int
    error: Optional[str]
    new_last_read: str
    done: bool


def process_thread_update(
    messages: List[dict],
    last_read_message_id: str,
    answer_fn: Callable[[List[dict], dict], str],
    post_fn: Callable[[str, dict], bool],
    allowed_user_id: str = "",
    max_answers: Optional[int] = None,
    history: Optional[List[dict]] = None,
) -> ProcessResult:
    """取得済みメッセージ列 (Discord 標準の新しい順) を古い順に処理し、
    人間の質問ごとに answer_fn で生成した返信を post_fn で送信する。

    history は今回の差分より前の発言 (古い順) で、対話の文脈にのみ使う。
    new_last_read は「処理を完了した最後のメッセージ」の ID。処理が途中で止まった
    場合は未処理質問の直前の ID を返すため、次回 poll で after=<その ID> を
    指定すると同一の質問を再処理できる。
    """
    context = list(history or [])
    ordered = list(reversed([m for m in messages if m and m.get("id")]))
    if not ordered:
        return ProcessResult(0, None, str(last_read_message_id), True)
    safe_read = str(last_read_message_id)
    seen: List[dict] = []
    replies_sent = 0
    for message in ordered:
        mid = str(message["id"])
        if is_human_message(message, allowed_user_id):
            if max_answers is not None and replies_sent >= max_answers:
                return ProcessResult(replies_sent, None, safe_read, False)
            try:
                answer = (answer_fn(qa_transcript(context + seen, allowed_user_id), message) or "").strip()
            except Exception as exc:
                return ProcessResult(replies_sent, short_error(exc), safe_read, False)
            if not answer:
                return ProcessResult(replies_sent, "model returned an empty reply", safe_read, False)
            if not post_fn(answer, message):
                return ProcessResult(replies_sent, "failed to post the reply to Discord", safe_read, False)
            seen.append(message)
            # 実投稿は次回 fetch でしか現れないため、バッチ内の後続質問へは
            # 自分の回答を合成メッセージとして文脈に渡す
            seen.append(
                {"author": {"id": "bot", "bot": True, "system": False}, "type": 0, "content": answer}
            )
            replies_sent += 1
        else:
            seen.append(message)
        safe_read = mid
    return ProcessResult(replies_sent, None, safe_read, True)


def qa_user_message(paper: dict, memo: Optional[dict], paper_text: str, transcript: List[dict], question: dict) -> str:
    qa_context = ""
    if memo:
        qa_context = yaml_free_memo(memo)
    dialogue = "\n\n".join(
        f"{'ユーザー' if m['role'] == 'user' else 'アシスタント'}: {m['content']}"
        for m in transcript
    )
    parts = ""
    if qa_context:
        parts += f"## 既存のリーディングメモ\n{qa_context}\n\n"
    parts += f"## 論文テキスト\n{paper_text}\n\n"
    if dialogue:
        parts += f"## これまでの対話\n{dialogue}\n\n"
    parts += f"## 現在の質問\n{(question.get('content') or '').strip()}"
    return parts


def yaml_free_memo(memo: dict) -> str:
    lines = []
    for key, label in (
        ("conclusion", "30秒で分かる結論"),
        ("main_claims", "主定理・主張"),
        ("method_outline", "証明・手法の骨格"),
        ("research_connection", "研究との接点"),
        ("reading_guide", "読むならここ"),
    ):
        if memo.get(key):
            lines.append(f"- {label}: {memo[key]}")
    return "\n".join(lines)


def qa_seed_last_read(thread_id: str) -> Optional[str]:
    """この機能の導入前に作られたスレッドの初回 poll 用の既読位置。
    履歴は読み飛ばしつつ、復帰直後の最新の 1 件だけ処理対象に残す。"""
    latest = fetch_thread_messages(thread_id, after_message_id=None, limit=2)
    if not latest:
        return None
    if len(latest) >= 2 and latest[1].get("id"):
        return str(latest[1]["id"])
    return "0"


def ensure_daily_job(state: dict) -> Optional[dict]:
    window = search_window()
    for job in state["jobs"]:
        if job.get("window") == window and job.get("papers"):
            return job
    papers = search_papers()
    if not papers:
        print(f"No papers found for window {window}.")
        return None
    job = new_pipeline(window, papers)
    state["jobs"].append(job)
    save_state(state)
    print(f"Created pipeline {job['pipeline_id']} for window {window} ({len(papers)} papers)")
    return job


def run_interest_phase(job: dict, state: dict) -> bool:
    """未判定の論文に興味判定を実行する。個別の失敗は state に記録して
    次回 poll で再試行し、失敗が max(3, 論文数の半分) を超えたら系統不良として False。"""
    failures = 0
    checked = 0
    for paper in job["papers"]:
        pid = paper["paper_id"]
        if pid in job["interest_results"]:
            continue
        try:
            result = llm_structured(
                "interest", PROMPT_CHECK_INTEREST, paper_prompt(paper), InterestCheck
            )
            job["interest_results"][pid] = result.interested_in
            job["interest_reasons"][pid] = result.reason
            checked += 1
            print(f"Interest for {pid}: {result.interested_in} ({result.reason})")
        except Exception as exc:
            failures += 1
            job["last_error"] = f"interest check failed for {pid}: {short_error(exc)}"
            print(f"Interest check failed for {pid}: {short_error(exc)}")
            if failures > max(3, len(job["papers"]) // 2):
                job["retry_count"] = int(job.get("retry_count", 0)) + 1
                mark_job_updated(job)
                save_state(state)
                return False
        mark_job_updated(job)
        save_state(state)
        time.sleep(LLM_CALL_INTERVAL_SECONDS)
    if checked:
        print(f"Interest phase: {checked} new checks, {failures} failures")
    return True


def run_send_phase(job: dict, state: dict) -> bool:
    """興味あり論文のうち未投稿のものだけを要約して Discord へ送る。
    投稿 ID は 📖 付与より先に state へ保存し、中断しても二重投稿しない。"""
    interested = interested_papers(job)
    sent = set(job["sent_paper_ids"])
    pending = [paper for paper in interested if paper["paper_id"] not in sent]
    if pending and not job["notification_sent"]:
        content = truncate_discord_text(
            f"新しい論文が見つかったぞ。目は通せよ（{len(pending)}件）", DISCORD_CONTENT_LIMIT
        )
        if post_discord_payload(DISCORD_WEBHOOK_URL, {"content": content}, "notification"):
            job["notification_sent"] = True
            mark_job_updated(job)
            save_state(state)
        else:
            job["last_error"] = "failed to send notification message"
            mark_job_updated(job)
            save_state(state)
            return False

    for paper in interested:
        pid = paper["paper_id"]
        if pid not in sent and pid not in job["summaries"]:
            try:
                summary = llm_structured(
                    "summary", PROMPT_SUMMARIZE, paper_prompt(paper), Summary
                )
                job["summaries"][pid] = summary.model_dump()
            except Exception as exc:
                job["last_error"] = f"summary failed for {pid}: {short_error(exc)}"
                print(f"Summary failed for {pid}: {short_error(exc)}")
                mark_job_updated(job)
                save_state(state)
                continue
            mark_job_updated(job)
            save_state(state)
            time.sleep(LLM_CALL_INTERVAL_SECONDS)
        if pid in sent:
            continue
        message_state = (job.get("discord_messages") or {}).get(pid)
        if message_state is None:
            message = post_summary_to_discord(DISCORD_WEBHOOK_URL, paper, job["summaries"][pid])
            if not message:
                job["last_error"] = f"failed to post summary for {pid}"
                mark_job_updated(job)
                save_state(state)
                continue
            message_state = {
                "message_id": message["id"],
                "channel_id": message["channel_id"],
                "reaction_added": False,
                "read_requested": False,
                "read_window_expired": False,
                "reading_memo_sent": False,
                "paper_thread_id": None,
                "qa_last_read_message_id": None,
            }
            job["discord_messages"][pid] = message_state
            mark_job_updated(job)
            save_state(state)
        if not message_state.get("reaction_added") and add_read_reaction(
            message_state["channel_id"], message_state["message_id"]
        ):
            message_state["reaction_added"] = True
        if message_state.get("reaction_added") and pid not in sent:
            job["sent_paper_ids"].append(pid)
            sent.add(pid)
        mark_job_updated(job)
        save_state(state)
        time.sleep(1.5)

    interest_complete = len(job.get("interest_results", {})) == len(job.get("papers", []))
    if len(job["sent_paper_ids"]) == len(interested):
        if not interest_complete:
            # 興味判定に未完了がある間は確定しない (失敗分は次回 poll で再試行)
            mark_job_updated(job)
            save_state(state)
            return False
        job["status"] = "completed"
        job["finalized_at"] = now_iso_utc()
        job["last_error"] = None
        mark_job_updated(job)
        save_state(state)
        return True
    job["retry_count"] = int(job.get("retry_count", 0)) + 1
    job["last_error"] = job.get("last_error") or "some papers were not sent"
    mark_job_updated(job)
    save_state(state)
    return False


def process_daily_jobs(state: dict) -> int:
    ensure_daily_job(state)
    status = 0
    for job in [j for j in state["jobs"] if j.get("status") == "processing"]:
        if not run_interest_phase(job, state):
            status = 1
            continue
        if not job.get("interest_results"):
            status = 1
            continue
        if not interested_papers(job):
            job["status"] = "completed_no_interests"
            job["finalized_at"] = now_iso_utc()
            job["last_error"] = None
            mark_job_updated(job)
            save_state(state)
            continue
        if not run_send_phase(job, state):
            status = 1
    return status


def run_stage_poll_reading_requests() -> int:
    memos_generated = 0
    state = load_state()
    updated = False
    for job in state["jobs"]:
        if is_older_than_days(job.get("created_at", ""), READING_REQUEST_WINDOW_DAYS):
            for message_state in (job.get("discord_messages") or {}).values():
                if message_state.get("read_requested") or message_state.get("reading_memo_sent"):
                    continue
                if not message_state.get("read_window_expired"):
                    message_state["read_window_expired"] = True
                    updated = True
            continue
        papers_by_id = {paper["paper_id"]: paper for paper in job.get("papers", [])}
        for paper_id, message_state in (job.get("discord_messages") or {}).items():
            if message_state.get("reading_memo_sent") or message_state.get("read_window_expired"):
                continue
            if not message_state.get("read_requested"):
                if not has_read_request(message_state["channel_id"], message_state["message_id"]):
                    continue
                message_state["read_requested"] = True
                message_state["read_requested_at"] = now_iso_utc()
                updated = True
            paper = papers_by_id.get(paper_id)
            if paper is None:
                message_state["reading_last_error"] = "paper metadata is missing"
                updated = True
                continue
            if paper_id not in job.get("reading_memos", {}):
                try:
                    paper_text = extract_paper_text(paper, READING_MAX_PAPER_CHARS)
                    memo = llm_structured(
                        "reading",
                        PROMPT_READING_MEMO,
                        reading_user_message(paper, paper_text),
                        ReadingMemo,
                    )
                    job.setdefault("reading_memos", {})[paper_id] = memo.model_dump()
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
                DISCORD_FORUM_CHANNEL_ID, paper, job["reading_memos"][paper_id]
            )
            if forum_post:
                message_state["reading_memo_sent"] = True
                message_state["paper_thread_id"] = forum_post["id"]
                message_state["reading_memo_sent_at"] = now_iso_utc()
                message_state["reading_last_error"] = None
                message_state["qa_last_read_message_id"] = forum_post.get("last_message_id") or "0"
                print(f"Created Forum post for {paper_id}: {forum_post['id']}")
                memos_generated += 1
            else:
                message_state["reading_retry_count"] = int(
                    message_state.get("reading_retry_count", 0)
                ) + 1
                message_state["reading_last_error"] = "failed to create Forum post"
            updated = True
    if updated:
        save_state(state)
    print(f"Reading requests: {memos_generated} memo(s) generated")
    return 0


def reading_user_message(paper: dict, paper_text: str) -> str:
    return (
        f"Title: {paper['title']}\nURL: {paper['entry_id']}\n\n"
        f"論文全文 (PDF から自動抽出したテキスト。数式は崩れている可能性がある):\n{paper_text}"
    )


def run_stage_poll_paper_qa() -> int:
    if not DISCORD_FORUM_CHANNEL_ID:
        print("DISCORD_FORUM_CHANNEL_ID is not set.")
        return 1
    active_threads = {str(t.get("id")): t for t in list_active_forum_threads(DISCORD_FORUM_CHANNEL_ID)}
    if not active_threads:
        return 0
    state = load_state()
    answers_posted = 0
    for job in state["jobs"]:
        papers_by_id = {paper["paper_id"]: paper for paper in job.get("papers", [])}
        for paper_id, message_state in (job.get("discord_messages") or {}).items():
            thread_id = str(message_state.get("paper_thread_id") or "")
            if not thread_id or thread_id not in active_threads:
                continue
            if message_state.get("qa_last_read_message_id") is None:
                seeded = qa_seed_last_read(thread_id)
                if seeded is None:
                    continue
                message_state["qa_last_read_message_id"] = seeded
                save_state(state)
            last_read = str(message_state["qa_last_read_message_id"])
            messages = fetch_thread_messages(thread_id, last_read)
            if not messages:
                continue
            if not [m for m in messages if is_human_message(m, DISCORD_USER_ID)]:
                message_state["qa_last_read_message_id"] = str(messages[0].get("id"))
                save_state(state)
                continue
            paper = papers_by_id.get(paper_id)
            if paper is None:
                message_state["qa_last_error"] = "paper metadata is missing"
                save_state(state)
                continue
            memo = (job.get("reading_memos") or {}).get(paper_id)
            text_cache: Dict[str, str] = {}
            history_window = fetch_thread_messages(
                thread_id, after_message_id=None, limit=QA_HISTORY_MESSAGES
            )
            batch_ids = {str(m.get("id")) for m in messages}
            history = [m for m in reversed(history_window) if str(m.get("id")) not in batch_ids]

            def answer_fn(transcript, question):
                if "text" not in text_cache:
                    text_cache["text"] = extract_paper_text(paper, QA_MAX_PAPER_CHARS)
                return llm_text(
                    "qa",
                    PROMPT_PAPER_QA,
                    qa_user_message(paper, memo, text_cache["text"], transcript, question),
                )

            def post_fn(reply, question):
                return post_thread_reply(thread_id, reply, "send paper QA reply")

            result = process_thread_update(
                messages,
                last_read,
                answer_fn,
                post_fn,
                allowed_user_id=DISCORD_USER_ID,
                max_answers=QA_MAX_REPLIES_PER_POLL,
                history=history,
            )
            answers_posted += result.replies_sent
            if result.error:
                message_state["qa_retry_count"] = int(message_state.get("qa_retry_count", 0)) + 1
                message_state["qa_last_error"] = result.error
            else:
                message_state["qa_last_error"] = None
            message_state["qa_last_read_message_id"] = result.new_last_read
            # スレッドごとに保存し、中断時に未反映の回答が再送される窓を最小化する
            save_state(state)
    print(f"Paper QA: {answers_posted} replies posted")
    return 0


def run_stage_poll() -> int:
    if not DISCORD_WEBHOOK_URL:
        print("ARXIV_RECOMMENDER_WEBHOOK_URL is not set.")
        return 1
    if not DISCORD_BOT_TOKEN:
        print("DISCORD_BOT_TOKEN is not set.")
        return 1
    state = load_state()
    daily_status = process_daily_jobs(state)
    memos_status = run_stage_poll_reading_requests()
    qa_status = run_stage_poll_paper_qa()
    return 1 if 1 in (daily_status, memos_status, qa_status) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="arXiv recommender poll pipeline")
    parser.add_argument(
        "--stage",
        choices=["poll", "daily", "reading_requests", "paper_qa"],
        default=os.getenv("PIPELINE_STAGE", "poll"),
        help="Pipeline stage to execute",
    )
    args = parser.parse_args()

    if args.stage == "daily":
        if not DISCORD_WEBHOOK_URL:
            print("ARXIV_RECOMMENDER_WEBHOOK_URL is not set.")
            return 1
        return process_daily_jobs(load_state())
    if args.stage == "reading_requests":
        return run_stage_poll_reading_requests()
    if args.stage == "paper_qa":
        return run_stage_poll_paper_qa()
    return run_stage_poll()


if __name__ == "__main__":
    raise SystemExit(main())
