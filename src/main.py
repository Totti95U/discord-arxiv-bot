from google import genai
from pydantic import BaseModel, Field
from typing import List, Optional
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
INTEREST_MODEL = "gemini-2.5-flash"
SUMMARY_MODEL = "gemini-3-flash-preview"
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


def check_interest_sequential_papers(papers: List[dict]) -> List[bool]:
    print("Checking interest sequentially...")
    interest_check: List[bool] = []
    for i, paper in enumerate(papers):
        title = f"\nTitle: {paper['title']}\n"
        abstract = f"\nAbstract: {paper['summary']}\n"
        response = client_genai.models.generate_content(
            model=INTEREST_MODEL,
            contents=title + abstract + prompt_check_interest,
            config={
                "response_mime_type": "application/json",
                "response_schema": InterestCheck,
            },
        )
        is_interest = InterestCheck.model_validate_json(response.text)
        interest_check.append(is_interest.interested_in)
        print(f"Result for paper {i + 1}: Interested: {is_interest.interested_in}")
    return interest_check


def summarize_sequential_papers(papers: List[dict], existing_summaries: dict) -> dict:
    print("Summarizing papers sequentially...")
    summaries = dict(existing_summaries)
    for i, paper in enumerate(papers):
        paper_id = paper["paper_id"]
        if paper_id in summaries:
            continue

        title = f"\nTitle: {paper['title']}\n"
        abstract = f"\nAbstract: {paper['summary']}\n"
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
    return summaries


def extract_interest_check(batch_job, papers_len: int) -> List[bool]:
    interest_check = [False for _ in range(papers_len)]
    for i, inline_response in enumerate(batch_job.dest.inlined_responses):
        if i >= papers_len:
            break
        if inline_response.response:
            is_interest = InterestCheck.model_validate_json(inline_response.response.text)
            interest_check[i] = is_interest.interested_in
    return interest_check


def extract_summaries(batch_job, papers: List[dict]) -> dict:
    summaries = {}
    for i, inline_response in enumerate(batch_job.dest.inlined_responses):
        if i >= len(papers):
            break
        if inline_response.response:
            summary = Summary.model_validate_json(inline_response.response.text)
            paper_id = papers[i]["paper_id"]
            summaries[paper_id] = {
                "title": summary.title,
                "summary": summary.summary,
                "keywords": summary.keywords,
                "appendix": summary.appendix,
            }
    return summaries


def post_summary_to_discord(webhook_url: str, paper: dict, summary: dict) -> bool:
    authors = ", ".join(paper["authors"])
    embed = {
        "author": {
            "name": "arXiv",
            "url": "https://arxiv.org/",
            "icon_url": "https://shuyaojiang.github.io/assets/images/badges/arXiv.png",
        },
        "title": summary["title"],
        "url": paper["entry_id"],
        "color": 0xE12D2D,
        "timestamp": datetime.datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        "fields": [
            {
                "name": "著者",
                "value": authors,
                "inline": False,
            },
            {
                "name": "概要",
                "value": summary["summary"],
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
        embed["fields"].append({"name": "補足情報", "value": summary["appendix"], "inline": False})

    keywords = summary.get("keywords", [])
    embed["fields"].append({"name": "keywords", "value": ", ".join(keywords), "inline": False})

    message = {"embeds": [embed]}
    headers = {"Content-Type": "application/json"}
    response = requests.post(webhook_url, data=json.dumps(message), headers=headers)
    if response.status_code == 204:
        print(f"Sent paper: {paper['title']}")
        return True

    print(f"Failed to send paper {paper['paper_id']}. Status: {response.status_code}, Body: {response.text}")
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
                continue

            if batch_state != "JOB_STATE_SUCCEEDED":
                job["status"] = "failed"
                job["last_error"] = f"interest batch ended with {batch_state}"
                mark_job_updated(job)
                updated = True
                continue

            interests = extract_interest_check(batch_job, len(job["papers"]))
            interested_ids = [
                paper["paper_id"]
                for i, paper in enumerate(job["papers"])
                if i < len(interests) and interests[i]
            ]
            job["interested_paper_ids"] = interested_ids
            updated = True

            if len(interested_ids) == 0:
                job["status"] = "completed_no_interests"
                job["finalized_at"] = now_iso_utc()
                mark_job_updated(job)
                continue

            interested_papers = [
                paper for paper in job["papers"] if paper["paper_id"] in set(interested_ids)
            ]
            summarize_job_name = submit_summary_batch(interested_papers)
            job["summarize_job_name"] = summarize_job_name
            job["status"] = "summarize_submitted"
            job["last_error"] = None
            mark_job_updated(job)
            continue

        try:
            interests = check_interest_sequential_papers(job.get("papers", []))
            interested_ids = [
                paper["paper_id"]
                for i, paper in enumerate(job.get("papers", []))
                if i < len(interests) and interests[i]
            ]
            job["interested_paper_ids"] = interested_ids
            updated = True

            if len(interested_ids) == 0:
                job["status"] = "completed_no_interests"
                job["finalized_at"] = now_iso_utc()
                job["last_error"] = None
                mark_job_updated(job)
                continue

            interested_papers = [
                paper for paper in job["papers"] if paper["paper_id"] in set(interested_ids)
            ]
            summarize_job_name = submit_summary_batch(interested_papers)
            job["summarize_job_name"] = summarize_job_name
            job["status"] = "summarize_submitted"
            job["last_error"] = None
            mark_job_updated(job)
            updated = True
        except Exception as exc:
            job["status"] = "interest_fallback_running"
            job["retry_count"] = int(job.get("retry_count", 0)) + 1
            job["last_error"] = f"interest fallback failed: {exc}"
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
                continue

            if batch_state != "JOB_STATE_SUCCEEDED":
                job["status"] = "failed"
                job["last_error"] = f"summary batch ended with {batch_state}"
                mark_job_updated(job)
                updated = True
                continue

            interested_set = set(job.get("interested_paper_ids", []))
            interested_papers = [paper for paper in job.get("papers", []) if paper["paper_id"] in interested_set]
            extracted = extract_summaries(batch_job, interested_papers)
            job["summaries"].update(extracted)
            updated = True

        if job.get("status") == "summary_fallback_running":
            interested_set = set(job.get("interested_paper_ids", []))
            interested_papers = [paper for paper in job.get("papers", []) if paper["paper_id"] in interested_set]
            try:
                job["summaries"] = summarize_sequential_papers(interested_papers, job.get("summaries", {}))
                job["last_error"] = None
                mark_job_updated(job)
                updated = True
            except Exception as exc:
                job["status"] = "summary_fallback_running"
                job["retry_count"] = int(job.get("retry_count", 0)) + 1
                job["last_error"] = f"summary fallback failed: {exc}"
                mark_job_updated(job)
                updated = True
                continue

        pending_ids = [
            paper_id
            for paper_id in job.get("interested_paper_ids", [])
            if paper_id in job.get("summaries", {}) and paper_id not in set(job.get("sent_paper_ids", []))
        ]

        if len(pending_ids) == 0:
            if len(job.get("interested_paper_ids", [])) == len(job.get("sent_paper_ids", [])):
                job["status"] = "completed"
                job["finalized_at"] = now_iso_utc()
                job["last_error"] = None
                mark_job_updated(job)
                updated = True
            continue

        if not job.get("notification_sent", False):
            message = {"content": f"新しい論文が見つかったぞ。目は通せよ（{len(pending_ids)}件）"}
            headers = {"Content-Type": "application/json"}
            response = requests.post(discord_webhook_url, data=json.dumps(message), headers=headers)
            if response.status_code == 204:
                print("Notification sent successfully to Discord.")
                job["notification_sent"] = True
                job["last_error"] = None
                mark_job_updated(job)
                updated = True
            else:
                print(f"Failed to send notification to Discord. Status code: {response.status_code}, Response: {response.text}")
                job["status"] = "send_failed"
                job["last_error"] = "failed to send notification message"
                mark_job_updated(job)
                updated = True
                continue

        all_success = True
        papers_by_id = {paper["paper_id"]: paper for paper in job.get("papers", [])}
        for paper_id in pending_ids:
            paper = papers_by_id.get(paper_id)
            summary = job.get("summaries", {}).get(paper_id)
            if paper is None or summary is None:
                continue
            is_sent = post_summary_to_discord(discord_webhook_url, paper, summary)
            if is_sent:
                if paper_id not in job["sent_paper_ids"]:
                    job["sent_paper_ids"].append(paper_id)
                    updated = True
            else:
                all_success = False
            time.sleep(1.5)

        if all_success and len(job["sent_paper_ids"]) == len(job.get("interested_paper_ids", [])):
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
