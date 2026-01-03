from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Optional
import arxiv
import time, os, datetime
from zoneinfo import ZoneInfo
import requests, json

# set up clients for arXiv, GenAI, and Discord
client_arxiv = arxiv.Client()
client_genai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def search_papers():
    # search for papers submitted yesterday
    yesterday = datetime.datetime.now(ZoneInfo("America/New_York")) - datetime.timedelta(days=3)
    search_start = yesterday.strftime("%Y%m%d0000")
    search_end = yesterday.strftime("%Y%m%d2359")
    print(f"Searching papers from {search_start} to {search_end}")

    # 検索条件を指定する。
    # query: 検索キーワードなどを指定する。
    # max_results: 取得する論文の最大件数を指定する。
    # sort_by: 論文の並び替え条件を指定する。ここでは投稿日時の降順（最新順）。
    # cat:math.DS+cat:math.CO+cat:math.GR+cat:cs.LO+cat:cs.FL+cat:cs.DM
    search = arxiv.Search(
        query = f"(cat:math.DS OR cat:math.CO OR cat:math.GR OR cat:cs.LO OR cat:cs.FL OR cat:cs.DM) AND submittedDate:[{search_start} TO {search_end}]",
        max_results = None,
        sort_by = arxiv.SortCriterion.SubmittedDate
    )

    # 検索を実行し、結果を取得する。
    results = client_arxiv.results(search)
    return results

# # 取得した論文のタイトルを1件ずつ表示する。
# results = search_papers()
# counter = 0
# for r in results:
#     counter += 1
#     print(r.title, " ", r.published)
#     print(r.categories)
#     print(r.summary)
#     print("")
# print(counter, "papers found.")

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

def check_interest(papers_info:arxiv.Generator[arxiv.Result, None, None]):
    print("Checking interest for paper...")
    # batch request を作る
    # 構造は次の通り
    """
    request_raw = [
        {
            'contents': [{
                'parts': [{'text': prompt_check_interest + title1 + abstract1}],
            }],
            'config': {
                'response_mime_type': 'application/json',
                'response_schema': InterestCheck,
            }
        },
        {
            'contents': [{
                'parts': [{'text': prompt_check_interest + title2 + abstract2}],
            }],
            'config': {
                'response_mime_type': 'application/json',
                'response_schema': InterestCheck,
            }
        },...]
    """
    inline_request:List[dict] = []
    for paper_info in papers_info:
        title = f"\nTitle: {paper_info.title}\n"
        abstract = f"\nAbstract: {paper_info.summary}\n"
        request_item = {
            'contents': [{
                'parts': [{'text': title + abstract + prompt_check_interest}],
            }],
            'config': {
                'response_mime_type': 'application/json',
                'response_schema': InterestCheck, # batch api では json schema などをつけるとエラーになる
            }
        }
        inline_request.append(request_item)

    # batch request を送信する
    batch_job = client_genai.batches.create(
        model="gemini-2.5-flash",
        src=inline_request,
        config={
            'display_name': 'Interest Check Batch Job',
        }
    )
    print(f"Batch job created: {batch_job.name}")
    print(f"Number of papers in batch: {len(inline_request)}")
    print("Waiting for batch job to complete...")
    # batch request の結果を取得する
    batch_start_time = time.time()
    completed_status = ('JOB_STATE_SUCCEEDED','JOB_STATE_FAILED','JOB_STATE_CANCELLED','JOB_STATE_EXPIRED')
    while batch_job.state.name not in completed_status:
        print(f"Current state: {batch_job.state.name} ({time.time() - batch_start_time:.0f}s)",end='\r')
        time.sleep(30)
        batch_job = client_genai.batches.get(name = batch_job.name)

    print(f"Final state: {batch_job.state.name} ({time.time() - batch_start_time:.0f}s)")
    interest_check = []
    for i, inline_response in enumerate(batch_job.dest.inlined_responses):
        # print(f"Result for paper {i+1}:")
        if inline_response.response:
            is_interest = InterestCheck.model_validate_json(inline_response.response.text)
            interest_check.append(is_interest.interested_in)
            # print(f"  Interested: {is_interest.interested_in}")
    return interest_check

def check_interest_sequential(papers_info:arxiv.Generator[arxiv.Result, None, None]):
    print("Checking interest for paper sequentially...")
    interest_check = []
    for i, paper_info in enumerate(papers_info):
        title = f"\nTitle: {paper_info.title}\n"
        abstract = f"\nAbstract: {paper_info.summary}\n"
        response = client_genai.models.generate_content(
            model = "gemini-2.5-flash",
            contents = title + abstract + prompt_check_interest,
            config = {
                "response_mime_type": "application/json",
                "response_schema": InterestCheck,
            }
        )
        is_interest = InterestCheck.model_validate_json(response.text)
        interest_check.append(is_interest.interested_in)
        print(f"Result for paper {i+1}: Interested: {is_interest.interested_in}")
    return interest_check

def summarize_paper(papers_info:List[arxiv.Result]):
    if len(papers_info) == 0:
        return []
    print("Summarizing paper...")
    # batch request を作る
    inline_request = []
    for paper_info in papers_info:
        title = f"\nTitle: {paper_info.title}\n"
        abstract = f"\nAbstract: {paper_info.summary}\n"
        request_item = {
            'contents': [{
                'parts': [{'text': title + abstract + prompt_summarize}],
            }],
            'config': {
                'response_mime_type': 'application/json',
                'response_schema': Summary, # batch api では json schema などをつけるとエラーになる
                'thinking_config': {'thinking_level': 'low'}
            }
        }
        inline_request.append(request_item)

    # batch request を送信する
    batch_job = client_genai.batches.create(
        model="gemini-3-flash-preview",
        src=inline_request,
        config={
            'display_name': 'Summarize Paper Batch Job',
        }
    )
    print(f"Batch job created: {batch_job.name}")
    print(f"Number of papers in batch: {len(inline_request)}")
    print("Waiting for batch job to complete...")
    # batch request の結果を取得する
    batch_start_time = time.time()
    completed_status = ('JOB_STATE_SUCCEEDED','JOB_STATE_FAILED','JOB_STATE_CANCELLED','JOB_STATE_EXPIRED')
    while batch_job.state.name not in completed_status:
        print(f"Current state: {batch_job.state.name} ({time.time() - batch_start_time:.0f}s)",end='\r')
        time.sleep(30)
        batch_job = client_genai.batches.get(name = batch_job.name)
    print(f"Final state: {batch_job.state.name} ({time.time() - batch_start_time:.0f}s)")
    summaries = []
    for i, inline_response in enumerate(batch_job.dest.inlined_responses):
        print(f"Result for paper {i+1}:")
        if inline_response.response:
            summary = Summary.model_validate_json(inline_response.response.text)
            summaries.append(summary)
            print(f"  Title: {summary.title}")
    return summaries

def summarize_paper_sequential(papers_info:List[arxiv.Result]):
    if len(papers_info) == 0:
        return []
    print("Summarizing paper sequentially...")
    summaries = []
    for i, paper_info in enumerate(papers_info):
        title = f"\nTitle: {paper_info.title}\n"
        abstract = f"\nAbstract: {paper_info.summary}\n"
        response = client_genai.models.generate_content(
            model = "gemini-3-flash-preview",
            contents = title + abstract + prompt_summarize,
            config = {
                "response_mime_type": "application/json",
                "response_schema": Summary,
                "thinking_config": {'thinking_level': 'low'}
            }
        )
        summary = Summary.model_validate_json(response.text)
        summaries.append(summary)
        print(f"Result for paper {i+1}: Title: {summary.title}")
    return summaries

def main():
    discord_webhook_url = os.getenv("ARXIV_RECOMMENDER_WEBHOOK_URL")
    search_results = list(search_papers())

    if len(search_results) == 0:
        print("No papers found, exiting.")
        exit(0)

    print(f"{len(search_results)} papers found in total.")
    interests = check_interest(search_results)
    # interests = check_interest_sequential(search_results)
    # interests = [True, False, True, True, False]  # テスト用ダミーデータ
    # interested な論文だけを抽出する
    results = list(filter(lambda x: interests.pop(0), search_results))
    
    # Discord に送信する
    if len(results) == 0:
        print("No interesting papers found, exiting.")
        exit(0)

    summaries = summarize_paper(results)
    # summaries = summarize_paper_sequential(results)
    is_sending_successful = True
    message = {"content": f"新しい論文が見つかったぞ。目は通せよ（{len(results)}件）"}
    headers = {"Content-Type": "application/json"}
    response = requests.post(discord_webhook_url, data=json.dumps(message), headers=headers)
    if response.status_code == 204:
        print("Notification sent successfully to Discord.")
    else:
        print(f"Failed to send notification to Discord. Status code: {response.status_code}, Response: {response.text}")
        is_sending_successful = False
    for i, paper in enumerate(results):
        summary = summaries[i]
        authors = ', '.join([str(author) for author in paper.authors])
        embed = {
            "author": {
                "name": "arXiv",
                "url": "https://arxiv.org/",
                "icon_url": "https://shuyaojiang.github.io/assets/images/badges/arXiv.png"
            },
            "title": f"{summary.title}",
            "url": f"{paper.entry_id}",
            "color": 0xe12d2d,
            "timestamp": (datetime.datetime.now(ZoneInfo("Asia/Tokyo"))).isoformat(), # 日本時間に変換
            "fields": [
                {
                    "name": "著者",
                    "value": authors,
                    "inline": False
                },
                {
                    "name": "概要",
                    "value": summary.summary,
                    "inline": False
                },
            ],
            "thumbnail": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/7/7a/ArXiv_logo_2022.png"
            },
            "footer": {
                "text": "arXiv Summarizer",
                "icon_url": "https://cdn.discordapp.com/embed/avatars/4.png"
            }
        }
        
        if summary.appendix:
            embed["fields"].append({"name": "補足情報", "value": summary.appendix, "inline": False})
        embed["fields"].append({"name": "keywords", "value": ', '.join(summary.keywords), "inline": False})

        message = {"embeds": [embed]}
        headers = {"Content-Type": "application/json"}
        response = requests.post(discord_webhook_url, data=json.dumps(message), headers=headers)
        if response.status_code == 204:
            print("Message sent successfully to Discord.")
        else:
            print(f"Failed to send message to Discord. Status code: {response.status_code}, Response: {response.text}")
            print(json.dumps(message))
            is_sending_successful = False
        time.sleep(1.5)  # Discord のレート制限対策のため、1.5秒待機する
    
    print("All done, exiting.")
    if is_sending_successful:
        exit(0)
    else:
        exit(1)
    exit(0)

main()