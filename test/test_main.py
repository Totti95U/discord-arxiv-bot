import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

import main  # noqa: E402


class JsonExtractionTest(unittest.TestCase):
    def test_plain_object(self):
        self.assertEqual(main.extract_json_object('{"a": 1}'), {"a": 1})

    def test_fenced(self):
        text = '前置き\n```json\n{"interested_in": true}\n```\n後置き'
        self.assertEqual(main.extract_json_object(text), {"interested_in": True})

    def test_prose_wrapped(self):
        text = '答えは次の通りです: {"interested_in": false, "reason": "無関係"} 以上'
        self.assertEqual(
            main.extract_json_object(text),
            {"interested_in": False, "reason": "無関係"},
        )

    def test_garbage_returns_none(self):
        self.assertIsNone(main.extract_json_object("これは JSON ではない"))


class RoleConfigTest(unittest.TestCase):
    def env(self, **kwargs):
        base = {
            "LLM_API_BASE": "https://api.example.com/v1",
            "LLM_API_KEY": "shared",
            "INTEREST_MODEL": "decider-1",
        }
        base.update(kwargs)
        return base

    def test_defaults_chain(self):
        with mock.patch.dict(os.environ, self.env(), clear=False):
            cfg = main.role_config("summary")
            self.assertEqual(cfg["model"], "decider-1")
            cfg = main.role_config("qa")
            self.assertEqual(cfg["model"], "decider-1")

    def test_summary_falls_back_to_interest_not_reading(self):
        env = self.env(READING_MODEL="reader-9")
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertEqual(main.role_config("summary")["model"], "decider-1")
            self.assertEqual(main.role_config("reading")["model"], "reader-9")
            self.assertEqual(main.role_config("qa")["model"], "reader-9")

    def test_reading_falls_back_to_summary(self):
        env = self.env(SUMMARY_MODEL="writer-2")
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertEqual(main.role_config("reading")["model"], "writer-2")
            self.assertEqual(main.role_config("interest")["model"], "decider-1")

    def test_per_role_override(self):
        env = self.env(
            SUMMARY_MODEL="writer-2",
            SUMMARY_API_BASE="https://other.example.com/v1",
            SUMMARY_API_KEY="other-key",
        )
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = main.role_config("summary")
            self.assertEqual(cfg["model"], "writer-2")
            self.assertEqual(cfg["base_url"], "https://other.example.com/v1")
            self.assertEqual(cfg["api_key"], "other-key")
            cfg = main.role_config("interest")
            self.assertEqual(cfg["api_key"], "shared")

    def test_missing_model_raises(self):
        env = {"LLM_API_BASE": "https://api.example.com/v1", "LLM_API_KEY": "k"}
        with mock.patch.dict(os.environ, env, clear=False):
            with self.assertRaises(RuntimeError):
                main.role_config("interest")


class MessageSplitTest(unittest.TestCase):
    def test_short_text_single_chunk(self):
        self.assertEqual(main.split_discord_message("短い", 10), ["短い"])

    def test_split_on_newline_boundary(self):
        text = "一行目\n二行目\n三行目"
        self.assertEqual(len(main.split_discord_message(text, 10)), 2)
        for chunk in main.split_discord_message(text, 10):
            self.assertLessEqual(len(chunk), 10)

    def test_hard_split_when_no_boundary(self):
        text = "あ" * 25
        chunks = main.split_discord_message(text, 10)
        self.assertEqual([len(c) for c in chunks], [10, 10, 5])
        self.assertTrue(all(len(c) <= 10 for c in chunks))


class HumanMessageTest(unittest.TestCase):
    def msg(self, **kwargs):
        base = {
            "id": "100",
            "type": 0,
            "content": "質問",
            "author": {"id": "42", "bot": False, "system": False},
            "webhook_id": None,
        }
        base.update(kwargs)
        return base

    def test_human_default(self):
        self.assertTrue(main.is_human_message(self.msg()))

    def test_bot_message_rejected(self):
        self.assertFalse(main.is_human_message(self.msg(author={"id": "1", "bot": True, "system": False})))

    def test_webhook_rejected(self):
        self.assertFalse(main.is_human_message(self.msg(webhook_id="77")))

    def test_empty_content_rejected(self):
        self.assertFalse(main.is_human_message(self.msg(content="  ")))

    def test_system_type_rejected(self):
        self.assertFalse(main.is_human_message(self.msg(type=7)))

    def test_reply_type_accepted(self):
        self.assertTrue(main.is_human_message(self.msg(type=19)))

    def test_user_filter(self):
        self.assertTrue(main.is_human_message(self.msg(), allowed_user_id="42"))
        self.assertFalse(main.is_human_message(self.msg(), allowed_user_id="99"))


class ThreadUpdateTest(unittest.TestCase):
    def messages(self, specs):
        msgs = []
        for i, (author, content) in enumerate(specs, start=1):
            is_bot = author == "bot"
            msgs.append(
                {
                    "id": str(1000 + i),
                    "type": 0,
                    "content": content,
                    "author": {"id": "9" if is_bot else "42", "bot": is_bot, "system": False},
                    "webhook_id": None,
                }
            )
        # Discord returns newest first
        return list(reversed(msgs))

    def test_answers_in_order_and_advances(self):
        msgs = self.messages([("bot", "メモ"), ("human", "Q1"), ("human", "Q2")])
        replies = []
        answers = []

        def answer_fn(transcript, question):
            answers.append(question["content"])
            return "回答:" + question["content"]

        def post_fn(reply, question):
            replies.append(reply)
            return True

        sent, error, last_read, done = main.process_thread_update(
            msgs, "0", answer_fn, post_fn
        )
        self.assertEqual(sent, 2)
        self.assertIsNone(error)
        self.assertTrue(done)
        self.assertEqual(last_read, msgs[0]["id"])
        self.assertEqual(answers, ["Q1", "Q2"])
        self.assertEqual(replies, ["回答:Q1", "回答:Q2"])

    def test_failure_keeps_last_read_before_failed_question(self):
        msgs = self.messages([("human", "Q1"), ("human", "Q2")])

        def answer_fn(transcript, question):
            if question["content"] == "Q2":
                raise RuntimeError("boom")
            return "ok:" + question["content"]

        def post_fn(reply, question):
            return True

        sent, error, last_read, done = main.process_thread_update(
            msgs, "0", answer_fn, post_fn
        )
        self.assertEqual(sent, 1)
        self.assertFalse(done)
        self.assertIn("boom", error)
        # Q1 = id 1001 は処理済み、Q2 = id 1002 は未処理のまま返る
        self.assertEqual(last_read, "1001")

    def test_retry_after_failure_reprocesses_failed_question(self):
        msgs = self.messages([("human", "Q1")])
        calls = []

        def answer_fn(transcript, question):
            calls.append(question["content"])
            if len(calls) == 1:
                raise RuntimeError("boom")
            return "ok"

        def post_fn(reply, question):
            return True

        first = main.process_thread_update(msgs, "0", answer_fn, post_fn)
        self.assertEqual(first[3], False)
        second = main.process_thread_update(msgs, first[2], answer_fn, post_fn)
        self.assertEqual(second[0], 1)
        self.assertTrue(second[3])
        self.assertEqual(calls, ["Q1", "Q1"])

    def test_transcript_includes_own_answer_within_batch(self):
        msgs = self.messages([("human", "Q1"), ("human", "Q2")])
        seen = []

        def answer_fn(transcript, question):
            seen.append([m["content"] for m in transcript])
            return "A:" + question["content"]

        sent, error, last_read, done = main.process_thread_update(
            msgs, "0", answer_fn, lambda r, q: True
        )
        self.assertEqual(sent, 2)
        # Q2 を答える時点で、直前の質問と Bot の回答が文脈に乗っている
        self.assertEqual(seen[1], ["Q1", "A:Q1"])
        self.assertTrue(main.qa_transcript([{"author": {"bot": True}, "type": 0, "content": "A:Q1"}])[0]["role"] == "assistant")

    def test_history_provides_cross_poll_context(self):
        msgs = self.messages([("human", "Q2")])
        history = list(
            reversed(
                self.messages([("human", "Q1"), ("bot", "A1")])
            )
        )
        transcripts = []
        main.process_thread_update(
            msgs, "1002", lambda t, q: (transcripts.append(t), "ok")[1], lambda r, q: True,
            history=history,
        )
        # 別 poll で発生した Q1 と回答が履歴として渡る
        self.assertEqual([m["role"] for m in transcripts[0]], ["user", "assistant"])
        self.assertEqual([m["content"] for m in transcripts[0]], ["Q1", "A1"])

    def test_empty_answer_marks_failure(self):
        msgs = self.messages([("human", "Q1")])
        sent, error, last_read, done = main.process_thread_update(
            msgs, "0", lambda t, q: "  ", lambda r, q: True
        )
        self.assertEqual((sent, done), (0, False))
        self.assertIn("empty", error)
        self.assertEqual(last_read, "0")

    def test_max_answers_stops_without_error(self):
        msgs = self.messages([("human", "Q1"), ("human", "Q2"), ("human", "Q3")])

        main.process_thread_update(
            msgs, "0", lambda t, q: "ok", lambda r, q: True, max_answers=2
        )
        sent, error, last_read, done = main.process_thread_update(
            msgs, "0", lambda t, q: "ok", lambda r, q: True, max_answers=2
        )
        self.assertEqual((sent, done), (2, False))
        self.assertIsNone(error)
        self.assertEqual(last_read, "1002")


class PromptFilesTest(unittest.TestCase):
    def test_prompt_files_exist_and_nonempty(self):
        for name in (
            "prompt_check_interest.txt",
            "prompt_summarize.txt",
            "prompt_reading_memo.txt",
            "prompt_paper_qa.txt",
        ):
            path = os.path.join(main.PROMPT_DIR, name)
            self.assertTrue(os.path.isfile(path), name)
            self.assertGreater(os.path.getsize(path), 20, name)


class StateFileTest(unittest.TestCase):
    def test_load_missing_state_creates_fresh(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state", "state.json")
            with mock.patch.object(main, "STATE_FILE_PATH", path):
                state = main.load_state()
                self.assertEqual(state["jobs"], [])
                self.assertEqual(state["schema_version"], 2)
                self.assertTrue(os.path.isfile(path))

    def test_load_invalid_json_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            with io.open(path, "w", encoding="utf-8") as f:
                f.write("not json at all")
            with mock.patch.object(main, "STATE_FILE_PATH", path):
                with self.assertRaises(json.JSONDecodeError):
                    main.load_state()


class QaSeedTest(unittest.TestCase):
    def test_seed_with_single_message_resumes_from_start(self):
        latest = [{"id": "500", "content": "Q", "author": {"id": "1", "bot": False}, "type": 0}]
        with mock.patch.object(main, "fetch_thread_messages", return_value=latest):
            self.assertEqual(main.qa_seed_last_read("t"), "0")

    def test_seed_keeps_newest_message_pending(self):
        latest = [
            {"id": "600", "content": "Q new", "author": {"id": "1", "bot": False}, "type": 0},
            {"id": "599", "content": "memo", "author": {"id": "9", "bot": True}, "type": 0},
        ]
        with mock.patch.object(main, "fetch_thread_messages", return_value=latest):
            # 最新 1 件だけ処理対象に残し、それより前は読み飛ばす
            self.assertEqual(main.qa_seed_last_read("t"), "599")

    def test_seed_returns_none_for_empty_thread(self):
        with mock.patch.object(main, "fetch_thread_messages", return_value=[]):
            self.assertIsNone(main.qa_seed_last_read("t"))


class SendPhaseTest(unittest.TestCase):
    def job(self, interest_results, sent):
        return {
            "pipeline_id": "p",
            "window": "20261116",
            "status": "processing",
            "papers": [
                {"paper_id": "p1", "entry_id": "https://arxiv.org/abs/1", "title": "T1", "summary": "S", "authors": []},
                {"paper_id": "p2", "entry_id": "https://arxiv.org/abs/2", "title": "T2", "summary": "S", "authors": []},
            ],
            "interest_results": interest_results,
            "interest_reasons": {},
            "summaries": {"p1": {"title": "T1", "summary": "s", "keywords": [], "appendix": None}},
            "sent_paper_ids": sent,
            "discord_messages": {
                "p1": {
                    "message_id": "m1",
                    "channel_id": "c1",
                    "reaction_added": False,
                    "read_requested": False,
                    "read_window_expired": False,
                    "reading_memo_sent": False,
                    "paper_thread_id": None,
                    "qa_last_read_message_id": None,
                }
            },
            "reading_memos": {},
            "notification_sent": True,
            "retry_count": 0,
            "last_error": None,
            "created_at": "2026-11-19T00:00:00+00:00",
            "updated_at": "2026-11-19T00:00:00+00:00",
            "finalized_at": None,
        }

    def run_phase(self, job):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(main, "STATE_FILE_PATH", os.path.join(tmp, "state.json")):
                with mock.patch.object(main, "post_summary_to_discord") as post_summary:
                    with mock.patch.object(main, "add_read_reaction", return_value=True):
                        with mock.patch.object(main, "post_discord_payload", return_value=True):
                            result = main.run_send_phase(job, {"jobs": [job]})
        return result, post_summary

    def test_existing_message_state_prevents_double_posting(self):
        job = self.job({"p1": True, "p2": False}, [])
        result, post_summary = self.run_phase(job)
        self.assertTrue(result)
        post_summary.assert_not_called()
        self.assertEqual(job["sent_paper_ids"], ["p1"])

    def test_job_stays_processing_until_interest_complete(self):
        # p2 が未判定のまま全部投稿しても、完了扱いにしない
        job = self.job({"p1": True}, [])
        result, _ = self.run_phase(job)
        self.assertFalse(result)
        self.assertEqual(job["status"], "processing")
        self.assertEqual(job["sent_paper_ids"], ["p1"])

    def test_completes_when_interest_complete(self):
        job = self.job({"p1": True, "p2": False}, [])
        result, _ = self.run_phase(job)
        self.assertTrue(result)
        self.assertEqual(job["status"], "completed")


class PollCompositionTest(unittest.TestCase):
    def test_one_memo_does_not_fail_the_poll(self):
        with mock.patch.object(main, "DISCORD_WEBHOOK_URL", "https://example.com/hook"):
            with mock.patch.object(main, "DISCORD_BOT_TOKEN", "tok"):
                with mock.patch.object(main, "load_state", return_value={"jobs": []}):
                    with mock.patch.object(main, "process_daily_jobs", return_value=0) as pdj:
                        with mock.patch.object(main, "run_stage_poll_reading_requests", return_value=0):
                            with mock.patch.object(main, "run_stage_poll_paper_qa", return_value=0):
                                self.assertEqual(main.run_stage_poll(), 0)
                                pdj.assert_called_once()


class HelperTest(unittest.TestCase):
    def test_pdf_url_for_paper(self):
        self.assertEqual(
            main.pdf_url_for_paper({"entry_id": "http://arxiv.org/abs/2608.12345"}),
            "https://arxiv.org/pdf/2608.12345",
        )
        self.assertEqual(
            main.pdf_url_for_paper({"pdf_url": "https://arxiv.org/pdf/2608.999"}),
            "https://arxiv.org/pdf/2608.999",
        )

    def test_reaction_users_filter(self):
        users = [{"id": "bot", "bot": True}]
        self.assertFalse(main.reaction_users_include_request(users))
        users = [{"id": "user", "bot": False}]
        self.assertTrue(main.reaction_users_include_request(users, "user"))
        self.assertFalse(main.reaction_users_include_request(users, "other"))

    def test_embed_limits_hold_for_long_memo(self):
        memo = {
            "conclusion": "a" * 2000,
            "main_claims": "b" * 2000,
            "method_outline": "c" * 2000,
            "research_connection": "d" * 2000,
            "reading_guide": "e" * 2000,
            "follow_up_questions": ["f" * 1000] * 3,
        }
        embed = main.build_reading_memo_embed(
            {"title": "t" * 300, "entry_id": "https://arxiv.org/abs/2608.1"}, memo
        )
        self.assertLessEqual(main.discord_embed_text_length(embed), main.DISCORD_EMBED_TOTAL_LIMIT)
        for field in embed["fields"]:
            self.assertLessEqual(len(field["value"]), main.DISCORD_EMBED_FIELD_VALUE_LIMIT)

    def test_cap_text(self):
        self.assertEqual(main.cap_text("abc", 10), "abc")
        self.assertLessEqual(len(main.cap_text("x" * 20, 10)), 30)


if __name__ == "__main__":
    unittest.main()
