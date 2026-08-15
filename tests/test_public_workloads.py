from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from llama_lab.public_workloads import (
    PublicPrompt,
    build_trace_driven_replay,
    load_azure_trace,
    load_burstgpt_trace,
    load_longbench_records,
    longbench_qa_f1,
    select_longbench_prompts,
)


class PublicWorkloadsTest(unittest.TestCase):
    def test_longbench_qa_f1_matches_official_normalization(self) -> None:
        self.assertEqual(longbench_qa_f1("The New York!", "new york"), 1.0)
        self.assertAlmostEqual(longbench_qa_f1("alpha beta", "alpha gamma"), 0.5)

    def test_longbench_loader_requires_official_schema_and_stable_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "qasper_e.jsonl"
            rows = [
                {
                    "input": "question two", "context": "context two",
                    "answers": ["answer two"], "length": 9000,
                    "dataset": "qasper_e", "language": "en", "all_classes": [], "_id": "b",
                },
                {
                    "input": "question one", "context": "context one",
                    "answers": ["answer one"], "length": 3000,
                    "dataset": "qasper_e", "language": "en", "all_classes": [], "_id": "a",
                },
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            loaded = load_longbench_records(path, expected_dataset="qasper")

            self.assertEqual([row.record_id for row in loaded], ["a", "b"])
            self.assertEqual(loaded[0].answers, ("answer one",))
            bad = dict(rows[0])
            del bad["answers"]
            path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "LongBench schema"):
                load_longbench_records(path, expected_dataset="qasper")

    def test_longbench_selection_is_task_and_length_stratified(self) -> None:
        prompts = [
            PublicPrompt("qasper", "a", "en", "p1", ("a",), 1000, 120),
            PublicPrompt("qasper", "b", "en", "p2", ("b",), 7000, 700),
            PublicPrompt("multifieldqa_zh", "c", "zh", "p3", ("c",), 1000, 130),
            PublicPrompt("multifieldqa_zh", "d", "zh", "p4", ("d",), 9000, 900),
        ]

        selected = select_longbench_prompts(
            prompts, tasks=("qasper", "multifieldqa_zh"),
            local_token_buckets=((0, 256), (512, 1024)), per_task_bucket=1, seed=7,
        )

        self.assertEqual({row.record_id for row in selected}, {"a", "b", "c", "d"})
        self.assertEqual(len(selected), 4)

    def test_burstgpt_loader_preserves_contiguous_arrival_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "burst.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "Timestamp", "Session ID", "Elapsed time", "Model",
                    "Request tokens", "Response tokens", "Total tokens", "Log Type",
                ])
                writer.writeheader()
                writer.writerows([
                    {"Timestamp": "10.0", "Session ID": "s1", "Elapsed time": "1.0", "Model": "GPT-4", "Request tokens": "128", "Response tokens": "8", "Total tokens": "136", "Log Type": "Conversation log"},
                    {"Timestamp": "10.2", "Session ID": "s0", "Elapsed time": "0.0", "Model": "GPT-4", "Request tokens": "0", "Response tokens": "0", "Total tokens": "0", "Log Type": "Conversation log"},
                    {"Timestamp": "10.5", "Session ID": "s1", "Elapsed time": "1.2", "Model": "GPT-4", "Request tokens": "256", "Response tokens": "0", "Total tokens": "256", "Log Type": "Conversation log"},
                    {"Timestamp": "11.5", "Session ID": "", "Elapsed time": "0.8", "Model": "GPT-3.5", "Request tokens": "512", "Response tokens": "16", "Total tokens": "528", "Log Type": "API log"},
                ])

            rows = load_burstgpt_trace(path, start_row=0, request_count=2, include_failures=False)

            self.assertEqual([row.source_row for row in rows], [1, 4])
            self.assertEqual([row.arrival_seconds for row in rows], [0.0, 1.5])
            self.assertEqual(rows[1].input_tokens, 512)

    def test_azure_loader_rejects_negative_or_missing_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "azure.csv"
            path.write_text(
                "TIMESTAMP,ContextTokens,GeneratedTokens\n"
                "2024-05-10 00:00:00.009930+00:00,64,8\n"
                "2024-05-10 00:00:00.017335+00:00,128,16\n",
                encoding="utf-8",
            )
            rows = load_azure_trace(path, start_row=0, request_count=2)
            self.assertAlmostEqual(rows[0].arrival_seconds, 0.0)
            self.assertAlmostEqual(rows[1].arrival_seconds, 0.007405, places=6)
            path.write_text(
                "TIMESTAMP,ContextTokens,GeneratedTokens\n"
                "2024-05-10 00:00:00.009930+00:00,-1,8\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "positive token lengths"):
                load_azure_trace(path, start_row=0, request_count=1)

    def test_trace_replay_is_deterministic_and_does_not_claim_joint_provenance(self) -> None:
        prompts = [
            PublicPrompt("qasper", "a", "en", "short", ("a",), 100, 120),
            PublicPrompt("qasper", "b", "en", "long", ("b",), 1000, 520),
        ]
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.csv"
            trace.write_text(
                "TIMESTAMP,ContextTokens,GeneratedTokens\n"
                "2024-05-10 00:00:10+00:00,128,4\n"
                "2024-05-10 00:00:12+00:00,500,8\n",
                encoding="utf-8",
            )
            rows = load_azure_trace(trace, start_row=0, request_count=2)

        replay = build_trace_driven_replay(
            rows, prompts, seed=19, time_scale=0.1, maximum_output_tokens=6,
        )

        self.assertEqual([row.prompt_id for row in replay], ["a", "b"])
        self.assertEqual([row.arrival_seconds for row in replay], [0.0, 0.2])
        self.assertEqual([row.output_tokens for row in replay], [4, 6])
        self.assertEqual(replay[0].provenance, "trace-driven-public-content-synthetic-replay")


if __name__ == "__main__":
    unittest.main()
