import unittest

from llama_lab.bench import parse_llama_bench_json
from llama_lab.quality import score_response
from llama_lab.streaming import parse_sse_lines


class ParsingTests(unittest.TestCase):
    def test_parses_llama_bench_list(self) -> None:
        rows = parse_llama_bench_json('[{"avg_ts": 42.0}]')
        self.assertEqual(rows[0]["avg_ts"], 42.0)

    def test_rejects_non_object_rows(self) -> None:
        with self.assertRaises(ValueError):
            parse_llama_bench_json('[1, 2, 3]')

    def test_parses_stream_and_stops_at_done(self) -> None:
        events = list(
            parse_sse_lines(
                [
                    b": keepalive\n",
                    b'data: {"choices":[{"delta":{"content":"hi"}}]}\n',
                    b"data: [DONE]\n",
                    b'data: {"ignored":true}\n',
                ]
            )
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["choices"][0]["delta"]["content"], "hi")

    def test_quality_score_requires_every_pattern(self) -> None:
        self.assertTrue(score_response("输入必须有序，复杂度 O(log n)", ["有序", "log"]))
        self.assertFalse(score_response("复杂度 O(log n)", ["有序", "log"]))
        self.assertFalse(
            score_response(
                "数组长度是偶数且有序，复杂度 O(log n)",
                ["有序", "log"],
                ["长度是偶数"],
            )
        )


if __name__ == "__main__":
    unittest.main()
