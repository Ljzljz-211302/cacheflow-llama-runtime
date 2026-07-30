import threading
import unittest

from cacheflow.checkpoint import CheckpointStore
from cacheflow.domain import QueueFullError, SlotRecord
from cacheflow.executor import ExecutorResult
from cacheflow.routing import ModelArchitecture, ModelProfile, ModelRouter
from cacheflow.scheduler import SchedulerCore
from cacheflow.service import CacheFlowEngine


class FakeExecutor:
    def __init__(self) -> None:
        self.slot_tokens: dict[int, tuple[int, ...]] = {}
        self.checkpoints: dict[str, tuple[int, ...]] = {}
        self.saved: list[str] = []
        self.restored: list[str] = []
        self.deleted: list[str] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.fail_save = False

    def tokenize_messages(self, messages):
        text = "\n".join(str(message.get("content", "")) for message in messages)
        return tuple(text.encode("utf-8"))

    def complete(self, payload, slot_id, prompt_tokens):
        self.started.set()
        if not self.release.wait(2):
            raise TimeoutError("fake executor remained blocked")
        previous = self.slot_tokens.get(slot_id, ())
        reused = 0
        for left, right in zip(previous, prompt_tokens):
            if left != right:
                break
            reused += 1
        self.slot_tokens[slot_id] = prompt_tokens
        processed = max(len(prompt_tokens) - reused, 1)
        return ExecutorResult(
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            processed,
            float(processed),
            2.0,
            prompt_tokens,
        )

    def save_slot(self, slot_id, filename):
        if self.fail_save:
            raise OSError("disk full")
        tokens = self.slot_tokens.get(slot_id, ())
        self.checkpoints[filename] = tokens
        self.saved.append(filename)
        return len(tokens), len(tokens) * 8

    def restore_slot(self, slot_id, filename):
        tokens = self.checkpoints[filename]
        self.slot_tokens[slot_id] = tokens
        self.restored.append(filename)
        return len(tokens)

    def erase_slot(self, slot_id):
        return len(self.slot_tokens.pop(slot_id, ()))

    def delete_checkpoint(self, filename):
        self.checkpoints.pop(filename, None)
        self.deleted.append(filename)


def make_engine(executor, *, max_queue=8, checkpoint_budget=10_000):
    profile = ModelProfile(
        "q4",
        ModelArchitecture(28, 4, 128, 4096),
        quality_score=0.8,
        weight_mib=600,
        runtime_mib=100,
        cache_bytes_per_element=2,
        prefill_ms_per_token=1,
        decode_ms_per_token=2,
        max_slots=1,
    )
    return CacheFlowEngine(
        router=ModelRouter([profile]),
        executors={"q4": executor},
        scheduler=SchedulerCore([SlotRecord(0, "q4")], max_queue=max_queue),
        checkpoints=CheckpointStore(checkpoint_budget),
        tokenizer_model="q4",
        checkpoint_min_tokens=4,
    )


def payload(text):
    return {"messages": [{"role": "user", "content": text}], "max_tokens": 4}


class CacheFlowServiceTests(unittest.TestCase):
    def test_request_runs_through_router_scheduler_and_executor(self):
        executor = FakeExecutor()
        with make_engine(executor) as engine:
            result = engine.submit(payload("hello"), conversation_id="a").result(2)
            self.assertEqual(result.model, "q4")
            self.assertEqual(result.slot_id, 0)
            self.assertEqual(result.prompt_tokens_processed, 5)
            self.assertEqual(engine.snapshot()["scheduler"]["queue_depth"], 0)

    def test_bounded_queue_rejects_excess_work(self):
        executor = FakeExecutor()
        executor.release.clear()
        with make_engine(executor, max_queue=1) as engine:
            running = engine.submit(payload("first"), conversation_id="a")
            self.assertTrue(executor.started.wait(1))
            queued = engine.submit(payload("second"), conversation_id="b")
            rejected = engine.submit(payload("third"), conversation_id="c")
            with self.assertRaises(QueueFullError):
                rejected.result(1)
            executor.release.set()
            running.result(2)
            queued.result(2)

    def test_l2_checkpoint_restores_evicted_conversation(self):
        executor = FakeExecutor()
        with make_engine(executor) as engine:
            engine.submit(payload("alpha-history"), conversation_id="a").result(2)
            engine.submit(payload("bravo-history"), conversation_id="b").result(2)
            revisited = engine.submit(
                payload("alpha-history-more"), conversation_id="a"
            ).result(2)
            self.assertTrue(revisited.checkpoint_restored)
            self.assertEqual(len(executor.saved), 2)
            self.assertEqual(len(executor.restored), 1)
            self.assertLess(revisited.prompt_tokens_processed, len("alpha-history-more"))

    def test_checkpoint_save_failure_degrades_to_recompute(self):
        executor = FakeExecutor()
        with make_engine(executor) as engine:
            engine.submit(payload("alpha-history"), conversation_id="a").result(2)
            executor.fail_save = True
            result = engine.submit(payload("bravo-history"), conversation_id="b").result(2)
            self.assertFalse(result.checkpoint_restored)
            metrics = engine.snapshot()["metrics"]
            self.assertEqual(metrics["l2_save_failures_total"], 1)


if __name__ == "__main__":
    unittest.main()
