import unittest

from cacheflow.domain import QueueFullError, RequestRecord, RequestState, SlotRecord
from cacheflow.policy import AgingCachePolicy
from cacheflow.prefix_index import PrefixIndex
from cacheflow.scheduler import SchedulerCore


def request(
    request_id: str,
    conversation_id: str,
    tokens: tuple[int, ...],
    created: float,
    deadline: float = 100,
) -> RequestRecord:
    item = RequestRecord(
        request_id,
        conversation_id,
        "model",
        tokens,
        {},
        created,
        deadline,
        state_changed_at=created,
    )
    item.transition(RequestState.TOKENIZING, created)
    return item


class PrefixIndexTests(unittest.TestCase):
    def test_upsert_remove_and_longest_matches(self) -> None:
        index = PrefixIndex()
        index.upsert(0, (1, 2, 3, 4))
        index.upsert(1, (1, 2, 9))
        self.assertEqual(index.match_lengths((1, 2, 3, 8)), {0: 3, 1: 2})
        index.upsert(0, (7, 8))
        self.assertEqual(index.match_lengths((1, 2, 3)), {1: 2})
        index.remove(1)
        self.assertEqual(len(index), 1)


class RequestStateTests(unittest.TestCase):
    def test_terminal_state_rejects_further_transition(self) -> None:
        item = request("r", "c", (1,), 0)
        item.transition(RequestState.QUEUED, 1)
        item.transition(RequestState.CANCELLED, 2)
        with self.assertRaisesRegex(ValueError, "terminal request"):
            item.transition(RequestState.RUNNING, 3)


class SchedulerCoreTests(unittest.TestCase):
    def test_backpressure_rejects_beyond_queue_limit(self) -> None:
        core = SchedulerCore([SlotRecord(0, "model", running_request_id="busy")], max_queue=1)
        core.submit(request("a", "a", (1,), 0), 0)
        with self.assertRaises(QueueFullError):
            core.submit(request("b", "b", (2,), 0), 0)

    def test_cache_score_selects_matching_slot_and_updates_index(self) -> None:
        core = SchedulerCore(
            [
                SlotRecord(0, "model", "old", (1, 2, 3, 4), last_used=1),
                SlotRecord(1, "model", "other", (9, 9), last_used=0),
            ],
            policy=AgingCachePolicy(eviction_penalty=0),
        )
        core.submit(request("r", "new", (1, 2, 3, 8), 2), 2)
        decision = core.plan(2)[0]
        self.assertEqual(decision.slot_id, 0)
        self.assertEqual(decision.reused_tokens, 3)
        core.mark_running("r", 2.1)
        core.complete("r", 3, cached_tokens=(1, 2, 3, 8, 10), result={"ok": True})
        self.assertEqual(core.requests["r"].state, RequestState.COMPLETED)
        self.assertEqual(core.indexes["model"].match_lengths((1, 2, 3, 8, 11))[0], 4)

    def test_aging_prevents_hot_request_from_starving_old_request(self) -> None:
        core = SchedulerCore(
            [SlotRecord(0, "model", "hot", (1, 2, 3), last_used=1)],
            policy=AgingCachePolicy(eviction_penalty=0, wait_age_weight=0, max_wait_ms=100),
        )
        core.submit(request("old", "cold", (8, 8), 0, deadline=10), 0)
        core.submit(request("hot", "hot", (1, 2, 3, 4), 0.95, deadline=10), 0.95)
        decision = core.plan(1.0)[0]
        self.assertEqual(decision.request_id, "old")
        self.assertTrue(decision.urgent)

    def test_expiry_and_cancel_leave_no_dispatchable_request(self) -> None:
        core = SchedulerCore([SlotRecord(0, "model")])
        core.submit(request("expired", "c", (1,), 0, deadline=1), 0)
        self.assertEqual(core.expire(2), ["expired"])
        core.submit(request("cancelled", "c", (1,), 2, deadline=10), 2)
        self.assertTrue(core.cancel("cancelled", 3))
        self.assertEqual(core.plan(3), [])


if __name__ == "__main__":
    unittest.main()
