import unittest

from cacheflow.checkpoint import CheckpointEntry, CheckpointStore


class CheckpointStoreTests(unittest.TestCase):
    def test_budget_evicts_least_recently_used_entry(self) -> None:
        store = CheckpointStore(100)
        old = CheckpointEntry("old", "m", "old.bin", 60, (1,), 1)
        recent = CheckpointEntry("recent", "m", "recent.bin", 60, (2,), 2)
        self.assertEqual(store.register(old), [])
        self.assertEqual(store.register(recent), [old])
        self.assertEqual(store.used_bytes, 60)

    def test_get_refreshes_lru_and_replacement_does_not_leak_bytes(self) -> None:
        store = CheckpointStore(120)
        store.register(CheckpointEntry("a", "m", "a.bin", 50, (1,), 1))
        store.register(CheckpointEntry("b", "m", "b.bin", 50, (2,), 2))
        store.get("m", "a", 3)
        evicted = store.register(CheckpointEntry("c", "m", "c.bin", 50, (3,), 4))
        self.assertEqual([entry.conversation_id for entry in evicted], ["b"])
        store.register(CheckpointEntry("a", "m", "a2.bin", 40, (1, 2), 5))
        self.assertEqual(store.used_bytes, 90)

    def test_filename_does_not_expose_conversation_identifier(self) -> None:
        filename = CheckpointStore.filename_for("model", "../../private/user")
        self.assertRegex(filename, r"^cacheflow-[0-9a-f]{24}\.bin$")


if __name__ == "__main__":
    unittest.main()
