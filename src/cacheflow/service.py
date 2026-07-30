from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .checkpoint import CheckpointEntry, CheckpointStore
from .domain import QueueFullError, RequestRecord, RequestState, SchedulingDecision
from .executor import Executor, ExecutorResult
from .observability import MetricsRegistry
from .routing import ModelRouter, RouteRequest, RoutingDecision
from .scheduler import SchedulerCore


@dataclass(frozen=True)
class CacheFlowResult:
    request_id: str
    model: str
    slot_id: int
    response: dict[str, Any]
    routing: RoutingDecision
    queue_ms: float
    total_ms: float
    prompt_tokens_processed: int
    checkpoint_restored: bool


class RequestHandle:
    def __init__(self, engine: "CacheFlowEngine", request_id: str) -> None:
        self._engine = engine
        self.request_id = request_id
        self._event = threading.Event()
        self._result: CacheFlowResult | None = None
        self._error: BaseException | None = None

    def _resolve(
        self, result: CacheFlowResult | None = None, error: BaseException | None = None
    ) -> None:
        self._result = result
        self._error = error
        self._event.set()

    def result(self, timeout: float | None = None) -> CacheFlowResult:
        if not self._event.wait(timeout):
            raise TimeoutError(f"request did not finish: {self.request_id}")
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result

    def cancel(self) -> bool:
        return self._engine.cancel(self.request_id)


class CacheFlowEngine:
    def __init__(
        self,
        *,
        router: ModelRouter,
        executors: dict[str, Executor],
        scheduler: SchedulerCore,
        checkpoints: CheckpointStore,
        tokenizer_model: str,
        metrics: MetricsRegistry | None = None,
        workers: int | None = None,
        checkpoint_min_tokens: int = 64,
        clock=time.monotonic,
    ) -> None:
        missing = set(router.profiles) - set(executors)
        if missing:
            raise ValueError(f"missing executors for models: {sorted(missing)}")
        if tokenizer_model not in executors:
            raise ValueError(f"tokenizer executor missing: {tokenizer_model}")
        self.router = router
        self.executors = executors
        self.scheduler = scheduler
        self.checkpoints = checkpoints
        self.tokenizer_model = tokenizer_model
        self.metrics = metrics or MetricsRegistry()
        self.checkpoint_min_tokens = checkpoint_min_tokens
        self.clock = clock
        self._condition = threading.Condition(threading.RLock())
        self._handles: dict[str, RequestHandle] = {}
        self._routing: dict[str, RoutingDecision] = {}
        self._stopping = False
        self._workers = ThreadPoolExecutor(
            max_workers=workers or len(self.scheduler.slots),
            thread_name_prefix="cacheflow-worker",
        )
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop, name="cacheflow-dispatcher", daemon=True
        )
        self._dispatcher.start()

    def submit(
        self,
        payload: dict[str, Any],
        *,
        conversation_id: str,
        quality_floor: float = 0.0,
        latency_slo_ms: float = 2000.0,
        timeout_ms: float = 30_000.0,
        available_vram_mib: float = float("inf"),
    ) -> RequestHandle:
        request_id = f"cf-{uuid.uuid4().hex}"
        created = self.clock()
        handle = RequestHandle(self, request_id)
        record = RequestRecord(
            request_id=request_id,
            conversation_id=conversation_id,
            model="",
            tokens=(),
            payload=dict(payload),
            created_at=created,
            deadline_at=created + timeout_ms / 1000,
            state_changed_at=created,
        )
        record.transition(RequestState.TOKENIZING, created)
        try:
            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a non-empty list")
            tokens = self.executors[self.tokenizer_model].tokenize_messages(messages)
            # Queued work must remain admissible while every backend slot is
            # busy.  Use current concurrency as the cost-model signal; actual
            # capacity and backpressure are enforced by SchedulerCore.
            active = max(1, sum(
                slot.running_request_id is not None
                for slot in self.scheduler.slots.values()
            ))
            routing_request = RouteRequest(
                input_tokens=len(tokens),
                output_tokens=int(payload.get("max_tokens", 64)),
                quality_floor=quality_floor,
                latency_slo_ms=latency_slo_ms,
                active_sequences=active,
                available_vram_mib=available_vram_mib,
            )
            routing = self.router.route(routing_request)
            record.model = routing.selected_model
            record.tokens = tokens
            with self._condition:
                if self._stopping:
                    raise RuntimeError("CacheFlow is stopping")
                self.scheduler.submit(record, self.clock())
                self._handles[request_id] = handle
                self._routing[request_id] = routing
                self.metrics.increment("requests_admitted_total")
                self.metrics.gauge("queue_depth", len(self.scheduler.pending))
                self.metrics.increment(f"model_route_{routing.selected_model}_total")
                self._condition.notify_all()
        except BaseException as exc:
            self.metrics.increment("requests_rejected_total")
            handle._resolve(error=exc)
        return handle

    def cancel(self, request_id: str) -> bool:
        with self._condition:
            cancelled = self.scheduler.cancel(request_id, self.clock())
            if cancelled:
                self.metrics.increment("requests_cancelled_total")
                self.metrics.gauge("queue_depth", len(self.scheduler.pending))
                self._handles[request_id]._resolve(error=RuntimeError("request cancelled"))
                self._condition.notify_all()
            return cancelled

    def _dispatch_loop(self) -> None:
        while True:
            with self._condition:
                if self._stopping:
                    return
                expired = self.scheduler.expire(self.clock())
                for request_id in expired:
                    self.metrics.increment("requests_timed_out_total")
                    self._handles[request_id]._resolve(
                        error=TimeoutError("request expired in queue")
                    )
                decisions = self.scheduler.plan(self.clock())
                self.metrics.gauge("queue_depth", len(self.scheduler.pending))
                self.metrics.gauge(
                    "running_slots",
                    sum(
                        slot.running_request_id is not None
                        for slot in self.scheduler.slots.values()
                    ),
                )
                for decision in decisions:
                    self._workers.submit(self._execute, decision)
                self._condition.wait(timeout=0.05)

    def _save_evicted_slot(self, decision: SchedulingDecision, now: float) -> None:
        slot = self.scheduler.slots[decision.slot_id]
        if (
            slot.conversation_id is None
            or len(slot.tokens) < self.checkpoint_min_tokens
            or slot.conversation_id
            == self.scheduler.requests[decision.request_id].conversation_id
        ):
            return
        executor = self.executors[slot.model]
        filename = CheckpointStore.filename_for(slot.model, slot.conversation_id)
        try:
            _, written = executor.save_slot(slot.executor_slot_id, filename)
            entry = CheckpointEntry(
                slot.conversation_id,
                slot.model,
                filename,
                written,
                slot.tokens,
                now,
            )
            with self._condition:
                victims = self.checkpoints.register(entry)
                self.metrics.increment("l2_saves_total")
                self.metrics.increment("l2_bytes_written_total", written)
                self.metrics.gauge("l2_bytes", self.checkpoints.used_bytes)
            for victim in victims:
                self.executors[victim.model].delete_checkpoint(victim.filename)
                self.metrics.increment("l2_evictions_total")
        except Exception:
            self.metrics.increment("l2_save_failures_total")

    def _execute(self, decision: SchedulingDecision) -> None:
        started = self.clock()
        with self._condition:
            request = self.scheduler.requests[decision.request_id]
            slot = self.scheduler.slots[decision.slot_id]
            executor = self.executors[request.model]
            checkpoint = self.checkpoints.get(
                request.model, request.conversation_id, started
            )
        try:
            self._save_evicted_slot(decision, started)
            restored = False
            if checkpoint is not None and slot.conversation_id != request.conversation_id:
                with self._condition:
                    self.scheduler.mark_restoring(request.request_id, self.clock())
                executor.restore_slot(slot.executor_slot_id, checkpoint.filename)
                restored = True
                self.metrics.increment("l2_restores_total")
                cached_tokens = checkpoint.tokens
            else:
                cached_tokens = slot.tokens
                if slot.conversation_id == request.conversation_id:
                    self.metrics.increment("l1_hits_total")
                else:
                    self.metrics.increment("cache_misses_total")
            with self._condition:
                self.scheduler.mark_running(request.request_id, self.clock())

            result: ExecutorResult = executor.complete(
                request.payload, slot.executor_slot_id, request.tokens
            )
            finished = self.clock()
            with self._condition:
                self.scheduler.complete(
                    request.request_id,
                    finished,
                    cached_tokens=result.cached_tokens,
                    result=result.response,
                )
                route_request = RouteRequest(
                    input_tokens=len(request.tokens),
                    output_tokens=int(request.payload.get("max_tokens", 64)),
                    quality_floor=0,
                    latency_slo_ms=max((request.deadline_at - request.created_at) * 1000, 1),
                )
                self.router.observe(
                    request.model,
                    route_request,
                    actual_prefill_ms=result.prompt_ms,
                    actual_decode_ms=result.decode_ms,
                )
                queue_ms = max(0.0, (started - request.created_at) * 1000)
                total_ms = (finished - request.created_at) * 1000
                response = CacheFlowResult(
                    request.request_id,
                    request.model,
                    slot.slot_id,
                    result.response,
                    self._routing[request.request_id],
                    queue_ms,
                    total_ms,
                    result.prompt_tokens_processed,
                    restored,
                )
                self.metrics.increment("requests_completed_total")
                self.metrics.increment("prompt_tokens_processed_total", result.prompt_tokens_processed)
                self.metrics.observe("queue_ms", queue_ms)
                self.metrics.observe("request_total_ms", total_ms)
                self.metrics.gauge(
                    "running_slots",
                    sum(
                        item.running_request_id is not None
                        for item in self.scheduler.slots.values()
                    ),
                )
                self._handles[request.request_id]._resolve(result=response)
                self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                try:
                    self.scheduler.fail(request.request_id, self.clock(), str(exc))
                except ValueError:
                    pass
                self.metrics.increment("backend_failures_total")
                self._handles[request.request_id]._resolve(error=exc)
                self._condition.notify_all()

    def snapshot(self) -> dict:
        with self._condition:
            return {
                "scheduler": self.scheduler.snapshot(),
                "checkpoints": self.checkpoints.snapshot(),
                "metrics": self.metrics.snapshot(),
            }

    def shutdown(self, wait: bool = True) -> None:
        with self._condition:
            self._stopping = True
            for request_id in list(self.scheduler.pending):
                self.scheduler.cancel(request_id, self.clock())
                self._handles[request_id]._resolve(
                    error=RuntimeError("CacheFlow shut down")
                )
            self._condition.notify_all()
        self._dispatcher.join(timeout=5)
        self._workers.shutdown(wait=wait, cancel_futures=False)

    def __enter__(self) -> "CacheFlowEngine":
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()
