"""A live, steerable optimization run, driven by a background thread.

`LiveSession` wraps an `OptimizationSession` and turns protocol messages into
steering calls (pin, reheat, set weights, …). Its state transitions
(`apply`, `tick`, `frame`) are plain synchronous methods; `start()` runs them
in a worker thread that steps the optimizer at a fixed frame rate and
publishes the latest frame to subscribers. Publishing is latest-only: a
subscriber that falls behind just sees the newest frame next time.

Like d3-force, the run *settles*: after each reheat (any interaction counts
as one) it steps for `settle_iters` iterations, then idles until the next
interaction, so an untouched layout does not burn CPU.
"""

import logging
import queue
import threading
import time
from collections.abc import Callable

import numpy as np

from ..session import OptimizationSession
from .protocol import (
    ClientMessage,
    DragEndMessage,
    DragMessage,
    DragStartMessage,
    FrameMessage,
    HelloMessage,
    HistoryPoint,
    Metrics,
    PauseMessage,
    ReheatMessage,
    ResetMessage,
    ResumeMessage,
    SetWeightMessage,
    UnpinMessage,
)

logger = logging.getLogger(__name__)


class LiveSession:
    """Serves one optimization session to interactive clients.

    Args:
        session: The session to drive; its problem must have a
            `scene_configuration`.
        steps_per_frame: Optimization steps between two published frames.
        fps: Target frames per second while running.
        settle_iters: Iterations to run after each interaction before
            idling; defaults to the session config's `n_iters` (the length
            of its learning-rate decay).
        max_history: Bound on the loss history kept for new clients; when
            exceeded, every other point is dropped (keeping the latest).

    Raises:
        ValueError: If the problem has no `scene_configuration`.
    """

    def __init__(
        self,
        session: OptimizationSession,
        steps_per_frame: int = 10,
        fps: float = 30.0,
        settle_iters: int | None = None,
        max_history: int = 2000,
    ) -> None:
        if session.problem.scene_configuration is None:
            raise ValueError("LiveSession needs a problem with a scene_configuration.")
        self.session = session
        self.steps_per_frame = steps_per_frame
        self.frame_interval = 1.0 / fps
        self.settle_iters = settle_iters or session.config.n_iters
        self.paused = False
        self._seed = session.config.seed
        self._settle_at = session.iteration + self.settle_iters
        self.max_history = max_history
        self._history: list[HistoryPoint] = []

        self._commands: queue.Queue[ClientMessage] = queue.Queue()
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._version = 0
        self._subscribers: list[Callable[[], None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- state ---

    @property
    def running(self) -> bool:
        """Whether the optimizer is stepping: not paused and not yet settled."""
        return not self.paused and self.session.iteration < self._settle_at

    def hello(self) -> HelloMessage:
        """The greeting sent to a newly connected client."""
        weights = self.session.weights
        return HelloMessage(
            terms=list(weights),
            weights=weights,
            adjustable_terms=[
                t.name for t in self.session.problem.terms if t.multiplier != 0.0
            ],
            learning_rate=float(self.session.config.learning_rate),
            steps_per_frame=self.steps_per_frame,
            history=self.history(),
        )

    def history(self) -> list[HistoryPoint]:
        """Loss values of the current run at past published frames (thread-safe)."""
        with self._lock:
            return list(self._history)

    def frame(self) -> FrameMessage:
        """Snapshot of the current state."""
        session = self.session
        metrics = None
        if session.last_step is not None:
            record = session.record()
            metrics = Metrics(
                total=record["total"],
                terms={t.name: record[t.name] for t in session.problem.terms},
            )
        return FrameMessage(
            iteration=session.iteration,
            running=self.running,
            paused=self.paused,
            scene=session.scene(),
            metrics=metrics,
            pinned=self._pinned_indices(),
            weights=session.weights,
        )

    def _pinned_indices(self) -> dict[str, list[int]]:
        pinned = {}
        for name in self.session.vars:
            mask = np.asarray(self.session.is_pinned(name))
            if mask.ndim == 0:
                continue
            rows = np.nonzero(mask.reshape(mask.shape[0], -1).any(axis=1))[0]
            if rows.size:
                pinned[name] = rows.tolist()
        return pinned

    # --- transitions ---

    def validate(self, message: ClientMessage) -> None:
        """Check that a message can be applied, without applying it.

        Raises:
            ValueError: If the message names an unknown variable, an
                out-of-range index, or a term whose weight cannot be changed.
        """
        if isinstance(
            message, DragStartMessage | DragMessage | DragEndMessage | UnpinMessage
        ):
            if message.var not in self.session.vars:
                raise ValueError(f"Unknown variable: {message.var!r}")
            shape = np.shape(self.session.is_pinned(message.var))
            if message.index is not None and not (
                len(shape) >= 1 and 0 <= message.index < shape[0]
            ):
                raise ValueError(
                    f"Index {message.index} out of range for {message.var!r} "
                    f"of shape {shape}."
                )
        elif isinstance(message, SetWeightMessage):
            adjustable = {
                t.name for t in self.session.problem.terms if t.multiplier != 0.0
            }
            if message.name not in adjustable:
                raise ValueError(f"Term {message.name!r} cannot be adjusted live.")

    def apply(self, message: ClientMessage) -> None:
        """Apply a client message to the session.

        Every message except pause counts as an interaction and reheats the
        run. Call `validate` first for messages from untrusted clients.
        """
        session = self.session
        match message:
            case DragStartMessage(var=var, index=index):
                session.pin(var, index)
            case DragMessage(var=var, index=index, x=x, y=y):
                session.pin(var, index, value=[x, y])
            case DragEndMessage(var=var, index=index, keep_pinned=keep_pinned):
                if not keep_pinned:
                    session.unpin(var, index)
            case UnpinMessage(var=var, index=index):
                session.unpin(var, index)
            case PauseMessage():
                self.paused = True
                return
            case ResumeMessage():
                self.paused = False
            case ReheatMessage():
                pass
            case SetWeightMessage(name=name, value=value):
                session.set_weight(name, value)
            case ResetMessage(seed=seed):
                self._seed = self._seed + 1 if seed is None else seed
                self.session = session.problem.session(session.config, self._seed)
                self._settle_at = self.session.iteration + self.settle_iters
                with self._lock:
                    self._history = []
                return
        self.reheat()

    def reheat(self) -> None:
        """Restart the learning-rate decay and the settle countdown."""
        self.session.reheat()
        self._settle_at = self.session.iteration + self.settle_iters

    def tick(self) -> bool:
        """Advance by one frame's worth of steps, if running.

        Returns:
            Whether any step was performed.
        """
        if not self.running:
            return False
        n = min(self.steps_per_frame, self._settle_at - self.session.iteration)
        self.session.step(n)
        return True

    # --- threading ---

    def submit(self, message: ClientMessage) -> None:
        """Queue a message for the worker thread (thread-safe)."""
        self._commands.put(message)

    def subscribe(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Register a callback invoked (from the worker thread) on each new frame.

        Returns:
            A function that removes the subscription.
        """
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def latest(self) -> tuple[int, dict | None]:
        """The latest published frame as a JSON dict, with its version number."""
        with self._lock:
            return self._version, self._latest

    def publish(self) -> None:
        """Serialize the current frame and notify subscribers."""
        frame = self.frame()
        payload = frame.model_dump(mode="json", exclude_none=True)
        with self._lock:
            self._latest = payload
            self._version += 1
            subscribers = list(self._subscribers)
            if frame.metrics is not None and (
                not self._history or self._history[-1].iteration != frame.iteration
            ):
                self._history.append(
                    HistoryPoint(
                        iteration=frame.iteration, **frame.metrics.model_dump()
                    )
                )
                if len(self._history) > self.max_history:
                    last = self._history[-1]
                    self._history = self._history[:-1:2] + [last]
        for callback in subscribers:
            callback()

    def start(self) -> None:
        """Start the worker thread."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the worker thread and wait for it to finish."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _drain(self, timeout: float) -> bool:
        """Apply queued messages, waiting up to `timeout` for the first one."""
        changed = False
        try:
            message = self._commands.get(timeout=timeout) if timeout > 0 else None
            while True:
                if message is None:
                    message = self._commands.get_nowait()
                try:
                    self.apply(message)
                    changed = True
                except (KeyError, ValueError, IndexError) as error:
                    # validate() runs before submit(), but a reset may race it.
                    logger.warning("Ignoring %r: %s", message, error)
                message = None
        except queue.Empty:
            pass
        return changed

    def _run(self) -> None:
        self.publish()
        while not self._stop.is_set():
            deadline = time.monotonic() + self.frame_interval
            changed = self._drain(timeout=0 if self.running else self.frame_interval)
            if self.tick():
                changed = True
            if changed:
                self.publish()
            self._stop.wait(max(0.0, deadline - time.monotonic()))
