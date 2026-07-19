"""Typed HTTP/WebSocket bridge for the Expo Go monitoring application."""

from __future__ import annotations

import asyncio
import socket
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from threading import RLock
from time import monotonic, time_ns
from typing import Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from sightly_assist.depth_association import DepthStatus
from sightly_assist.free_space import CorridorDirection, CorridorStatus
from sightly_assist.motion_estimation import MotionStatus
from sightly_assist.perception_risk import PerceptionRiskStatus
from sightly_assist.replay_pipeline import ReplayResult
from sightly_assist.warning_policy import AlertAction


class MobileBoundingBox(BaseModel):
    """Normalized image-space box used by the phone preview overlay."""

    x_min: float = Field(ge=0, le=1)
    y_min: float = Field(ge=0, le=1)
    x_max: float = Field(ge=0, le=1)
    y_max: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_extents(self) -> MobileBoundingBox:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("mobile bounding box must have positive width and height")
        return self


class MobileHazard(BaseModel):
    """One tracked object shown in the mobile interface."""

    track_id: int = Field(ge=0)
    class_name: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    bbox: MobileBoundingBox
    depth_m: float | None = Field(default=None, gt=0)
    velocity_x_mps: float | None = None
    velocity_z_mps: float | None = None
    speed_mps: float | None = Field(default=None, ge=0)
    risk_score: float | None = Field(default=None, ge=0, le=1)
    time_to_closest_approach_s: float | None = Field(default=None, ge=0)
    distance_at_closest_approach_m: float | None = Field(default=None, ge=0)
    predicted_collision: bool | None = None
    assessment_status: str = Field(min_length=1)


class MobileFreeSpace(BaseModel):
    """Conservative left, center, and right corridor states."""

    left: CorridorStatus = CorridorStatus.UNKNOWN
    center: CorridorStatus = CorridorStatus.UNKNOWN
    right: CorridorStatus = CorridorStatus.UNKNOWN


class MobileMetrics(BaseModel):
    """Current end-to-end and stage-level performance values."""

    processing_fps: float = Field(ge=0)
    total_latency_ms: float = Field(ge=0)
    detector_latency_ms: float = Field(ge=0)
    tracker_latency_ms: float = Field(ge=0)
    depth_latency_ms: float = Field(ge=0)
    motion_latency_ms: float = Field(ge=0)
    risk_latency_ms: float = Field(ge=0)


class MobileSystemState(BaseModel):
    """Single versioned telemetry snapshot sent to Expo clients."""

    schema_version: Literal["1.0"] = "1.0"
    source: str = Field(min_length=1)
    sequence_id: str = Field(min_length=1)
    frame_id: int = Field(ge=0)
    timestamp_ns: int = Field(ge=0)
    warning_action: AlertAction
    warning_should_emit: bool
    warning_reason: str = Field(min_length=1)
    free_space: MobileFreeSpace
    hazards: tuple[MobileHazard, ...] = ()
    metrics: MobileMetrics
    synchronized: bool = True
    orientation_available: bool = False
    system_message: str = Field(min_length=1)


class MobileHealth(BaseModel):
    """Small readiness response used before opening a WebSocket."""

    status: Literal["ok"] = "ok"
    service: Literal["sightly-mobile-bridge"] = "sightly-mobile-bridge"
    schema_version: Literal["1.0"] = "1.0"
    source: str
    frame_id: int = Field(ge=0)
    history_size: int = Field(ge=0)


class MobileStateStore:
    """Thread-safe latest-state store with a bounded telemetry history."""

    def __init__(
        self, initial_state: MobileSystemState | None = None, history_size: int = 300
    ) -> None:
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        self._lock = RLock()
        self._history: deque[MobileSystemState] = deque(maxlen=history_size)
        self._version = 0
        self._latest = initial_state or idle_mobile_state()
        self._history.append(self._latest)

    def publish(self, state: MobileSystemState) -> int:
        """Atomically publish a new state and return its monotonically increasing version."""

        with self._lock:
            self._latest = state
            self._history.append(state)
            self._version += 1
            return self._version

    def snapshot(self) -> tuple[int, MobileSystemState]:
        """Return the current version and state as one consistent snapshot."""

        with self._lock:
            return self._version, self._latest

    def history(self, limit: int = 100) -> tuple[MobileSystemState, ...]:
        """Return at most ``limit`` newest states in chronological order."""

        if limit <= 0:
            raise ValueError("limit must be positive")
        with self._lock:
            items = tuple(self._history)
        return items[-limit:]


def idle_mobile_state() -> MobileSystemState:
    """Create a safe startup snapshot before a producer publishes data."""

    return MobileSystemState(
        source="idle",
        sequence_id="not-started",
        frame_id=0,
        timestamp_ns=time_ns(),
        warning_action=AlertAction.ABSTAIN,
        warning_should_emit=False,
        warning_reason="waiting for perception data",
        free_space=MobileFreeSpace(),
        metrics=MobileMetrics(
            processing_fps=0,
            total_latency_ms=0,
            detector_latency_ms=0,
            tracker_latency_ms=0,
            depth_latency_ms=0,
            motion_latency_ms=0,
            risk_latency_ms=0,
        ),
        synchronized=False,
        orientation_available=False,
        system_message="Backend connected; no perception frames have arrived.",
    )


def mobile_state_from_replay_result(
    result: ReplayResult,
    *,
    source: str = "replay",
    sequence_id: str = "replay-sequence",
) -> MobileSystemState:
    """Convert the existing perception result into the stable mobile schema."""

    depth_by_track = {item.track_id: item for item in result.depth_associations}
    motion_by_track = {item.track_id: item for item in result.motion_estimates}
    risk_by_track = {item.track_id: item for item in result.risk_results}
    hazards: list[MobileHazard] = []

    for track in result.tracks:
        depth = depth_by_track.get(track.track_id)
        motion = motion_by_track.get(track.track_id)
        risk = risk_by_track.get(track.track_id)
        assessment = risk.assessment if risk is not None else None
        velocity = (
            motion.velocity if motion is not None and motion.status is MotionStatus.VALID else None
        )
        assessment_status = (
            risk.status.value if risk is not None else PerceptionRiskStatus.MOTION_UNAVAILABLE.value
        )
        hazards.append(
            MobileHazard(
                track_id=track.track_id,
                class_name=track.class_name,
                confidence=track.confidence,
                bbox=MobileBoundingBox(
                    x_min=_normalize(track.bbox.x_min, result.frame.width_px),
                    y_min=_normalize(track.bbox.y_min, result.frame.height_px),
                    x_max=_normalize(track.bbox.x_max, result.frame.width_px),
                    y_max=_normalize(track.bbox.y_max, result.frame.height_px),
                ),
                depth_m=(
                    depth.depth_m
                    if depth is not None and depth.status is DepthStatus.VALID
                    else None
                ),
                velocity_x_mps=velocity.x_mps if velocity is not None else None,
                velocity_z_mps=velocity.z_mps if velocity is not None else None,
                speed_mps=velocity.speed_mps if velocity is not None else None,
                risk_score=assessment.risk_score if assessment is not None else None,
                time_to_closest_approach_s=(
                    assessment.time_to_closest_approach_s if assessment is not None else None
                ),
                distance_at_closest_approach_m=(
                    assessment.distance_at_closest_approach_m if assessment is not None else None
                ),
                predicted_collision=(
                    assessment.predicted_collision if assessment is not None else None
                ),
                assessment_status=assessment_status,
            )
        )

    warning = result.warning_decision
    timings = result.stage_timings
    total_ms = timings.total_ms if timings is not None else 0.0
    free_space = MobileFreeSpace()
    if result.free_space_analysis is not None:
        by_direction = {
            item.direction: item.status for item in result.free_space_analysis.corridors
        }
        free_space = MobileFreeSpace(
            left=by_direction.get(CorridorDirection.LEFT, CorridorStatus.UNKNOWN),
            center=by_direction.get(CorridorDirection.CENTER, CorridorStatus.UNKNOWN),
            right=by_direction.get(CorridorDirection.RIGHT, CorridorStatus.UNKNOWN),
        )

    return MobileSystemState(
        source=source,
        sequence_id=sequence_id,
        frame_id=result.frame.frame_id,
        timestamp_ns=result.frame.timestamp_ns,
        warning_action=warning.action if warning is not None else AlertAction.NO_ALERT,
        warning_should_emit=warning.should_emit if warning is not None else False,
        warning_reason=warning.reason if warning is not None else "warning policy not enabled",
        free_space=free_space,
        hazards=tuple(hazards),
        metrics=MobileMetrics(
            processing_fps=1000.0 / total_ms if total_ms > 0 else 0.0,
            total_latency_ms=total_ms,
            detector_latency_ms=timings.detector_ms if timings is not None else 0.0,
            tracker_latency_ms=timings.tracker_ms if timings is not None else 0.0,
            depth_latency_ms=(
                timings.depth_load_ms + timings.depth_association_ms if timings is not None else 0.0
            ),
            motion_latency_ms=timings.motion_ms if timings is not None else 0.0,
            risk_latency_ms=(timings.risk_ms + timings.warning_ms if timings is not None else 0.0),
        ),
        synchronized=result.frame.synchronized,
        orientation_available=result.frame.orientation_wxyz is not None,
        system_message=_system_message(
            warning.action if warning is not None else AlertAction.NO_ALERT
        ),
    )


class DemoStateGenerator:
    """Deterministic, looping hazard sequence for Expo testing without hardware."""

    def __init__(self) -> None:
        self._started_at = monotonic()
        self._frame_id = 0

    def reset(self) -> MobileSystemState:
        self._started_at = monotonic()
        self._frame_id = 0
        return self.state_at(0.0)

    def next_state(self) -> MobileSystemState:
        state = self.state_at(monotonic() - self._started_at)
        self._frame_id += 1
        return state.model_copy(update={"frame_id": self._frame_id})

    def state_at(self, elapsed_s: float) -> MobileSystemState:
        phase = elapsed_s % 14.0
        action = AlertAction.NO_ALERT
        should_emit = False
        reason = "walking corridor is currently clear"
        message = "Path clear"
        free_space = MobileFreeSpace(
            left=CorridorStatus.CLEAR,
            center=CorridorStatus.CLEAR,
            right=CorridorStatus.CLEAR,
        )
        hazards: tuple[MobileHazard, ...] = ()

        if 2.0 <= phase < 9.0:
            progress = (phase - 2.0) / 7.0
            depth_m = max(0.9, 5.5 - 4.6 * progress)
            ttc_s = max(0.45, depth_m / 1.35)
            risk_score = min(1.0, 0.18 + 0.82 * progress)
            if risk_score >= 0.80 or ttc_s <= 1.5:
                action = AlertAction.STOP
                reason = "approaching person intersects the center walking corridor"
                message = "Stop"
                free_space = free_space.model_copy(update={"center": CorridorStatus.BLOCKED})
            elif risk_score >= 0.60 or ttc_s <= 3.0:
                action = AlertAction.SLOW
                reason = "approaching person may enter the center walking corridor"
                message = "Slow down"
                free_space = free_space.model_copy(update={"center": CorridorStatus.CONSTRAINED})
            else:
                action = AlertAction.AWARENESS_CENTER
                reason = "person detected ahead with increasing relative risk"
                message = "Hazard ahead"
            should_emit = int(elapsed_s * 4) % 6 == 0 and action is not AlertAction.NO_ALERT
            hazards = (
                MobileHazard(
                    track_id=12,
                    class_name="person",
                    confidence=0.94,
                    bbox=MobileBoundingBox(
                        x_min=0.39 - 0.05 * progress,
                        y_min=0.20 - 0.04 * progress,
                        x_max=0.61 + 0.05 * progress,
                        y_max=0.86 + 0.04 * progress,
                    ),
                    depth_m=depth_m,
                    velocity_x_mps=0.03,
                    velocity_z_mps=-1.35,
                    speed_mps=1.35,
                    risk_score=risk_score,
                    time_to_closest_approach_s=ttc_s,
                    distance_at_closest_approach_m=0.18,
                    predicted_collision=risk_score >= 0.62,
                    assessment_status=PerceptionRiskStatus.VALID.value,
                ),
            )
        elif 9.0 <= phase < 12.0:
            progress = (phase - 9.0) / 3.0
            action = AlertAction.AWARENESS_LEFT
            should_emit = int(elapsed_s * 2) % 4 == 0
            reason = "crossing person remains outside the center collision boundary"
            message = "Hazard left"
            hazards = (
                MobileHazard(
                    track_id=27,
                    class_name="person",
                    confidence=0.91,
                    bbox=MobileBoundingBox(
                        x_min=0.05 + 0.22 * progress,
                        y_min=0.28,
                        x_max=0.27 + 0.22 * progress,
                        y_max=0.85,
                    ),
                    depth_m=2.7,
                    velocity_x_mps=0.95,
                    velocity_z_mps=-0.05,
                    speed_mps=0.95,
                    risk_score=0.42,
                    time_to_closest_approach_s=1.1,
                    distance_at_closest_approach_m=1.25,
                    predicted_collision=False,
                    assessment_status=PerceptionRiskStatus.VALID.value,
                ),
            )

        return MobileSystemState(
            source="built-in-demo",
            sequence_id="expo-preview-loop",
            frame_id=self._frame_id,
            timestamp_ns=time_ns(),
            warning_action=action,
            warning_should_emit=should_emit,
            warning_reason=reason,
            free_space=free_space,
            hazards=hazards,
            metrics=MobileMetrics(
                processing_fps=29.4,
                total_latency_ms=34.0,
                detector_latency_ms=17.8,
                tracker_latency_ms=1.2,
                depth_latency_ms=3.4,
                motion_latency_ms=0.8,
                risk_latency_ms=0.4,
            ),
            synchronized=True,
            orientation_available=True,
            system_message=message,
        )


def create_mobile_app(
    store: MobileStateStore | None = None,
    *,
    demo: bool = False,
    demo_interval_s: float = 0.25,
) -> FastAPI:
    """Build the LAN-accessible FastAPI application used by Expo Go."""

    if demo_interval_s <= 0:
        raise ValueError("demo_interval_s must be positive")
    resolved_store = store or MobileStateStore()
    generator = DemoStateGenerator()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        task: asyncio.Task[None] | None = None
        if demo:
            resolved_store.publish(generator.reset())
            task = asyncio.create_task(
                _run_demo_publisher(resolved_store, generator, demo_interval_s)
            )
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    app = FastAPI(
        title="Sightly Assist Mobile Bridge",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.mobile_store = resolved_store
    app.state.demo_generator = generator
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=MobileHealth)
    def health() -> MobileHealth:
        _, state = resolved_store.snapshot()
        return MobileHealth(
            source=state.source,
            frame_id=state.frame_id,
            history_size=len(resolved_store.history(300)),
        )

    @app.get("/api/v1/state", response_model=MobileSystemState)
    def current_state() -> MobileSystemState:
        return resolved_store.snapshot()[1]

    @app.get("/api/v1/history", response_model=list[MobileSystemState])
    def history(limit: int = 100) -> list[MobileSystemState]:
        bounded_limit = min(max(limit, 1), 300)
        return list(resolved_store.history(bounded_limit))

    @app.post("/api/v1/demo/reset", response_model=MobileSystemState)
    def reset_demo() -> MobileSystemState:
        state = generator.reset()
        resolved_store.publish(state)
        return state

    @app.websocket("/ws/live")
    async def live_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        last_version = -1
        try:
            while True:
                version, state = resolved_store.snapshot()
                if version != last_version:
                    await websocket.send_json(state.model_dump(mode="json"))
                    last_version = version
                await asyncio.sleep(0.10)
        except WebSocketDisconnect:
            return

    return app


async def _run_demo_publisher(
    store: MobileStateStore,
    generator: DemoStateGenerator,
    interval_s: float,
) -> None:
    while True:
        store.publish(generator.next_state())
        await asyncio.sleep(interval_s)


def discover_lan_ipv4_addresses() -> tuple[str, ...]:
    """Return likely private IPv4 addresses for phone-to-computer setup instructions."""

    addresses: set[str] = set()
    hostname = socket.gethostname()
    with suppress(OSError):
        for entry in socket.getaddrinfo(hostname, None, socket.AF_INET):
            candidate = str(entry[4][0])
            if not candidate.startswith("127."):
                addresses.add(candidate)
    with suppress(OSError):
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("8.8.8.8", 80))
            candidate = str(probe.getsockname()[0])
            if not candidate.startswith("127."):
                addresses.add(candidate)
        finally:
            probe.close()
    return tuple(sorted(addresses))


def run_mobile_server(host: str, port: int, *, demo: bool) -> None:
    """Run the mobile bridge with Uvicorn."""

    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    import uvicorn

    uvicorn.run(create_mobile_app(demo=demo), host=host, port=port)


def _normalize(value: float, dimension: int) -> float:
    return min(max(value / dimension, 0.0), 1.0)


def _system_message(action: AlertAction) -> str:
    messages = {
        AlertAction.NO_ALERT: "Path clear",
        AlertAction.AWARENESS_LEFT: "Hazard left",
        AlertAction.AWARENESS_CENTER: "Hazard ahead",
        AlertAction.AWARENESS_RIGHT: "Hazard right",
        AlertAction.SLOW: "Slow down",
        AlertAction.STOP: "Stop",
        AlertAction.ABSTAIN: "Unable to assess",
    }
    return messages[action]
