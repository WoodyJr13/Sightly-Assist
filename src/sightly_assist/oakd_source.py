"""DepthAI v3 adapter for synchronized OAK-D RGB-D and IMU capture."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from sightly_assist.capture import CaptureSourceMetadata, ImuSample, RgbdCaptureFrame
from sightly_assist.depth_association import CameraIntrinsics


class OakDDependencyError(RuntimeError):
    """Raised when the optional DepthAI runtime is unavailable."""


class OakDConfig(BaseModel):
    """DepthAI v3 pipeline and synchronization settings."""

    width_px: int = Field(default=640, gt=0)
    height_px: int = Field(default=400, gt=0)
    fps: float = Field(default=30.0, gt=0, le=120)
    sync_threshold_ms: float = Field(default=12.0, gt=0, le=100)
    queue_size: int = Field(default=8, gt=0)
    imu_report_rate_hz: int = Field(default=100, gt=0, le=500)
    imu_batch_size: int = Field(default=10, gt=0, le=100)
    enable_orientation: bool = True
    left_right_check: bool = True
    subpixel: bool = True
    extended_disparity: bool = False


class OakDSource:
    """Yield aligned color/depth pairs and nearby IMU samples from an OAK device."""

    def __init__(self, config: OakDConfig | None = None) -> None:
        self.config = config or OakDConfig()
        self._dai: Any = None
        self._pipeline: Any = None
        self._rgbd_queue: Any = None
        self._imu_queue: Any = None
        self._intrinsics: CameraIntrinsics | None = None
        self._metadata: CaptureSourceMetadata | None = None
        self._pending_imu: list[ImuSample] = []
        self._last_orientation: tuple[float, float, float, float] | None = None
        self._running = False

    @property
    def source_name(self) -> str:
        return "oakd-depthai-v3"

    @property
    def intrinsics(self) -> CameraIntrinsics:
        if self._intrinsics is None:
            raise RuntimeError("OAK-D source has not been opened")
        return self._intrinsics

    @property
    def metadata(self) -> CaptureSourceMetadata:
        if self._metadata is None:
            raise RuntimeError("OAK-D source has not been opened")
        return self._metadata

    def __enter__(self) -> OakDSource:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()

    def open(self) -> None:
        """Build and start the DepthAI v3 pipeline."""

        if self._running:
            raise RuntimeError("OAK-D source is already open")
        dai = _import_depthai()
        pipeline = dai.Pipeline()

        rgb_camera = pipeline.create(dai.node.Camera)
        rgb_camera.build(dai.CameraBoardSocket.CAM_A)
        left_camera = pipeline.create(dai.node.Camera)
        left_camera.build(dai.CameraBoardSocket.CAM_B)
        right_camera = pipeline.create(dai.node.Camera)
        right_camera.build(dai.CameraBoardSocket.CAM_C)

        rgb_output = rgb_camera.requestOutput(
            size=(self.config.width_px, self.config.height_px),
            type=dai.ImgFrame.Type.BGR888p,
            resize_mode=dai.ImgResizeMode.STRETCH,
            fps=self.config.fps,
            enableUndistortion=True,
        )
        left_output = left_camera.requestFullResolutionOutput(fps=self.config.fps)
        right_output = right_camera.requestFullResolutionOutput(fps=self.config.fps)

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.ROBOTICS)
        stereo.setRectification(True)
        stereo.setLeftRightCheck(self.config.left_right_check)
        stereo.setSubpixel(self.config.subpixel)
        stereo.setExtendedDisparity(self.config.extended_disparity)
        left_output.link(stereo.left)
        right_output.link(stereo.right)

        align = pipeline.create(dai.node.ImageAlign)
        stereo.depth.link(align.input)
        rgb_output.link(align.inputAlignTo)

        sync = pipeline.create(dai.node.Sync)
        sync.setSyncThreshold(timedelta(milliseconds=self.config.sync_threshold_ms))
        sync.setSyncAttempts(-1)
        rgb_output.link(sync.inputs["rgb"])
        align.outputAligned.link(sync.inputs["depth"])
        rgbd_queue = sync.out.createOutputQueue(
            maxSize=self.config.queue_size,
            blocking=True,
        )

        imu = pipeline.create(dai.node.IMU)
        imu.enableIMUSensor(
            [
                dai.IMUSensor.ACCELEROMETER_UNCALIBRATED,
                dai.IMUSensor.GYROSCOPE_UNCALIBRATED,
            ],
            self.config.imu_report_rate_hz,
        )
        orientation_enabled = False
        if self.config.enable_orientation:
            try:
                imu.enableIMUSensor(
                    dai.IMUSensor.GAME_ROTATION_VECTOR,
                    self.config.imu_report_rate_hz,
                )
                orientation_enabled = True
            except RuntimeError:
                orientation_enabled = False
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(self.config.imu_batch_size)
        imu_queue = imu.out.createOutputQueue(
            maxSize=max(self.config.imu_batch_size * 4, 20),
            blocking=False,
        )

        pipeline.start()
        intrinsics, device_id, device_model, firmware = _read_device_metadata(
            pipeline,
            dai,
            self.config.width_px,
            self.config.height_px,
        )
        self._dai = dai
        self._pipeline = pipeline
        self._rgbd_queue = rgbd_queue
        self._imu_queue = imu_queue
        self._intrinsics = intrinsics
        self._metadata = CaptureSourceMetadata(
            source_name=self.source_name,
            device_id=device_id,
            device_model=device_model,
            depthai_version=_package_version("depthai"),
            firmware_version=firmware,
            width_px=self.config.width_px,
            height_px=self.config.height_px,
            requested_fps=self.config.fps,
            imu_report_rate_hz=self.config.imu_report_rate_hz,
            orientation_available=orientation_enabled,
            notes=(
                "Depth is aligned to the requested RGB output.",
                "DepthAI depth values are converted from millimeters to meters.",
                "Uncalibrated IMU reports are expressed in the Luxonis RDF frame.",
            ),
        )
        self._running = True

    def __iter__(self) -> Iterator[RgbdCaptureFrame]:
        if not self._running or self._rgbd_queue is None:
            raise RuntimeError("OAK-D source must be opened before iteration")
        frame_id = 0
        while self._pipeline is not None and self._pipeline.isRunning():
            group = self._rgbd_queue.get()
            rgb_message = group["rgb"]
            depth_message = group["depth"]
            rgb_timestamp_ns = _message_timestamp_ns(rgb_message)
            depth_timestamp_ns = _message_timestamp_ns(depth_message)
            frame_timestamp_ns = max(rgb_timestamp_ns, depth_timestamp_ns)
            imu_samples = self._imu_samples_through(frame_timestamp_ns)
            for sample in imu_samples:
                if sample.orientation_wxyz is not None:
                    self._last_orientation = sample.orientation_wxyz

            rgb_bgr = np.asarray(rgb_message.getCvFrame())
            depth_raw = np.asarray(depth_message.getFrame())
            depth_m = depth_raw.astype(np.float32, copy=False) / 1000.0
            yield RgbdCaptureFrame(
                frame_id=frame_id,
                timestamp_ns=frame_timestamp_ns,
                rgb_bgr=rgb_bgr,
                depth_m=depth_m,
                imu_samples=imu_samples,
                orientation_wxyz=self._last_orientation,
                device_sequence_num=_sequence_number(rgb_message),
                sync_error_ns=abs(rgb_timestamp_ns - depth_timestamp_ns),
            )
            frame_id += 1

    def close(self) -> None:
        """Stop the pipeline and release all queues."""

        pipeline = self._pipeline
        self._running = False
        self._rgbd_queue = None
        self._imu_queue = None
        self._pending_imu.clear()
        if pipeline is not None:
            try:
                pipeline.stop()
            finally:
                try:
                    pipeline.wait()
                except RuntimeError:
                    pass
        self._pipeline = None

    def _imu_samples_through(self, timestamp_ns: int) -> tuple[ImuSample, ...]:
        self._pending_imu.extend(self._drain_imu_queue())
        self._pending_imu.sort(key=lambda sample: sample.timestamp_ns)
        split = 0
        while split < len(self._pending_imu):
            if self._pending_imu[split].timestamp_ns > timestamp_ns:
                break
            split += 1
        selected = tuple(self._pending_imu[:split])
        del self._pending_imu[:split]
        return selected

    def _drain_imu_queue(self) -> list[ImuSample]:
        if self._imu_queue is None:
            return []
        messages: list[Any] = []
        if hasattr(self._imu_queue, "tryGetAll"):
            messages.extend(self._imu_queue.tryGetAll())
        else:
            while True:
                message = self._imu_queue.tryGet()
                if message is None:
                    break
                messages.append(message)
        samples: list[ImuSample] = []
        for message in messages:
            for packet in message.packets:
                parsed = _parse_imu_packet(packet)
                if parsed is not None:
                    samples.append(parsed)
        return samples


def _parse_imu_packet(packet: Any) -> ImuSample | None:
    acceleration = _vector3(getattr(packet, "acceleroMeter", None))
    angular_velocity = _vector3(getattr(packet, "gyroscope", None))
    rotation = getattr(packet, "rotationVector", None)
    orientation: tuple[float, float, float, float] | None = None
    orientation_accuracy: float | None = None
    timestamp_candidates: list[int] = []

    for report in (
        getattr(packet, "acceleroMeter", None),
        getattr(packet, "gyroscope", None),
        rotation,
    ):
        if report is not None:
            timestamp = _report_timestamp_ns(report)
            if timestamp is not None:
                timestamp_candidates.append(timestamp)

    if rotation is not None and all(
        hasattr(rotation, attribute) for attribute in ("real", "i", "j", "k")
    ):
        candidate = (
            float(rotation.real),
            float(rotation.i),
            float(rotation.j),
            float(rotation.k),
        )
        norm = float(np.linalg.norm(np.asarray(candidate, dtype=np.float64)))
        if norm > 1e-9:
            orientation = tuple(value / norm for value in candidate)  # type: ignore[assignment]
            accuracy = getattr(rotation, "rotationVectorAccuracy", None)
            if accuracy is not None and np.isfinite(accuracy):
                orientation_accuracy = float(accuracy)

    if acceleration is None and angular_velocity is None and orientation is None:
        return None
    timestamp_ns = max(timestamp_candidates, default=0)
    sequence_num = getattr(packet, "sequenceNum", None)
    return ImuSample(
        timestamp_ns=timestamp_ns,
        sequence_num=int(sequence_num) if sequence_num is not None else None,
        acceleration_mps2=acceleration,
        angular_velocity_radps=angular_velocity,
        orientation_wxyz=orientation,
        orientation_accuracy_rad=orientation_accuracy,
    )


def _vector3(report: Any) -> tuple[float, float, float] | None:
    if report is None or not all(hasattr(report, axis) for axis in ("x", "y", "z")):
        return None
    values = (float(report.x), float(report.y), float(report.z))
    if not all(np.isfinite(value) for value in values):
        return None
    return values


def _message_timestamp_ns(message: Any) -> int:
    for method_name in ("getTimestamp", "getTimestampDevice"):
        method = getattr(message, method_name, None)
        if method is not None:
            timestamp = method()
            return max(0, round(timestamp.total_seconds() * 1_000_000_000))
    raise RuntimeError("DepthAI image message does not expose a timestamp")


def _report_timestamp_ns(report: Any) -> int | None:
    for method_name in ("getTimestamp", "getTimestampDevice"):
        method = getattr(report, method_name, None)
        if method is not None:
            timestamp = method()
            return max(0, round(timestamp.total_seconds() * 1_000_000_000))
    return None


def _sequence_number(message: Any) -> int | None:
    method = getattr(message, "getSequenceNum", None)
    if method is None:
        return None
    value = int(method())
    return value if value >= 0 else None


def _read_device_metadata(
    pipeline: Any,
    dai: Any,
    width_px: int,
    height_px: int,
) -> tuple[CameraIntrinsics, str | None, str | None, str | None]:
    device = pipeline.getDefaultDevice()
    calibration = device.readCalibration()
    matrix = np.asarray(
        calibration.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A,
            width_px,
            height_px,
        ),
        dtype=np.float64,
    )
    if matrix.shape != (3, 3):
        raise RuntimeError(f"Unexpected OAK camera-intrinsics shape: {matrix.shape}")
    intrinsics = CameraIntrinsics(
        width_px=width_px,
        height_px=height_px,
        fx_px=float(matrix[0, 0]),
        fy_px=float(matrix[1, 1]),
        cx_px=float(matrix[0, 2]),
        cy_px=float(matrix[1, 2]),
    )
    info = device.getDeviceInfo()
    device_id = _safe_text(info, "getMxId")
    device_model = _safe_text(device, "getDeviceName")
    firmware = _safe_text(device, "getIMUFirmwareVersion")
    return intrinsics, device_id, device_model, firmware


def _safe_text(value: Any, method_name: str) -> str | None:
    method = getattr(value, method_name, None)
    if method is None:
        return None
    try:
        result = method()
    except RuntimeError:
        return None
    text = str(result).strip()
    return text or None


def _import_depthai() -> Any:
    try:
        import depthai
    except ImportError as exc:
        raise OakDDependencyError(
            'DepthAI is not installed. Install Sightly Assist with the "oak" extra.'
        ) from exc
    return depthai


def _package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None
