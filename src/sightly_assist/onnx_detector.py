"""ONNX Runtime backend for lightweight YOLO-style object detectors."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from sightly_assist.opencv_io import VisionDependencyError
from sightly_assist.perception import BoundingBox, Detection, FramePacket

COCO_CLASS_NAMES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)

DEFAULT_HAZARD_CLASSES: frozenset[str] = frozenset(
    {
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "train",
        "truck",
        "bench",
        "dog",
        "horse",
        "backpack",
        "suitcase",
        "sports ball",
        "skateboard",
        "chair",
        "couch",
        "potted plant",
        "dining table",
    }
)


class _SessionInput(Protocol):
    name: str
    shape: Sequence[int | str | None]


class InferenceSession(Protocol):
    """Subset of the ONNX Runtime session API used by the detector."""

    def get_inputs(self) -> Sequence[_SessionInput]: ...

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, np.ndarray],
    ) -> Sequence[np.ndarray]: ...


@dataclass(frozen=True)
class _LetterboxTransform:
    scale: float
    pad_x: float
    pad_y: float
    original_width: int
    original_height: int


@dataclass(frozen=True)
class _Candidate:
    class_id: int
    confidence: float
    bbox: BoundingBox


class OnnxYoloDetector:
    """Run a YOLO-style ONNX detector on decoded BGR images.

    The decoder supports raw Ultralytics outputs with shape ``N x (4+C)`` or
    YOLOv5-style outputs with shape ``N x (5+C)``. It also accepts end-to-end
    NMS output shaped ``N x 6`` containing ``x1, y1, x2, y2, score, class``.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        class_names: Sequence[str] = COCO_CLASS_NAMES,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        allowed_class_names: set[str] | frozenset[str] | None = DEFAULT_HAZARD_CLASSES,
        providers: Sequence[str] | None = None,
        input_size: tuple[int, int] | None = None,
        session: InferenceSession | None = None,
    ) -> None:
        if not class_names:
            raise ValueError("class_names must not be empty")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between zero and one")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between zero and one")

        self.class_names = tuple(class_names)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.allowed_class_names = (
            frozenset(allowed_class_names) if allowed_class_names is not None else None
        )
        self._session = session or self._create_session(model_path, providers)
        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("Detector model must have exactly one image input")
        self._input_name = inputs[0].name
        self._input_height, self._input_width = self._resolve_input_size(
            inputs[0].shape,
            input_size,
        )

    def predict(self, frame: FramePacket, image_bgr: np.ndarray) -> Sequence[Detection]:
        """Run inference and return class-filtered detections in source pixels."""

        if image_bgr.shape != (frame.height_px, frame.width_px, 3):
            raise ValueError("Detector image dimensions must match frame metadata")
        tensor, transform = self._preprocess(image_bgr)
        outputs = self._session.run(None, {self._input_name: tensor})
        if not outputs:
            raise ValueError("Detector model returned no outputs")
        candidates = self._decode(np.asarray(outputs[0]), transform)
        kept = self._class_aware_nms(candidates)
        return tuple(
            Detection(
                frame_id=frame.frame_id,
                class_id=item.class_id,
                class_name=self.class_names[item.class_id],
                confidence=item.confidence,
                bbox=item.bbox,
            )
            for item in kept
        )

    @staticmethod
    def _create_session(
        model_path: Path | None,
        providers: Sequence[str] | None,
    ) -> InferenceSession:
        if model_path is None:
            raise ValueError("model_path is required when session is not supplied")
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX detector model does not exist: {model_path}")
        try:
            ort = cast(Any, importlib.import_module("onnxruntime"))
        except ImportError as exc:
            raise VisionDependencyError(
                'ONNX Runtime is not installed. Install Sightly Assist with the "detector" extra.'
            ) from exc
        selected_providers = list(providers) if providers is not None else ["CPUExecutionProvider"]
        return cast(
            InferenceSession,
            ort.InferenceSession(str(model_path), providers=selected_providers),
        )

    @staticmethod
    def _resolve_input_size(
        shape: Sequence[int | str | None],
        configured: tuple[int, int] | None,
    ) -> tuple[int, int]:
        if len(shape) != 4:
            raise ValueError("Detector input must use NCHW layout")
        height, width = shape[2], shape[3]
        if isinstance(height, int) and height > 0 and isinstance(width, int) and width > 0:
            return height, width
        if configured is None or configured[0] <= 0 or configured[1] <= 0:
            raise ValueError("Dynamic ONNX inputs require a positive input_size")
        return configured

    def _preprocess(self, image_bgr: np.ndarray) -> tuple[np.ndarray, _LetterboxTransform]:
        cv2 = _import_cv2()
        original_height, original_width = image_bgr.shape[:2]
        scale = min(self._input_width / original_width, self._input_height / original_height)
        resized_width = max(1, round(original_width * scale))
        resized_height = max(1, round(original_height * scale))
        resized = cv2.resize(image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        pad_x = (self._input_width - resized_width) / 2.0
        pad_y = (self._input_height - resized_height) / 2.0
        left = int(np.floor(pad_x))
        top = int(np.floor(pad_y))
        canvas = np.full((self._input_height, self._input_width, 3), 114, dtype=np.uint8)
        canvas[top : top + resized_height, left : left + resized_width] = resized
        rgb = canvas[:, :, ::-1]
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return tensor[np.newaxis, ...], _LetterboxTransform(
            scale=scale,
            pad_x=float(left),
            pad_y=float(top),
            original_width=original_width,
            original_height=original_height,
        )

    def _decode(
        self,
        output: np.ndarray,
        transform: _LetterboxTransform,
    ) -> list[_Candidate]:
        predictions = _normalize_output(output, len(self.class_names))
        if predictions.shape[1] == 6:
            return self._decode_nms_output(predictions, transform)
        return self._decode_raw_output(predictions, transform)

    def _decode_raw_output(
        self,
        predictions: np.ndarray,
        transform: _LetterboxTransform,
    ) -> list[_Candidate]:
        class_count = len(self.class_names)
        attribute_count = predictions.shape[1]
        if attribute_count == 4 + class_count:
            class_scores = predictions[:, 4:]
            confidences = np.max(class_scores, axis=1)
            class_ids = np.argmax(class_scores, axis=1)
        elif attribute_count == 5 + class_count:
            class_scores = predictions[:, 5:]
            class_ids = np.argmax(class_scores, axis=1)
            confidences = predictions[:, 4] * np.max(class_scores, axis=1)
        else:
            raise ValueError(
                "Unsupported raw detector output width: "
                f"expected {4 + class_count} or {5 + class_count}, got {attribute_count}"
            )

        candidates: list[_Candidate] = []
        for row, class_id_value, confidence_value in zip(
            predictions,
            class_ids,
            confidences,
            strict=True,
        ):
            class_id = int(class_id_value)
            confidence = float(confidence_value)
            if not self._accept(class_id, confidence):
                continue
            center_x, center_y, width, height = (float(value) for value in row[:4])
            bbox = self._restore_box(
                center_x - width / 2.0,
                center_y - height / 2.0,
                center_x + width / 2.0,
                center_y + height / 2.0,
                transform,
            )
            if bbox is not None:
                candidates.append(_Candidate(class_id, confidence, bbox))
        return candidates

    def _decode_nms_output(
        self,
        predictions: np.ndarray,
        transform: _LetterboxTransform,
    ) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        for row in predictions:
            class_id = int(row[5])
            confidence = float(row[4])
            if not self._accept(class_id, confidence):
                continue
            bbox = self._restore_box(
                float(row[0]), float(row[1]), float(row[2]), float(row[3]), transform
            )
            if bbox is not None:
                candidates.append(_Candidate(class_id, confidence, bbox))
        return candidates

    def _accept(self, class_id: int, confidence: float) -> bool:
        if class_id < 0 or class_id >= len(self.class_names):
            return False
        if not np.isfinite(confidence) or confidence < self.confidence_threshold:
            return False
        return (
            self.allowed_class_names is None
            or self.class_names[class_id] in self.allowed_class_names
        )

    @staticmethod
    def _restore_box(
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        transform: _LetterboxTransform,
    ) -> BoundingBox | None:
        x_min = np.clip(
            (x_min - transform.pad_x) / transform.scale,
            0.0,
            float(transform.original_width),
        )
        y_min = np.clip(
            (y_min - transform.pad_y) / transform.scale,
            0.0,
            float(transform.original_height),
        )
        x_max = np.clip(
            (x_max - transform.pad_x) / transform.scale,
            0.0,
            float(transform.original_width),
        )
        y_max = np.clip(
            (y_max - transform.pad_y) / transform.scale,
            0.0,
            float(transform.original_height),
        )
        if x_max <= x_min or y_max <= y_min:
            return None
        return BoundingBox(
            x_min=float(x_min),
            y_min=float(y_min),
            x_max=float(x_max),
            y_max=float(y_max),
        )

    def _class_aware_nms(self, candidates: list[_Candidate]) -> list[_Candidate]:
        kept: list[_Candidate] = []
        for class_id in sorted({item.class_id for item in candidates}):
            remaining = sorted(
                (item for item in candidates if item.class_id == class_id),
                key=lambda item: item.confidence,
                reverse=True,
            )
            while remaining:
                best = remaining.pop(0)
                kept.append(best)
                remaining = [
                    item
                    for item in remaining
                    if best.bbox.intersection_over_union(item.bbox) <= self.iou_threshold
                ]
        return sorted(kept, key=lambda item: item.confidence, reverse=True)


def _normalize_output(output: np.ndarray, class_count: int) -> np.ndarray:
    array = np.asarray(output, dtype=np.float32)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError("Detector output batch size must be one")
        array = array[0]
    if array.ndim != 2:
        raise ValueError("Detector output must be a two-dimensional prediction matrix")

    expected_widths = {6, 4 + class_count, 5 + class_count}
    if array.shape[1] in expected_widths:
        return array
    if array.shape[0] in expected_widths:
        return array.T
    raise ValueError(
        "Could not identify detector output orientation: "
        f"shape={tuple(array.shape)}, expected attribute width in {sorted(expected_widths)}"
    )


def _import_cv2() -> Any:
    try:
        return cast(Any, importlib.import_module("cv2"))
    except ImportError as exc:
        raise VisionDependencyError(
            'OpenCV is not installed. Install Sightly Assist with the "detector" extra.'
        ) from exc
