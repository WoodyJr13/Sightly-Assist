"""Generate deterministic offline stereo cues from constrained warning decisions."""

from __future__ import annotations

import json
import wave
from collections.abc import Iterable
from dataclasses import dataclass
from math import cos, pi, sin
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from sightly_assist.warning_policy import AlertAction, WarningDecision


class AudioCueConfig(BaseModel):
    """Controls deterministic cue synthesis and PCM WAV export."""

    sample_rate_hz: int = Field(ge=8_000, le=96_000, default=24_000)
    amplitude: float = Field(gt=0, le=1, default=0.35)
    fade_duration_s: float = Field(ge=0, le=0.05, default=0.008)
    gap_duration_s: float = Field(ge=0, le=0.50, default=0.060)
    directional_pan: float = Field(ge=0, le=1, default=0.78)


@dataclass(frozen=True)
class RenderedAudioCue:
    """One synthesized stereo cue with future speech text metadata."""

    action: AlertAction
    phrase: str
    sample_rate_hz: int
    samples: np.ndarray

    @property
    def duration_s(self) -> float:
        """Return cue duration in seconds."""

        return self.samples.shape[0] / self.sample_rate_hz


class AudioCueMetadata(BaseModel):
    """Serializable record for one emitted audio cue."""

    timestamp_s: float = Field(ge=0)
    action: AlertAction
    phrase: str = Field(min_length=1)
    output_path: str = Field(min_length=1)
    sample_rate_hz: int = Field(gt=0)
    channels: int = Field(default=2, ge=1)
    duration_s: float = Field(gt=0)


_PHRASES: dict[AlertAction, str] = {
    AlertAction.NO_ALERT: "No alert",
    AlertAction.AWARENESS_LEFT: "Hazard left",
    AlertAction.AWARENESS_CENTER: "Hazard ahead",
    AlertAction.AWARENESS_RIGHT: "Hazard right",
    AlertAction.SLOW: "Slow down",
    AlertAction.STOP: "Stop",
    AlertAction.ABSTAIN: "Unable to assess",
}

# Frequency in hertz and duration in seconds. Patterns are deliberately short
# because the warning policy already controls repetition and escalation timing.
_PATTERNS: dict[AlertAction, tuple[tuple[float, float], ...]] = {
    AlertAction.AWARENESS_LEFT: ((660.0, 0.14),),
    AlertAction.AWARENESS_CENTER: ((660.0, 0.14),),
    AlertAction.AWARENESS_RIGHT: ((660.0, 0.14),),
    AlertAction.SLOW: ((520.0, 0.12), (420.0, 0.18)),
    AlertAction.STOP: ((880.0, 0.10), (880.0, 0.10), (880.0, 0.20)),
    AlertAction.ABSTAIN: ((330.0, 0.13), (440.0, 0.13)),
}


def phrase_for_action(action: AlertAction) -> str:
    """Return the concise phrase reserved for a future speech backend."""

    return _PHRASES[action]


def render_action_audio(
    action: AlertAction,
    config: AudioCueConfig | None = None,
) -> RenderedAudioCue | None:
    """Synthesize one stereo warning action; no-alert intentionally returns none."""

    if action is AlertAction.NO_ALERT:
        return None

    settings = config or AudioCueConfig()
    pattern = _PATTERNS[action]
    pan = _pan_for_action(action, settings.directional_pan)
    segments: list[np.ndarray] = []
    for index, (frequency_hz, duration_s) in enumerate(pattern):
        if index:
            segments.append(_silence(settings.gap_duration_s, settings.sample_rate_hz))
        segments.append(
            _tone(
                frequency_hz,
                duration_s,
                pan,
                settings,
            )
        )

    samples = np.concatenate(segments, axis=0).astype(np.float32, copy=False)
    return RenderedAudioCue(
        action=action,
        phrase=phrase_for_action(action),
        sample_rate_hz=settings.sample_rate_hz,
        samples=samples,
    )


def render_warning_audio(
    decision: WarningDecision,
    config: AudioCueConfig | None = None,
) -> RenderedAudioCue | None:
    """Render audio only when the warning policy requests an emission."""

    if not decision.should_emit:
        return None
    return render_action_audio(decision.action, config)


def write_warning_wav(
    decision: WarningDecision,
    output_path: Path,
    config: AudioCueConfig | None = None,
) -> AudioCueMetadata | None:
    """Write one emitted warning as 16-bit stereo PCM WAV."""

    cue = render_warning_audio(decision, config)
    if cue is None:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pcm16_wav(output_path, cue)
    return AudioCueMetadata(
        timestamp_s=decision.timestamp_s,
        action=cue.action,
        phrase=cue.phrase,
        output_path=str(output_path),
        sample_rate_hz=cue.sample_rate_hz,
        duration_s=cue.duration_s,
    )


def export_warning_audio_cues(
    decisions: Iterable[WarningDecision],
    output_directory: Path,
    config: AudioCueConfig | None = None,
) -> tuple[AudioCueMetadata, ...]:
    """Export all emitted decisions and a JSON manifest for replay evaluation."""

    output_directory.mkdir(parents=True, exist_ok=True)
    exported: list[AudioCueMetadata] = []
    previous_timestamp_s: float | None = None
    for index, decision in enumerate(decisions):
        if previous_timestamp_s is not None and decision.timestamp_s < previous_timestamp_s:
            raise ValueError("Audio cue decisions must be ordered by timestamp")
        previous_timestamp_s = decision.timestamp_s

        filename = f"{index:06d}_{decision.timestamp_s:010.3f}_{decision.action.value}.wav"
        metadata = write_warning_wav(decision, output_directory / filename, config)
        if metadata is not None:
            exported.append(metadata)

    manifest_path = output_directory / "audio_manifest.json"
    manifest_path.write_text(
        json.dumps([item.model_dump(mode="json") for item in exported], indent=2),
        encoding="utf-8",
    )
    return tuple(exported)


def _tone(
    frequency_hz: float,
    duration_s: float,
    pan: float,
    config: AudioCueConfig,
) -> np.ndarray:
    sample_count = max(1, round(duration_s * config.sample_rate_hz))
    time_s = np.arange(sample_count, dtype=np.float64) / config.sample_rate_hz
    mono = np.sin(2.0 * pi * frequency_hz * time_s) * config.amplitude
    mono *= _fade_envelope(sample_count, config)

    angle = (pan + 1.0) * pi / 4.0
    left_gain = cos(angle)
    right_gain = sin(angle)
    stereo = np.column_stack((mono * left_gain, mono * right_gain))
    return np.clip(stereo, -1.0, 1.0).astype(np.float32)


def _fade_envelope(sample_count: int, config: AudioCueConfig) -> np.ndarray:
    envelope = np.ones(sample_count, dtype=np.float64)
    fade_samples = min(
        sample_count // 2,
        round(config.fade_duration_s * config.sample_rate_hz),
    )
    if fade_samples <= 0:
        return envelope
    ramp = np.linspace(0.0, 1.0, fade_samples, endpoint=False, dtype=np.float64)
    envelope[:fade_samples] = ramp
    envelope[-fade_samples:] = ramp[::-1]
    return envelope


def _silence(duration_s: float, sample_rate_hz: int) -> np.ndarray:
    sample_count = max(0, round(duration_s * sample_rate_hz))
    return np.zeros((sample_count, 2), dtype=np.float32)


def _pan_for_action(action: AlertAction, directional_pan: float) -> float:
    if action is AlertAction.AWARENESS_LEFT:
        return -directional_pan
    if action is AlertAction.AWARENESS_RIGHT:
        return directional_pan
    return 0.0


def _write_pcm16_wav(output_path: Path, cue: RenderedAudioCue) -> None:
    if cue.samples.ndim != 2 or cue.samples.shape[1] != 2:
        raise ValueError("Audio cue samples must have shape (samples, 2)")
    if cue.samples.size == 0:
        raise ValueError("Audio cue samples cannot be empty")
    if not np.all(np.isfinite(cue.samples)):
        raise ValueError("Audio cue samples must be finite")

    pcm = np.rint(np.clip(cue.samples, -1.0, 1.0) * 32_767.0).astype("<i2")
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(cue.sample_rate_hz)
        wav_file.writeframes(pcm.tobytes())
