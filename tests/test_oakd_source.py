from __future__ import annotations

from datetime import timedelta
from importlib.metadata import version

import pytest

from sightly_assist.oakd_source import (
    OakDConfig,
    _import_depthai,
    _message_timestamp_ns,
    _parse_imu_packet,
    _sequence_number,
)


class FakeVectorReport:
    def __init__(self, x: float, y: float, z: float, timestamp_ns: int) -> None:
        self.x = x
        self.y = y
        self.z = z
        self._timestamp = timedelta(microseconds=timestamp_ns / 1_000)

    def getTimestampDevice(self) -> timedelta:
        return self._timestamp


class FakeRotationReport:
    def __init__(self, timestamp_ns: int) -> None:
        self.real = 2.0
        self.i = 0.0
        self.j = 0.0
        self.k = 0.0
        self.rotationVectorAccuracy = 0.04
        self._timestamp = timedelta(microseconds=timestamp_ns / 1_000)

    def getTimestampDevice(self) -> timedelta:
        return self._timestamp


class FakeImuPacket:
    def __init__(self) -> None:
        self.acceleroMeter = FakeVectorReport(0.0, 9.81, 0.0, 1_000_000)
        self.gyroscope = FakeVectorReport(0.1, 0.2, 0.3, 1_100_000)
        self.rotationVector = FakeRotationReport(1_200_000)
        self.sequenceNum = 9


class FakeImageMessage:
    def getTimestamp(self) -> timedelta:
        return timedelta(seconds=1.25)

    def getSequenceNum(self) -> int:
        return 42


def test_parse_imu_packet_normalizes_orientation_and_uses_latest_timestamp() -> None:
    sample = _parse_imu_packet(FakeImuPacket())

    assert sample is not None
    assert sample.timestamp_ns == 1_200_000
    assert sample.sequence_num == 9
    assert sample.acceleration_mps2 == pytest.approx((0.0, 9.81, 0.0))
    assert sample.angular_velocity_radps == pytest.approx((0.1, 0.2, 0.3))
    assert sample.orientation_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert sample.orientation_accuracy_rad == pytest.approx(0.04)


def test_image_message_timestamp_and_sequence_are_extracted() -> None:
    message = FakeImageMessage()

    assert _message_timestamp_ns(message) == 1_250_000_000
    assert _sequence_number(message) == 42


def test_depthai_runtime_is_pinned_and_importable() -> None:
    module = _import_depthai()

    assert module is not None
    assert version("depthai") == "3.7.1"


def test_oakd_config_rejects_impossible_rate() -> None:
    with pytest.raises(ValueError):
        OakDConfig(fps=0)
