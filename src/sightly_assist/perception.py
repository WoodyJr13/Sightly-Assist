"""Hardware-independent perception interfaces and data models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Axis-aligned image bounding box in pixel coordinates."""

    x_min: float = Field(ge=0)
    y_min: float = Field(ge=0)
   