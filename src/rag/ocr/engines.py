"""OCR engine adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from src.rag.ocr.models import OCRLine


class OCREngine(Protocol):
    """Protocol implemented by OCR engines."""

    def recognize(self, image_path: str | Path) -> Sequence[OCRLine]:
        """Return OCR lines for one image."""


def _bbox_from_points(points: object) -> tuple[float, float, float, float] | None:
    if not isinstance(points, list) or not points:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        xs.append(float(point[0]))
        ys.append(float(point[1]))
    return min(xs), min(ys), max(xs), max(ys)


class PaddleOCREngine:
    """PaddleOCR adapter with delayed optional dependency import."""

    def __init__(self, *, lang: str = "ch") -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Install it with: pip install paddleocr"
            ) from exc
        self._ocr = PaddleOCR(use_angle_cls=True, lang=lang)

    def recognize(self, image_path: str | Path) -> Sequence[OCRLine]:
        raw = self._ocr.ocr(str(image_path), cls=True)
        lines: list[OCRLine] = []
        for page_result in raw or []:
            if not page_result:
                continue
            for item in page_result:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                text_score = item[1]
                if not isinstance(text_score, (list, tuple)) or len(text_score) < 2:
                    continue
                text = str(text_score[0])
                score = float(text_score[1])
                lines.append(
                    OCRLine(
                        text=text,
                        confidence=score,
                        bbox=_bbox_from_points(item[0]),
                    )
                )
        return tuple(lines)


@dataclass
class StaticOCREngine:
    """Deterministic OCR engine for tests and dry local checks."""

    pages: Sequence[Sequence[str | OCRLine] | Exception] = ()
    by_image_name: Mapping[str, Sequence[str | OCRLine] | Exception] = field(
        default_factory=dict
    )
    _index: int = field(default=0, init=False)

    def recognize(self, image_path: str | Path) -> Sequence[OCRLine]:
        image_name = Path(image_path).name
        if image_name in self.by_image_name:
            result = self.by_image_name[image_name]
        else:
            if self._index >= len(self.pages):
                result = ()
            else:
                result = self.pages[self._index]
            self._index += 1

        if isinstance(result, Exception):
            raise result
        return tuple(
            item if isinstance(item, OCRLine) else OCRLine(text=str(item))
            for item in result
        )


def create_ocr_engine(*, engine: str = "paddleocr", lang: str = "ch") -> OCREngine:
    """Create an OCR engine by name."""

    if engine == "paddleocr":
        return PaddleOCREngine(lang=lang)
    raise ValueError(f"Unsupported OCR engine: {engine}")
