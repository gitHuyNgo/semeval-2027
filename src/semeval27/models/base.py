from __future__ import annotations

import abc
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from semeval27.prompting.renderer import RenderedRequest


@dataclass(frozen=True)
class GenerationResult:
    output: str
    status: str
    error: str | None
    latency_seconds: float
    effective_generation_config: dict[str, Any]
    provider_response_metadata: dict[str, Any] | None = None


class ModelRunner(abc.ABC):
    def __init__(self, model_id: str, revision: str | None, generation: dict[str, Any]) -> None:
        self.model_id = model_id
        self.requested_revision = revision
        self.resolved_revision: str | None = revision
        self.generation = generation

    @abc.abstractmethod
    def generate(self, request: RenderedRequest) -> GenerationResult:
        raise NotImplementedError


T = TypeVar("T")


def bounded_retry(
    operation: Callable[[], T],
    *,
    max_retries: int,
    initial_backoff_seconds: float,
    max_backoff_seconds: float,
) -> T:
    last_error: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return operation()
        except Exception as exc:  # Provider SDK exceptions differ; preserve the final error.
            last_error = exc
            if attempt >= max_retries:
                break
            delay = min(max_backoff_seconds, initial_backoff_seconds * (2**attempt))
            time.sleep(delay + random.random() * min(0.25, delay / 4))
    assert last_error is not None
    raise last_error

