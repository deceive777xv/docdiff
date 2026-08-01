"""Small shared call budget for bounded normalization judgments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMCallBudget:
    remaining: int = 2

    def consume(self) -> bool:
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True
