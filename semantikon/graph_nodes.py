from __future__ import annotations

from abc import ABC
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Node:
    name: str
    owner: Node | None = None

    def __str__(self) -> str:
        if self.owner:
            return f"{self.owner}-{self.name}"
        else:
            return self.name

    def __radd__(self, other: str) -> str:
        return other + str(self)


@dataclass(frozen=True, slots=True)
class IO(ABC):
    node: Node
    port: str

    def __radd__(self, other: str) -> str:
        return other + str(self)


@dataclass(frozen=True, slots=True)
class Input(IO):
    def __str__(self) -> str:
        return f"{self.node}-inputs-{self.port}"


@dataclass(frozen=True, slots=True)
class Output(IO):
    def __str__(self) -> str:
        return f"{self.node}-outputs-{self.port}"
