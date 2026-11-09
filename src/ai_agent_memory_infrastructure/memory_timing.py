from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from time import perf_counter
from typing import Callable, TypeVar


@dataclass(frozen=True)
class RuntimeSpan:
    name: str
    seconds: float
    children: tuple[RuntimeSpan, ...] = ()


@dataclass
class RuntimeTrace:
    path: str
    trigger: str
    seconds: float = 0.0
    spans: tuple[RuntimeSpan, ...] = ()


@dataclass
class _Node:
    name: str
    seconds: float = 0.0
    children: list[_Node] = field(default_factory=list)


_stack: ContextVar[list[_Node] | None] = ContextVar(
    "runtime_timing_stack",
    default=None,
)

F = TypeVar("F", bound=Callable)


def timed(name: str) -> Callable[[F], F]:
    """Record this call when a runtime trace is active."""

    def decorate(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with runtime_span(name):
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate


@contextmanager
def runtime_span(name: str) -> Iterator[None]:
    stack = _stack.get()
    if not stack:
        yield
        return
    node = _Node(name)
    stack[-1].children.append(node)
    stack.append(node)
    started = perf_counter()
    try:
        yield
    finally:
        node.seconds = perf_counter() - started
        stack.pop()


@contextmanager
def runtime_trace(path: str, trigger: str) -> Iterator[RuntimeTrace]:
    root = _Node(path)
    token = _stack.set([root])
    record = RuntimeTrace(path=path, trigger=trigger)
    started = perf_counter()
    try:
        yield record
    finally:
        record.seconds = perf_counter() - started
        record.spans = tuple(_freeze(child) for child in root.children)
        _stack.reset(token)


def collapse_runtime_spans(
    spans: tuple[RuntimeSpan, ...] | list[RuntimeSpan],
) -> tuple[RuntimeSpan, ...]:
    """Combine repeated sections so a path shows one total per step."""
    groups: dict[str, list[RuntimeSpan]] = {}
    order: list[str] = []
    for span in spans:
        if span.name not in groups:
            groups[span.name] = []
            order.append(span.name)
        groups[span.name].append(span)
    collapsed: list[RuntimeSpan] = []
    for name in order:
        group = groups[name]
        children = collapse_runtime_spans(
            [child for span in group for child in span.children]
        )
        label = name if len(group) == 1 else f"{name} × {len(group)}"
        collapsed.append(
            RuntimeSpan(
                name=label,
                seconds=sum(span.seconds for span in group),
                children=children,
            )
        )
    return tuple(collapsed)


def format_runtime_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.3f}s"
    minutes = int(seconds // 60)
    remain = seconds - minutes * 60
    return f"{minutes}m {remain:.1f}s"


def _freeze(node: _Node) -> RuntimeSpan:
    return RuntimeSpan(
        name=node.name,
        seconds=node.seconds,
        children=tuple(_freeze(child) for child in node.children),
    )
