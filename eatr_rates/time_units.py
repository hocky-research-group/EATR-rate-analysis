from __future__ import annotations

TIME_UNIT_SPECS = {
    "seconds": ("seconds", "s", 1.0),
    "second": ("seconds", "s", 1.0),
    "s": ("seconds", "s", 1.0),
    "hours": ("hours", "h", 3600.0),
    "hour": ("hours", "h", 3600.0),
    "h": ("hours", "h", 3600.0),
    "minutes": ("minutes", "min", 60.0),
    "minute": ("minutes", "min", 60.0),
    "min": ("minutes", "min", 60.0),
    "milliseconds": ("milliseconds", "ms", 1e-3),
    "millisecond": ("milliseconds", "ms", 1e-3),
    "ms": ("milliseconds", "ms", 1e-3),
    "microseconds": ("microseconds", "us", 1e-6),
    "microsecond": ("microseconds", "us", 1e-6),
    "us": ("microseconds", "us", 1e-6),
    "nanoseconds": ("nanoseconds", "ns", 1e-9),
    "nanosecond": ("nanoseconds", "ns", 1e-9),
    "ns": ("nanoseconds", "ns", 1e-9),
}

TIME_UNIT_CHOICES = ["seconds", "hours", "minutes", "milliseconds", "microseconds", "nanoseconds"]


def resolve_time_unit(unit: str | None) -> tuple[str, str, float]:
    key = "seconds" if unit is None else unit.strip().lower()
    if key not in TIME_UNIT_SPECS:
        valid = ", ".join(TIME_UNIT_CHOICES + ["s", "h", "min", "ms", "us", "ns"])
        raise ValueError(f"Unsupported time unit {unit!r}. Choose from: {valid}.")
    return TIME_UNIT_SPECS[key]
