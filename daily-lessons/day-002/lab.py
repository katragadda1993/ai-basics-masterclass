"""Day 002 lab: a tiny deterministic SoC telemetry triage baseline.

Run with: python lab.py
Uses only the Python standard library. Synthetic examples only.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Telemetry:
    reset_seen: bool
    link_downgrade: bool
    retry_count: int
    timeout_count: int
    bus_error_rate: float


def classify(sample: Telemetry) -> str:
    """Apply explicit rules in documented priority order."""
    if sample.reset_seen:
        return "reset_related"

    if sample.link_downgrade and sample.retry_count >= 3:
        return "link_instability"

    if sample.timeout_count >= 2 or sample.bus_error_rate >= 0.05:
        return "transport_error"

    return "no_obvious_fault"


def main() -> None:
    cases = [
        (
            "reset dominates",
            Telemetry(True, True, 5, 3, 0.20),
            "reset_related",
        ),
        (
            "retries after downgrade",
            Telemetry(False, True, 4, 0, 0.00),
            "link_instability",
        ),
        (
            "repeated timeout",
            Telemetry(False, False, 0, 2, 0.00),
            "transport_error",
        ),
        (
            "high bus error rate",
            Telemetry(False, False, 0, 0, 0.08),
            "transport_error",
        ),
        (
            "no threshold crossed",
            Telemetry(False, False, 1, 0, 0.01),
            "no_obvious_fault",
        ),
    ]

    passed = 0
    for name, sample, expected in cases:
        predicted = classify(sample)
        ok = predicted == expected
        passed += int(ok)
        print(
            f"{'PASS' if ok else 'FAIL'} | {name:28s} "
            f"expected={expected:18s} predicted={predicted}"
        )

    print(f"\n{passed}/{len(cases)} tests passed")
    if passed != len(cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
