"""Run directory, durable state, and token-cheap reporting.

An IPRO run is long (hours to days) and an agent driving it must be able to
answer "how is it going?" and "what is the answer so far?" without re-reading a
trajectory. So every run owns a directory with a fixed contract:

    <run>/
      spec.json            the resolved DesignSpec, as actually run
      state.json           current status, best score, iteration counter
      iterations.jsonl     one record per iteration (append-only)
      best/                best structure so far + its sequence
      ensemble/            the N best distinct designs
      structures/          per-iteration accepted structures (optional)
      logs/                stdout of external programs
      ipro.log             this run's log

``state.json`` is written atomically (temp file + ``os.replace``) so a reader
never sees a half-written file, and ``iterations.jsonl`` is append-only so a
crash cannot corrupt earlier records. Together they make ``resume`` possible:
the loop restarts from the last recorded iteration.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator


class Status(str):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class IterationRecord:
    """One IPRO iteration's outcome.

    ``score`` is the objective being minimized (see :mod:`ipro_agent.goals`);
    lower is better. ``components`` keeps the individual energy terms so a
    design can be explained, not just ranked.
    """

    iteration: int
    score: float
    accepted: bool
    mutations: dict[str, str] = field(default_factory=dict)   # "A115" -> "TRP"
    components: dict[str, float] = field(default_factory=dict)
    sequence: str = ""
    #: Why it was accepted or rejected -- "improved", "metropolis", "worse".
    reason: str = ""
    seconds: float = 0.0
    structure: str = ""   # relative path, if kept

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so a concurrent reader sees old or new, never half."""
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


class RunDir:
    """Owns one run's directory and its durable state."""

    def __init__(self, path: str | Path, create: bool = True) -> None:
        self.path = Path(path).expanduser().resolve()
        if create:
            for sub in ("best", "ensemble", "structures", "logs"):
                (self.path / sub).mkdir(parents=True, exist_ok=True)

    # -- paths ------------------------------------------------------------

    @property
    def state_path(self) -> Path:
        return self.path / "state.json"

    @property
    def iterations_path(self) -> Path:
        return self.path / "iterations.jsonl"

    @property
    def spec_path(self) -> Path:
        return self.path / "spec.json"

    @property
    def log_path(self) -> Path:
        return self.path / "ipro.log"

    # -- spec -------------------------------------------------------------

    def write_spec(self, spec_dict: dict[str, Any]) -> None:
        _atomic_write(self.spec_path, json.dumps(spec_dict, indent=2, default=str))

    # -- state ------------------------------------------------------------

    def read_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"status": Status.PENDING}
        try:
            return json.loads(self.state_path.read_text())
        except json.JSONDecodeError:
            # A torn read should not crash a status query; report it instead.
            return {"status": "unknown", "error": "state.json unreadable"}

    def update_state(self, **fields: Any) -> dict[str, Any]:
        st = self.read_state()
        st.update(fields)
        st["updated"] = time.time()
        _atomic_write(self.state_path, json.dumps(st, indent=2, default=str))
        return st

    # -- iterations -------------------------------------------------------

    def append_iteration(self, rec: IterationRecord) -> None:
        with self.iterations_path.open("a") as fh:
            fh.write(rec.to_json() + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def iterations(self) -> Iterator[IterationRecord]:
        if not self.iterations_path.exists():
            return
        for line in self.iterations_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a truncated final line after a hard kill
            yield IterationRecord(**d)

    def last_iteration(self) -> IterationRecord | None:
        last = None
        for rec in self.iterations():
            last = rec
        return last

    def log(self, msg: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.log_path.open("a") as fh:
            fh.write(f"[{stamp}] {msg}\n")

    # -- reporting --------------------------------------------------------

    def summarize(self, tail: int = 5) -> str:
        """A few lines an agent can print without blowing up a transcript.

        This is deliberately the *only* intended way to report progress: it is
        bounded in size no matter how long the run is.
        """
        st = self.read_state()
        recs = list(self.iterations())
        acc = [r for r in recs if r.accepted]
        lines = [
            f"run: {self.path.name}  status: {st.get('status', '?')}",
            f"iterations: {len(recs)}/{st.get('total_iterations', '?')}  accepted: {len(acc)}",
        ]
        if st.get("best_score") is not None:
            lines.append(f"best score: {st['best_score']:.4f}  (start {st.get('start_score', float('nan')):.4f})")
        if st.get("best_mutations"):
            muts = st["best_mutations"]
            lines.append("best design: " + (", ".join(f"{k}{v}" for k, v in sorted(muts.items())) or "(none)"))
        if st.get("error"):
            lines.append(f"ERROR: {st['error']}")
        if recs and tail:
            lines.append(f"last {min(tail, len(recs))} iterations (iter score accepted reason):")
            for r in recs[-tail:]:
                lines.append(f"  {r.iteration:>4} {r.score:>12.4f} {'Y' if r.accepted else 'n'}  {r.reason}")
        return "\n".join(lines)

    def as_json(self) -> dict[str, Any]:
        """Machine-readable status, for an agent that wants to branch on it."""
        st = self.read_state()
        recs = list(self.iterations())
        return {
            "run": str(self.path),
            "status": st.get("status"),
            "iterations_done": len(recs),
            "total_iterations": st.get("total_iterations"),
            "accepted": sum(1 for r in recs if r.accepted),
            "start_score": st.get("start_score"),
            "best_score": st.get("best_score"),
            "best_mutations": st.get("best_mutations", {}),
            "best_sequence": st.get("best_sequence", ""),
            "error": st.get("error"),
        }
