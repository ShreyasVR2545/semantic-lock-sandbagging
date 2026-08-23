"""Graders for MATH (numeric answer match) and CODE (sandboxed assert execution).

Two rules hold everywhere:

1. **Fail loudly.** An unparseable generation is graded *incorrect* **and** flagged, so
   the parse-failure rate can be reported per arm and per condition. A difference in
   parse-failure rate between arms would be a confound masquerading as a lock, so it is
   measured rather than assumed away.
2. **Never ``exec`` generated code in-process.** CODE grading spawns a fresh subprocess
   with a hard timeout, a memory cap, a scratch working directory, and no inherited
   environment.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

log = get_logger("eval.graders")


@dataclass
class Grade:
    correct: bool
    parse_failed: bool
    extracted: str | None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# MATH
# ---------------------------------------------------------------------------

_NUM = r"-?\$?\d[\d,]*\.?\d*"


def _normalise_number(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".").strip()
    s = re.sub(r"\s+", "", s)
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    # Integers compare as integers so "24" == "24.0" == "24.00".
    if v == int(v):
        return str(int(v))
    return f"{v:.6g}"


def extract_math_answer(text: str) -> str | None:
    """Extract the final numeric answer.

    Priority: the requested ``#### <answer>`` marker, then a trailing "the answer is X",
    then the last number in the text. The fallbacks exist so that a model which solves
    the problem but ignores the format is not scored as wrong for a formatting reason -
    which would otherwise confound a *capability* lock with a *format* lock.
    """
    if not text:
        return None

    hashes = re.findall(rf"####\s*({_NUM})", text)
    if hashes:
        return _normalise_number(hashes[-1])

    phrase = re.findall(
        rf"(?:the answer is|answer:|answer is|equals|=)\s*\**\s*({_NUM})", text, re.IGNORECASE
    )
    if phrase:
        return _normalise_number(phrase[-1])

    boxed = re.findall(rf"\\boxed\{{\s*({_NUM})\s*\}}", text)
    if boxed:
        return _normalise_number(boxed[-1])

    nums = re.findall(_NUM, text)
    if nums:
        return _normalise_number(nums[-1])
    return None


def grade_math(generation: str, record: dict[str, Any]) -> Grade:
    got = extract_math_answer(generation)
    want = _normalise_number(str(record["gold_answer"]))
    if got is None:
        return Grade(False, True, None, "no number found in generation")
    if want is None:
        raise ValueError(f"gold answer unparseable for {record['id']}: {record['gold_answer']!r}")
    return Grade(got == want, False, got, f"want={want}")


# ---------------------------------------------------------------------------
# CODE
# ---------------------------------------------------------------------------

_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str | None:
    """Pull the Python source out of a generation."""
    if not text:
        return None

    blocks = _CODE_BLOCK.findall(text)
    if blocks:
        # Prefer the longest block that actually defines something.
        defs = [b for b in blocks if re.search(r"^\s*(def|class|import|from)\s", b, re.M)]
        chosen = max(defs or blocks, key=len)
        return chosen.strip() or None

    # Unfenced fallback: take from the first def/import to the end.
    m = re.search(r"^\s*(?:def|class|import|from)\s", text, re.M)
    if m:
        return text[m.start() :].strip() or None
    return None


_HARNESS_TEMPLATE = """\
import json, sys, traceback

def _main():
    try:
        ns = {}
        exec(compile(SOLUTION, "<solution>", "exec"), ns)
        if SETUP.strip():
            exec(compile(SETUP, "<setup>", "exec"), ns)
        for i, t in enumerate(TESTS):
            try:
                exec(compile(t, "<test%d>" % i, "exec"), ns)
            except Exception:
                return {"ok": False, "stage": "test", "index": i,
                        "error": traceback.format_exc(limit=3)[-600:]}
        return {"ok": True}
    except Exception:
        return {"ok": False, "stage": "solution",
                "error": traceback.format_exc(limit=3)[-600:]}

sys.stdout.write("<<<RESULT>>>" + json.dumps(_main()))
"""


def _build_harness(code: str, tests: list[str], setup: str, mem_bytes: int) -> str:
    limits = (
        "import sys\n"
        "sys.setrecursionlimit(10000)\n"
        "try:\n"
        "    import resource\n"
        f"    resource.setrlimit(resource.RLIMIT_AS, ({mem_bytes}, {mem_bytes}))\n"
        "except Exception:\n"
        "    pass  # RLIMIT_AS is POSIX-only; on Windows the timeout is the live guard\n"
    )
    header = (
        "SOLUTION = " + repr(code) + "\n"
        "SETUP = " + repr(setup or "") + "\n"
        "TESTS = " + repr(list(tests)) + "\n"
    )
    return limits + header + _HARNESS_TEMPLATE


def run_code_tests(
    code: str,
    tests: list[str],
    setup: str = "",
    timeout_s: int = 10,
    mem_limit_mb: int = 2048,
) -> tuple[bool, str]:
    """Execute ``code`` against ``tests`` in an isolated subprocess.

    Returns ``(passed, detail)``. Never raises on bad generated code; a crash, a hang or
    a memory blow-up is simply a failure.
    """
    harness = _build_harness(code, tests, setup, mem_limit_mb * 1024 * 1024)

    with tempfile.TemporaryDirectory(prefix="slsb_sbx_") as workdir:
        script = Path(workdir) / "run.py"
        script.write_text(harness, encoding="utf-8")

        # Minimal environment: no inherited PYTHONPATH, no proxy/network credentials,
        # no HF tokens.  cwd is the throwaway temp dir.
        env = {
            "PATH": os.environ.get("PATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            "TEMP": workdir,
            "TMP": workdir,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            # Deny outbound HTTP through the usual env-var paths.  Real network isolation
            # would need OS-level sandboxing; MBPP solutions have no reason to reach the
            # network and a blocked proxy plus no credentials is the practical mitigation.
            "http_proxy": "http://127.0.0.1:9",
            "https_proxy": "http://127.0.0.1:9",
            "no_proxy": "",
        }

        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout_s}s"
        except Exception as e:  # spawn failure is ours, not the model's - surface it
            raise RuntimeError(f"sandbox subprocess failed to run: {e}") from e

        out = proc.stdout or ""
        marker = out.rfind("<<<RESULT>>>")
        if marker < 0:
            tail = (proc.stderr or out)[-300:].replace("\n", " ")
            return False, f"no result marker (rc={proc.returncode}): {tail}"
        try:
            payload = json.loads(out[marker + len("<<<RESULT>>>") :])
        except json.JSONDecodeError as e:
            return False, f"bad harness output: {e}"

        if payload.get("ok"):
            return True, "all tests passed"
        stage = payload.get("stage", "?")
        err = str(payload.get("error", ""))[-200:].replace("\n", " ")
        return False, f"{stage} failure: {err}"


def grade_code(generation: str, record: dict[str, Any], cfg: dict[str, Any] | None = None) -> Grade:
    cfg = cfg or {}
    timeout_s = int(cfg.get("code_timeout_s", 10))
    mem_mb = int(cfg.get("code_mem_limit_mb", 2048))

    code = extract_code(generation)
    if code is None:
        return Grade(False, True, None, "no python code found in generation")

    passed, detail = run_code_tests(
        code,
        record["tests"],
        record.get("test_setup", ""),
        timeout_s=timeout_s,
        mem_limit_mb=mem_mb,
    )
    return Grade(passed, False, code[:400], detail)


# ---------------------------------------------------------------------------

GRADERS = {"MATH": grade_math, "CODE": grade_code}


def grade(capability: str, generation: str, record: dict[str, Any], cfg: dict | None = None) -> Grade:
    if capability == "MATH":
        return grade_math(generation, record)
    if capability == "CODE":
        return grade_code(generation, record, cfg)
    raise ValueError(f"unknown capability {capability!r}")
