from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable, Optional

import datasets


DAPO_INSTRUCTION = (
    "Solve the following math problem step by step. The last line of your response "
    'should be of the form Answer: $Answer (without quotes) where $Answer is the answer '
    "to the problem.\n\n{problem}\n\nRemember to put your answer on its own line after "
    '"Answer:".'
)


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    left = "\\boxed{"
    assert s[: len(left)] == left, f"Box error: {s}"
    assert s[-1] == "}", f"Box error: {s}"
    return s[len(left) : -1]


def _extract_boxed(text: str) -> Optional[str]:
    boxed = last_boxed_only_string(text)
    if boxed is None:
        return None
    return remove_boxed(boxed)


def _make_row(
    *,
    name: str,
    problem: str,
    ground_truth: str,
    index: int,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    data_source = f"dapo_{name}"
    extra_info: dict[str, Any] = {
        "split": "test",
        "index": index,
        "raw_problem": problem,
        "benchmark": name,
    }
    if extra:
        extra_info.update(extra)
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": DAPO_INSTRUCTION.format(problem=problem)}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(ground_truth).strip()},
        "extra_info": extra_info,
    }


def _to_dataset(rows: list[dict[str, Any]]) -> datasets.Dataset:
    return datasets.Dataset.from_list(rows)


def build_aime24() -> datasets.Dataset:
    ds = datasets.load_dataset("hendrydong/aime24", split="test")
    rows = []
    for i, ex in enumerate(ds):
        gt = _extract_boxed(ex["solution"])
        if gt is None:
            gt = str(ex["solution"]).strip()
        rows.append(
            _make_row(
                name="aime24",
                problem=ex["problem"],
                ground_truth=gt,
                index=i,
                extra={"id": ex.get("id"), "url": ex.get("url")},
            )
        )
    return _to_dataset(rows)


def build_aime25() -> datasets.Dataset:
    ds = datasets.load_dataset("yentinglin/aime_2025", split="train")
    rows = []
    for i, ex in enumerate(ds):
        rows.append(
            _make_row(
                name="aime25",
                problem=ex["problem"],
                ground_truth=ex["answer"],
                index=i,
            )
        )
    return _to_dataset(rows)


def build_amc23() -> datasets.Dataset:
    ds = datasets.load_dataset("knoveleng/AMC-23", split="train")
    rows = []
    for i, ex in enumerate(ds):
        problem = ex.get("problem") or ex.get("question")
        rows.append(
            _make_row(
                name="amc23",
                problem=problem,
                ground_truth=ex["answer"],
                index=i,
            )
        )
    return _to_dataset(rows)


def build_math500() -> datasets.Dataset:
    ds = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = []
    for i, ex in enumerate(ds):
        rows.append(
            _make_row(
                name="math500",
                problem=ex["problem"],
                ground_truth=ex["answer"],
                index=i,
                extra={"unique_id": ex.get("unique_id"), "subject": ex.get("subject"), "level": ex.get("level")},
            )
        )
    return _to_dataset(rows)


def build_gaokao2023en() -> datasets.Dataset:
    ds = datasets.load_dataset("MARIO-Math-Reasoning/Gaokao2023-Math-En", split="train")
    rows = []
    for i, ex in enumerate(ds):
        rows.append(
            _make_row(
                name="gaokao2023en",
                problem=ex["question"],
                ground_truth=ex["answer"],
                index=i,
                extra={"source": ex.get("source")},
            )
        )
    return _to_dataset(rows)


def build_olympiadbench() -> datasets.Dataset:
    ds = datasets.load_dataset("knoveleng/OlympiadBench", split="train")
    rows = []
    skipped = 0
    for i, ex in enumerate(ds):
        final = ex.get("final_answer")
        if isinstance(final, list) and final:
            gt = ", ".join(str(x) for x in final)
        elif isinstance(final, str) and final.strip():
            gt = final
        else:
            boxed = _extract_boxed(str(ex.get("answer", "")))
            if boxed is None:
                skipped += 1
                continue
            gt = boxed
        rows.append(
            _make_row(
                name="olympiadbench",
                problem=ex["question"],
                ground_truth=gt,
                index=i,
                extra={
                    "subfield": ex.get("subfield"),
                    "is_multiple_answer": ex.get("is_multiple_answer"),
                    "answer_type": ex.get("answer_type"),
                },
            )
        )
    if skipped:
        print(f"[OlympiadBench] skipped {skipped} rows without extractable answer")
    return _to_dataset(rows)


def build_minerva() -> datasets.Dataset:
    ds = datasets.load_dataset("knoveleng/Minerva-Math", split="train")
    rows = []
    skipped = 0
    for i, ex in enumerate(ds):
        gt = _extract_boxed(ex["solution"])
        if gt is None:
            skipped += 1
            continue
        rows.append(
            _make_row(
                name="minerva",
                problem=ex["problem"],
                ground_truth=gt,
                index=i,
                extra={"type": ex.get("type")},
            )
        )
    if skipped:
        print(f"[Minerva] skipped {skipped} rows without \\boxed{{}} answer")
    return _to_dataset(rows)


BUILDERS: dict[str, Callable[..., datasets.Dataset]] = {
    "aime24": build_aime24,
    "aime25": build_aime25,
    "amc23": build_amc23,
    "math500": build_math500,
    "gaokao2023en": build_gaokao2023en,
    "olympiadbench": build_olympiadbench,
    "minerva": build_minerva,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default="data",
        help="Root directory; each benchmark is written to <data_root>/<name>/<name>.parquet",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(BUILDERS.keys()),
        choices=list(BUILDERS.keys()),
        help="Benchmarks to build",
    )
    args = parser.parse_args()

    data_root = os.path.expanduser(args.data_root)
    os.makedirs(data_root, exist_ok=True)

    written: list[str] = []
    for name in args.benchmarks:
        print(f"Building {name} ...", flush=True)
        ds = BUILDERS[name]()
        out_dir = os.path.join(data_root, name)
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"{name}.parquet")
        ds.to_parquet(out)
        example_path = os.path.join(out_dir, f"{name}_example.json")
        with open(example_path, "w", encoding="utf-8") as f:
            json.dump(ds[0], f, indent=2, ensure_ascii=False)
        print(f"  -> {out} ({len(ds)} rows)")
        written.append(out)

    print("File list:\n[" + ", ".join(repr(p) for p in written) + "]")


if __name__ == "__main__":
    main()
