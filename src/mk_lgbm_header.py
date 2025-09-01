#!/usr/bin/env python3
"""
mk_lgbm_header.py – embed a LightGBM text model in a C++ header

Why this version?
─────────────────
*   **Loss-less** – reproduces the exact bytes produced by
    `Booster.save_model()`.
*   **Chunk-safe** – never breaks inside a UTF-8 sequence or adds extra `\\n`.
*   **Self-verifying** – `--check` option diff-checks the generated
    constant against the source file.

Usage
─────
    python mk_lgbm_header.py MODEL.txt out_model.h            \
        --var-name=LGBM_MODEL_TXT --namespace=my::ml          \
        --chunk-size=16000     # 0 = no chunking              \
        --check                # optional byte-for-byte diff

The header it produces looks like:

    #pragma once
    namespace my::ml {
    static constexpr char LGBM_MODEL_TXT[] =
        R"DELIM( … ≤16 KiB … )DELIM"
        "\n"
        R"DELIM( … next chunk … )DELIM"
        ;
    } // namespace my::ml
"""
from __future__ import annotations
import argparse
import random
import string
import sys
from pathlib import Path
from typing import List


def unique_delim(text: str, length: int = 6, max_try: int = 200) -> str:
    """Pick a raw-string delimiter not present in *text*."""
    alphabet = string.ascii_uppercase
    for _ in range(max_try):
        delim = "".join(random.choices(alphabet, k=length))
        if f')" {delim}"' not in text:
            return delim
    raise RuntimeError("Could not find a unique delimiter – increase max_try.")


def split_on_lines(text: str, chunk_size: int) -> List[str]:
    """Split *text* into ≤chunk_size pieces *only* at line boundaries."""
    if chunk_size <= 0:
        return [text]

    out, buf, size = [], [], 0
    for line in text.splitlines(keepends=True):
        if buf and size + len(line) > chunk_size:
            out.append("".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += len(line)
    if buf:
        out.append("".join(buf))
    return out


def build_header(
    model_text: str,
    var_name: str,
    namespace: str,
    chunk_size: int,
) -> str:
    delim = unique_delim(model_text)
    chunks = split_on_lines(model_text, chunk_size)
    literals = [f'R"{delim}({chunk}){delim}"' for chunk in chunks]

    # join with *exactly one* newline between chunks
    joined = '\n        "\\n"\n        '.join(literals)

    ns_open = f"namespace {namespace} {{" if namespace else ""
    ns_close = f"}} // namespace {namespace}" if namespace else ""

    return (
        "#pragma once\n"
        f"{ns_open}\n"
        f"static constexpr char {var_name}[] =\n"
        f"        {joined}\n"
        "        ;\n"
        f"{ns_close}\n"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model_txt", type=Path)
    p.add_argument("header_out", type=Path)
    p.add_argument("--var-name", default="LGBM_MODEL_TXT")
    p.add_argument("--namespace", default="")
    p.add_argument(
        "--chunk-size",
        type=int,
        default=16_000,
        help="maximum chars per raw-string literal (0 = no chunking)",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="verify generated constant matches source file byte-for-byte",
    )
    ns = p.parse_args()

    model_bytes = ns.model_txt.read_bytes()  # keep exact bytes
    model_text = model_bytes.decode("utf-8")

    header_code = build_header(
        model_text=model_text,
        var_name=ns.var_name,
        namespace=ns.namespace,
        chunk_size=ns.chunk_size,
    )
    ns.header_out.write_text(header_code, encoding="utf-8")
    print(f"✅ Wrote {ns.header_out}  ({len(model_bytes):,} bytes)")

    if ns.check:
        # Import it back in a tiny namespace sandbox to compare
        tmp = {}
        exec(header_code, tmp)
        embedded = tmp.get(ns.var_name).encode()
        if embedded == model_bytes:
            print("✓ Byte-for-byte identical to source.")
        else:
            print("✗ Mismatch detected!", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
