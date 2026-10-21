"""Sanitized test/runtime/source provenance. Run from the repository root."""

import argparse
import hashlib
import json
import platform
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy
import pytest
import torch
import transformers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", required=True)
    parser.add_argument("--expected-tests", required=True, type=int)
    parser.add_argument("--local-passed", required=True, type=int)
    parser.add_argument("--local-skipped", required=True, type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path.cwd()
    files = [root / "pyproject.toml"]
    for folder in ("src", "tests", "examples"):
        files.extend((root / folder).rglob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    suite = ET.parse(args.junit).getroot().find("testsuite")
    counts = {key: int(suite.attrib[key]) for key in ("tests", "errors", "failures", "skipped")}
    assert counts["tests"] == args.expected_tests
    assert all(counts[key] == 0 for key in ("errors", "failures", "skipped"))
    assert torch.cuda.is_available(), "This summary requires an actual CUDA test node"
    version = re.search(r'^version = "([^"]+)"', (root / "pyproject.toml").read_text(), re.M)[1]
    report = {
        "package_version": version,
        "python_source_and_tests_sha256": digest.hexdigest(),
        "hash_scope": (
            "relative file paths and bytes in src/tests/examples Python files plus pyproject.toml"
        ),
        "cuda_test_run": {
            **counts,
            "seconds": float(suite.attrib["time"]),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "transformers": transformers.__version__,
            "numpy": numpy.__version__,
            "pytest": pytest.__version__,
        },
        "local_macos_cpu": {
            "passed": args.local_passed,
            "skipped": args.local_skipped,
            "failed": 0,
        },
        "model_check": "Exact bytes, next logits and greedy continuation vs original model cache",
        "exclusions": "No HTTP load test, p99, large pretrained model or model-quality benchmark.",
    }
    encoded = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
