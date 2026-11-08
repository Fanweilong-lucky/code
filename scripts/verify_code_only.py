"""Reject non-code payloads and common private-path/credential patterns.

This is a local release check, not a guarantee that arbitrary source text is
nonconfidential. Review source edits before sharing. No files are uploaded.
"""

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SUFFIXES = {".py", ".md", ".toml", ".json", ".yaml", ".yml"}
ALLOWED_NAMES = {".gitignore", "requirements.txt", "pre-commit"}
PRIVATE_DIRECTORIES = {
    "data", "dataset", "datasets", "private", "reference", "weights",
    "checkpoints", "outputs", "runs", "logs", "splits", "annotations", "labels",
}
IGNORED_DIRECTORIES = {".git", "__pycache__", ".venv", "venv", ".pytest_cache"}
PATTERNS = {
    "Windows absolute path": re.compile(r"(?i)(?<![A-Za-z])[A-Z]:[\\/]"),
    "workstation home path": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "embedded base64 payload": re.compile(r"[A-Za-z0-9+/]{300,}={0,2}"),
}


def inspect_file(path, root=ROOT):
    relative = path.relative_to(root)
    issues = []
    if path.is_symlink():
        return ["symbolic links are not allowed"]
    if any(part.lower() in PRIVATE_DIRECTORIES for part in relative.parts[:-1]):
        issues.append("private-data/output directory")
    if any(part.lower().endswith(".egg-info") for part in relative.parts):
        issues.append("generated package metadata")
    if path.name not in ALLOWED_NAMES and path.suffix.lower() not in ALLOWED_SUFFIXES:
        issues.append("file type is not on the code/documentation allowlist")
    if path.name.endswith((".local.json", ".private.json")):
        issues.append("private configuration file")
    if path.stat().st_size > 1024 * 1024:
        issues.append("unexpected file size above 1 MiB")
        return issues
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return issues + ["non-UTF-8 or binary payload"]
    if "\x00" in text:
        issues.append("binary payload")
    for description, pattern in PATTERNS.items():
        if pattern.search(text):
            issues.append(description)
    if path.suffix == ".py":
        try:
            ast.parse(text, filename=str(relative))
        except SyntaxError as exc:
            issues.append(f"Python syntax error on line {exc.lineno}")
    if path.suffix == ".json":
        try:
            json.loads(text)
        except json.JSONDecodeError:
            issues.append("invalid JSON")
        if relative.parts[0] != "configs" or not path.name.endswith(".example.json"):
            issues.append("only example configuration JSON is permitted")
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracked", action="store_true", help="Check all files in the Git index")
    args = parser.parse_args()
    if args.tracked:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "-z"], cwd=ROOT,
            check=True, stdout=subprocess.PIPE,
        )
        files = [ROOT / name for name in result.stdout.decode("utf-8").split("\0") if name]
    else:
        files = [path for path in ROOT.rglob("*") if path.is_file()
                 and not any(part in IGNORED_DIRECTORIES for part in path.relative_to(ROOT).parts)]
    failures = []
    for path in sorted(files):
        if not path.is_file():
            failures.append((path.relative_to(ROOT).as_posix(), ["tracked file is missing"]))
            continue
        issues = inspect_file(path)
        if issues:
            failures.append((path.relative_to(ROOT).as_posix(), issues))
    if failures:
        for name, issues in failures:
            print(f"FAIL {name}: {'; '.join(issues)}")
        return 1
    print(f"PASS: {len(files)} code/documentation files; no disallowed payload detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
