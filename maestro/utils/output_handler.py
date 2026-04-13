"""
Saves generated code to the output/ directory with the correct extension.
"""
import re
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

_EXTENSIONS = {
    "python": ".py", "javascript": ".js", "typescript": ".ts",
    "java": ".java", "cpp": ".cpp", "csharp": ".cs",
    "go": ".go", "rust": ".rs", "html": ".html",
    "css": ".css", "sql": ".sql", "bash": ".sh",
}

_LANG_PATTERNS = [
    ("python",     r"\bdef\s+\w+\s*\(|\bimport\s+\w+|\bfrom\s+\w+\s+import"),
    ("javascript", r"\bfunction\s+\w+\s*\(|\bconst\s+\w+\s*=|\blet\s+\w+\s*="),
    ("typescript", r"\binterface\s+\w+|\btype\s+\w+\s*="),
    ("java",       r"\bpublic\s+class\s+\w+|\bpublic\s+static\s+void\s+main"),
    ("cpp",        r"#include\s*<|\bstd::"),
    ("csharp",     r"\busing\s+System|\bnamespace\s+\w+"),
    ("go",         r"\bpackage\s+main|\bfunc\s+\w+\s*\("),
    ("rust",       r"\bfn\s+main\s*\(\)|\buse\s+std::"),
    ("html",       r"<html|<body|<!DOCTYPE"),
    ("sql",        r"\bSELECT\b|\bINSERT\b|\bCREATE\s+TABLE\b"),
    ("bash",       r"^#!/bin/bash|^#!/bin/sh"),
]


def _detect_language(code: str, task: str = "") -> str:
    task_l = task.lower()
    for lang in ["python", "javascript", "typescript", "java", "cpp",
                 "csharp", "go", "rust", "html", "css", "sql", "bash"]:
        if lang in task_l:
            return lang
    for lang, pattern in _LANG_PATTERNS:
        if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
            return lang
    return "python"


def _extract_name(code: str, lang: str) -> str:
    if lang == "python":
        m = re.search(r"class\s+(\w+)", code) or re.search(r"def\s+(\w+)\s*\(", code)
        if m:
            return m.group(1).lower()
    elif lang in ("javascript", "typescript"):
        m = re.search(r"class\s+(\w+)", code) or re.search(r"function\s+(\w+)\s*\(", code)
        if m:
            return m.group(1).lower()
    elif lang == "java":
        m = re.search(r"public\s+class\s+(\w+)", code)
        if m:
            return m.group(1)
    return f"output_{datetime.now().strftime('%H%M%S')}"


def _clean(code: str) -> str:
    code = re.sub(r"^```\w*\n", "", code, flags=re.MULTILINE)
    code = re.sub(r"\n```$", "", code, flags=re.MULTILINE)
    return code.strip()


class OutputHandler:
    def save_code(self, code: str, task: str = "") -> Path:
        code = _clean(code)
        lang = _detect_language(code, task)
        ext  = _EXTENSIONS.get(lang, ".txt")
        name = _extract_name(code, lang)
        path = OUTPUT_DIR / f"{name}{ext}"
        counter = 1
        while path.exists():
            path = OUTPUT_DIR / f"{name}_{counter}{ext}"
            counter += 1
        path.write_text(code, encoding="utf-8")
        return path
