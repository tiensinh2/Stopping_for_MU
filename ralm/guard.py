"""Guard cách ly §1.1: file chọn k̂ không được import module oracle (retrain / eval_oracle)."""
from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN = {"retrain", "eval_oracle"}


def assert_no_oracle_imports(file: str):
    tree = ast.parse(Path(file).read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "")] + [a.name for a in node.names]
        for n in names:
            if set(n.replace("ralm.", "").split(".")) & FORBIDDEN:
                raise AssertionError(f"{Path(file).name} import '{n}' vi phạm retrain-only-for-eval (§1.1)")
