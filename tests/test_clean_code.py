import ast
import io
import tokenize
from pathlib import Path


def test_source_has_no_comments_or_docstrings():
    source_root = Path(__file__).parents[1] / "src" / "popncde"
    for path in source_root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        assert not any(token.type == tokenize.COMMENT for token in tokens), path.name
        tree = ast.parse(source)
        nodes = [tree]
        nodes.extend(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        )
        assert all(ast.get_docstring(node) is None for node in nodes), path.name
