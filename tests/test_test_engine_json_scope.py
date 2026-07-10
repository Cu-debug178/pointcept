import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_ENGINE = ROOT / "pointcept" / "engines" / "test.py"


class TestEngineJsonScopeTest(unittest.TestCase):
    def test_json_is_only_imported_at_module_scope(self):
        tree = ast.parse(TEST_ENGINE.read_text(encoding="utf-8"))
        module_imports = []
        local_imports = []

        for node in tree.body:
            if isinstance(node, ast.Import):
                module_imports.extend(
                    alias.name for alias in node.names if alias.name == "json"
                )

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Import):
                    for alias in child.names:
                        if alias.name == "json":
                            local_imports.append((node.name, child.lineno))

        self.assertEqual(module_imports, ["json"])
        self.assertEqual(local_imports, [])


if __name__ == "__main__":
    unittest.main()
