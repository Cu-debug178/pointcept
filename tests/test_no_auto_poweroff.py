import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHELL_FILES = [ROOT / "activate_env.sh", *sorted((ROOT / "scripts").glob("*.sh"))]
FORBIDDEN = re.compile(
    r"^[ \t]*(?:sudo[ \t]+)?(?:/sbin/)?(?:shutdown|poweroff|halt)(?:[ \t]|$)",
    re.MULTILINE,
)


class NoAutomaticPoweroffTest(unittest.TestCase):
    def test_shell_entrypoints_do_not_invoke_power_commands(self):
        violations = []
        for path in SHELL_FILES:
            text = path.read_text(encoding="utf-8")
            if FORBIDDEN.search(text):
                violations.append(str(path.relative_to(ROOT)))
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
