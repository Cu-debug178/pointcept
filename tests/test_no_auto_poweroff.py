import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHELL_FILES = [
    *sorted(ROOT.glob("*.sh")),
    *sorted((ROOT / "scripts").glob("*.sh")),
    *sorted((ROOT / ".ipynb_checkpoints").glob("*.sh")),
]
FORBIDDEN = re.compile(
    r"(?:^[ \t]*|;[ \t]*|&&[ \t]*|\|\|[ \t]*|[ \t]\|[ \t]+|\([ \t]*|\{[ \t]*)"
    r"(?:sudo[ \t]+)?"
    r"(?:/(?:usr/)?s?bin/)?"
    r"(?:shutdown|poweroff|halt)"
    r"(?=[ \t;&|)]|$)",
    re.MULTILINE,
)


class NoAutomaticPoweroffTest(unittest.TestCase):
    def test_power_command_patterns_are_recognized(self):
        commands = (
            "shutdown -c",
            "/usr/bin/shutdown --help",
            "/sbin/shutdown -h now",
            "sudo /usr/sbin/poweroff",
            "task; halt",
            "task | /usr/bin/shutdown -h now",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertRegex(command, FORBIDDEN)

    def test_shell_entrypoints_do_not_invoke_power_commands(self):
        violations = []
        for path in SHELL_FILES:
            text = path.read_text(encoding="utf-8")
            if FORBIDDEN.search(text):
                violations.append(str(path.relative_to(ROOT)))
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
