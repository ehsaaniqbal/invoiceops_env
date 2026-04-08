import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPENENV_PROJECT = ROOT.parent / "OpenEnv"
if not OPENENV_PROJECT.exists():
    OPENENV_PROJECT = ROOT.parent / "markov" / "OpenEnv"


def test_openenv_validate_passes() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(OPENENV_PROJECT),
            "openenv",
            "validate",
            str(ROOT),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[OK]" in result.stdout
