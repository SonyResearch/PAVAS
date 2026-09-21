import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "pavas"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import core  # noqa: F401
from core.model.sequence_config import CONFIG_16K  # noqa: F401

print("smoke_test_imports: ok")
