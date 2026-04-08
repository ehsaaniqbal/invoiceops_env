import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent

if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

for module_name in list(sys.modules):
    if module_name == "invoiceops_env" or module_name.startswith("invoiceops_env."):
        sys.modules.pop(module_name, None)
