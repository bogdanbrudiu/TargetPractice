import sys
from pathlib import Path


# Ensure `import targetweb` works regardless of pytest's chosen rootdir.
TARGETWEB_ROOT = Path(__file__).resolve().parent.parent
if str(TARGETWEB_ROOT) not in sys.path:
    sys.path.insert(0, str(TARGETWEB_ROOT))
