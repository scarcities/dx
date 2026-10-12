"""Pin cost assumptions before config.py is imported. The unit tests use pinned configs (see the
top of test_bot_offline.py); the CLI tests run with config.py defaults and must not depend on a
calibrated SLIPPAGE_PER_SIDE in your .env, or start_mac.sh's pre-flight could fail."""
import os

os.environ["SLIPPAGE_PER_SIDE"] = "0.0005"
