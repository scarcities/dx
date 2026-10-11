"""Pin cost assumptions before config.py is imported: the tests hard-code 5 bp per side, and a
calibrated SLIPPAGE_PER_SIDE in your .env must not make start_mac.sh's pre-flight fail."""
import os

os.environ["SLIPPAGE_PER_SIDE"] = "0.0005"
