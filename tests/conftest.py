from pathlib import Path
import sys

PROJECT_ROOT = Path(r"/content/football_gnn_experiment5_v2/football_gnn_experiment5_global_v2")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
