import sys
from pathlib import Path
sys.path.insert(1, str(Path(__file__).parent.parent))

from src.common.data_logger import DataLogger



if __name__ == "__main__":
    data = DataLogger.load('log/test_controller/', 'control')

    a = 1