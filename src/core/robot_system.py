import sys
from pathlib import Path
sys.path.insert(1, str(Path(__file__).parent.parent.parent))

import numpy as np
import time
from typing import Dict

class RobotSystem:
    def __init__(self, config=Dict):
        self.running = False


    def run(self):
        self.running = True