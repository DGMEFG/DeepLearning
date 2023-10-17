import sys
from pathlib import Path

def init():
    sys.path.append(Path(__file__).parent.parent.absolute().__str__())
