"""Interpreter selection for the separate learning and Isaac Sim environments."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def _interpreter(variable, default):
    path = Path(os.environ.get(variable, str(default))).expanduser().absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError(f'{variable}: executable not found: {path}. '
                                'See repro/README.md and set the interpreter path.')
    return str(path)


def learning_python():
    return _interpreter('DEMOGRASP_LEARNING_PYTHON', sys.executable)


def isaac_python():
    return _interpreter('DEMOGRASP_ISAAC_PYTHON', ROOT / '.venvs/isaacsim/bin/python')
