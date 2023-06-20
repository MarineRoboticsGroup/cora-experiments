from os.path import dirname, abspath, join, isdir
import os
from typing import List

EXPERIMENTS_UTILS_DIR = dirname(abspath(__file__))
EXPERIMENTS_DIR = dirname(EXPERIMENTS_UTILS_DIR)
REPO_BASE_DIR = dirname(EXPERIMENTS_DIR)
DATA_DIR = join(REPO_BASE_DIR, "data")
MANHATTAN_DATA_DIR = join(DATA_DIR, "manhattan")
PYFG_TO_MATLAB_DIR = join(REPO_BASE_DIR, "pyfg_to_matlab")

dirs = [
    EXPERIMENTS_UTILS_DIR,
    EXPERIMENTS_DIR,
    REPO_BASE_DIR,
    DATA_DIR,
    PYFG_TO_MATLAB_DIR,
]

for d in dirs:
    assert isdir(d), f"Directory {d} not found"


def get_leaf_dirs(root_dir: str) -> List[str]:
    """Recursively finds all of the leaf directories under the root directory.

    Args:
        root_dir (str): the root directory

    Returns:
        List[str]: the list of leaf directories
    """
    leaf_dirs = []
    assert isdir(root_dir)
    for root, dirs, files in os.walk(root_dir):
        if len(dirs) == 0:
            leaf_dirs.append(root)
    return leaf_dirs
