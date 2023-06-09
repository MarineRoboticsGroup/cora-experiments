from os.path import dirname, abspath, join, isdir

EXPERIMENTS_UTILS_DIR = dirname(abspath(__file__))
EXPERIMENTS_DIR = dirname(EXPERIMENTS_UTILS_DIR)
REPO_BASE_DIR = dirname(EXPERIMENTS_DIR)
DATA_DIR = join(REPO_BASE_DIR, "data")
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
