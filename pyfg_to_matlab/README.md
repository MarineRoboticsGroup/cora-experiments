# PyFactorGraph -> CORA Matlab Problem

The functions in this directory are used to convert a `FactorGraphData` object
into the format that our MATLAB implementation of CORA expects.

We tried to clean this up and remove unused code, but as these functions are
likely just called and are unlikely to need modifications we did not perform
exhaustive cleanup.

Also, the file `pymanopt_helpers.py` contains much of the code needed to begin a
Python implementation of CORA. Right now, we just use some if it's utility
functions and have stepped away from a Python implementation for a few reasons
(mainly improved MATLAB capabilities for a few linear-algebraic operations) but
welcome anyone interested in exploring this.