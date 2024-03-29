# cora-experiments
Experiments from "Certifiably Correct Range-Aided SLAM"

## Dependencies

We packaged a working `conda` environment which handles most of the
dependencies. However, there are a few repos that you will need to install
manually as of now. The below snippet should get you started.

```bash
# clone the repo
cd ~
git clone git@github.com:MarineRoboticsGroup/cora-experiments.git

# create a conda environment from environment.yml
cd cora-experiments
conda env create -f environment.yml
conda activate cora

# install other dependencies
cd ~
git clone git@github.com:MarineRoboticsGroup/cora.git # CORA (MATLAB)
git clone git@github.com:MarineRoboticsGroup/PyFactorGraph.git # PyFactorGraph (Python) - to hold problems
git clone git@github.com:MarineRoboticsGroup/gtsam-range-aided-slam.git # our GTSAM-based solver (Python)
cd ~/PyFactorGraph; pip install -e . # install PyFactorGraph
cd ~/gtsam-range-aided-slam; pip install -e . # install our GTSAM solver

# optional (only for Manhattan experiments)
cd ~
git clone git@github.com:MarineRoboticsGroup/manhattan-world-sim.git
cd ~/manhattan-world-sim; pip install -e . # install manhattan-world-sim
```

## Running experiments

The different experiments are all inside our `experiments/` directory. You
should be able to run any of the scripts to recreate the results in our paper.
