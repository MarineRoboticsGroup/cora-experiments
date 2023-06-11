import matlab.engine
import os
from typing import Optional


def run_cora(
    cora_matlab_dirpath: str = os.path.expanduser(
        "~/range-only-slam-mission-control/cora/MATLAB"
    ),
    experiment_fpath: Optional[str] = None,
    experiment_dir: Optional[str] = None,
    show_animation: bool = True,
    animation_show_gt: bool = True,
    look_for_cached_solns: bool = False,
):
    assert os.path.isdir(
        cora_matlab_dirpath
    ), f"Did not find CORA directory: {cora_matlab_dirpath}, please check that the path is correct"

    assert (
        experiment_fpath is not None or experiment_dir is not None
    ), "Must provide either an experiment file path or an experiment directory"
    assert (
        experiment_fpath is None or experiment_dir is None
    ), "Cannot provide both an experiment file path and an experiment directory"

    if experiment_fpath is None:
        assert experiment_dir is not None
        experiment_fpath = os.path.join(experiment_dir, "factor_graph.mat")

    assert os.path.isfile(
        experiment_fpath
    ), f"Experiment file does not exist: {experiment_fpath}"

    # start MATLAB engine
    eng = matlab.engine.start_matlab()

    # navigate to the directory where the script is located (${HOME}/range-only-slam-mission-control/cora/MATLAB)
    cora_dir = os.path.join(cora_matlab_dirpath)
    eng.cd(cora_dir, nargout=0)

    # add all subdirectories to the MATLAB path
    eng.addpath(eng.genpath(cora_dir), nargout=0)

    # run the Python/MATLAB entry point script for CORA
    eng.cora_python_interface(
        experiment_fpath,
        show_animation,
        animation_show_gt,
        look_for_cached_solns,
        nargout=0,
    )
