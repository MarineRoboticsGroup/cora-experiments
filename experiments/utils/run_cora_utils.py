import matlab.engine
import os
from typing import Optional
from .evaluate_utils import check_dir_ready_for_evaluation
from .logging_utils import logger


def experiment_inputs_are_valid(
    cora_matlab_dirpath: str,
    experiment_fpath: Optional[str],
    experiment_dir: Optional[str],
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
        experiment_fpath = os.path.join(experiment_dir, "factor_graph.pyfg")

    assert os.path.isfile(
        experiment_fpath
    ), f"Experiment file does not exist: {experiment_fpath}"

    return True


def get_experiment_fpath(
    experiment_fpath: Optional[str], experiment_dir: Optional[str]
):
    if experiment_fpath is None:
        assert experiment_dir is not None
        experiment_fpath = os.path.join(experiment_dir, "factor_graph.pyfg")
    return experiment_fpath


def setup_matlab_engine(cora_matlab_dirpath: str):
    # start MATLAB engine
    eng = matlab.engine.start_matlab()

    cora_dir = os.path.join(cora_matlab_dirpath)
    eng.cd(cora_dir, nargout=0)

    # add all subdirectories to the MATLAB path
    eng.addpath(eng.genpath(cora_dir), nargout=0)
    return eng


def run_cora(
    cora_matlab_dirpath: str = os.path.expanduser(
        "~/range-only-slam-mission-control/cora/MATLAB"
    ),
    experiment_fpath: Optional[str] = None,
    experiment_dir: Optional[str] = None,
    show_animation: bool = True,
    animation_show_gt: bool = True,
    look_for_cached_cora_solns: bool = False,
    solve_marginalized_problem: bool = False,
    save_iterates_info: bool = True,
):
    assert experiment_inputs_are_valid(
        cora_matlab_dirpath, experiment_fpath, experiment_dir
    )

    experiment_fpath = get_experiment_fpath(experiment_fpath, experiment_dir)

    # check if we can skip CORA
    if look_for_cached_cora_solns and not show_animation:
        try:
            check_dir_ready_for_evaluation(experiment_dir)
            logger.info(
                f"Found cached solutions for experiment: {experiment_dir}, skipping CORA"
            )
            return
        except FileNotFoundError:
            pass

    eng = setup_matlab_engine(cora_matlab_dirpath)
    eng.cora_python_interface(
        experiment_fpath,
        show_animation,
        animation_show_gt,
        look_for_cached_cora_solns,
        solve_marginalized_problem,
        save_iterates_info,
        nargout=0,
    )
