from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.modifiers import add_error_to_all_odom_measures
import numpy as np
import os
from os.path import join, isfile
from attrs import define, field
from typing import List, Dict
import evo.tools.plot as plot

# insert the pyfg_to_matlab directory into the path
import sys
from .paths import REPO_BASE_DIR

sys.path.insert(0, REPO_BASE_DIR)
from pyfg_to_matlab.matlab_interfaces import export_fg_to_matlab_cora_format

from .evaluate_utils import (
    evaluate_results,
    make_evo_ape_plots_from_trajs,
    make_evo_traj_plots,
)
from .run_cora_utils import run_cora


def _generate_noisy_problem(
    base_fg: FactorGraphData,
    noise_mod: np.ndarray,
    save_dir: str,
    use_cached_problem: bool,
    animate_trajs: bool = False,
):
    noisy_problem_mat_file = join(save_dir, "factor_graph.mat")
    noisy_pyfg_save_path = join(save_dir, "factor_graph.pickle")
    if (
        use_cached_problem
        and isfile(noisy_pyfg_save_path)
        and isfile(noisy_problem_mat_file)
    ):
        return

    trans_mod, rot_mod = noise_mod
    if trans_mod != 0.0 or rot_mod != 0.0:
        noisy_fg = add_error_to_all_odom_measures(base_fg, trans_mod, rot_mod)
    else:
        noisy_fg = base_fg

    if animate_trajs:
        noisy_fg.animate_odometry(
            show_gt=True,
            pause_interval=0.001,
            draw_range_lines=True,
            draw_range_circles=False,
            num_timesteps_keep_ranges=10,
        )

    # Save noisy problem
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    noisy_fg.save_to_file(noisy_pyfg_save_path)
    export_fg_to_matlab_cora_format(noisy_fg, matlab_filepath=noisy_problem_mat_file)


def _perform_evaluation(eval_dir: str, desired_plot_modes: list):
    ape_error_dfs, aligned_trajs = evaluate_results(eval_dir)
    make_evo_traj_plots(
        aligned_trajs, eval_dir, show_plots=True, valid_plot_views=desired_plot_modes
    )
    make_evo_ape_plots_from_trajs(aligned_trajs, eval_dir)


@define
class ExperimentConfigs:
    run_experiments_with_added_noise: bool = field()
    use_cached_problems: bool = field()
    animate_trajs: bool = field()

    run_cora: bool = field()
    show_solver_animation: bool = field()
    show_gt_cora_animation: bool = field()
    look_for_cached_solns: bool = field()

    perform_evaluation: bool = field()
    desired_plot_modes: List[plot.PlotMode] = field()


def run_experiments(
    base_pyfg: FactorGraphData,
    experiments: Dict[str, np.ndarray],
    config: ExperimentConfigs,
):
    assert all([len(noise_vals) == 2 for exp_dir, noise_vals in experiments.items()])

    # make noisy copies of problem and save
    for exp_dir, noise_mods in experiments.items():
        if not config.run_experiments_with_added_noise and "modified" in exp_dir:
            continue

        _generate_noisy_problem(
            base_pyfg,
            noise_mods,
            exp_dir,
            config.use_cached_problems,
            animate_trajs=config.animate_trajs,
        )

    # run CORA on the problems
    for exp_dir in experiments.keys():
        if not config.run_cora:
            continue

        run_cora(
            experiment_dir=exp_dir,
            show_animation=config.show_solver_animation,
            animation_show_gt=config.show_gt_cora_animation,
            look_for_cached_solns=config.look_for_cached_solns,
        )

    # evaluate results
    for exp_dir in experiments.keys():
        if not config.perform_evaluation:
            continue

        _perform_evaluation(exp_dir, config.desired_plot_modes)
