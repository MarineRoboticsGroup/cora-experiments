from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.modifiers import add_error_to_all_odom_measures
from py_factor_graph.io.pyfg_text import save_to_pyfg_text
import numpy as np
import os
from os.path import join, isfile
from attrs import define, field
from typing import List, Dict
import evo.tools.plot as plot

# insert the pyfg_to_matlab directory into the path

from .evaluate_utils import (
    get_aligned_traj_results_in_dir,
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
    noisy_pyfg_save_path = join(save_dir, "factor_graph.pyfg")
    if use_cached_problem and isfile(noisy_pyfg_save_path):
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

    save_to_pyfg_text(noisy_fg, noisy_pyfg_save_path)


def _perform_evaluation(
    eval_dir: str,
    desired_plot_modes: list,
    use_cached_trajs: bool,
    overlay_river_map=False,
):
    aligned_trajs = get_aligned_traj_results_in_dir(
        eval_dir, use_cached_results=use_cached_trajs
    )
    make_evo_traj_plots(
        aligned_trajs,
        eval_dir,
        show_plots=True,
        valid_plot_views=desired_plot_modes,
        overlay_river_image=overlay_river_map,
    )
    make_evo_ape_plots_from_trajs(aligned_trajs, eval_dir, show_plots=False)


@define
class ExperimentConfigs:
    run_experiments_with_added_noise: bool = field()
    use_cached_problems: bool = field()
    animate_trajs: bool = field()

    run_cora: bool = field()
    solve_marginalized_problem: bool = field()
    show_solver_animation: bool = field()
    show_gt_cora_animation: bool = field()
    look_for_cached_cora_solns: bool = field()

    perform_evaluation: bool = field()
    use_cached_trajs: bool = field()
    desired_plot_modes: List[plot.PlotMode] = field()
    overlay_river_map: bool = field(default=False)


def get_all_experiments_to_run(base_experiment_dir: str, exp_config: ExperimentConfigs):
    base_experiment = {
        base_experiment_dir: np.array([0.0, 0.0]),
    }
    if exp_config.run_experiments_with_added_noise:
        # make noisy version of problem
        trans_stddev = 0.05
        rot_stddev = 0.05

        MODS = {
            "trans_only": np.array([trans_stddev, 0.0]),
            "rot_only": np.array([0.0, rot_stddev]),
            "trans_and_rot": np.array([trans_stddev, rot_stddev]),
        }
        modified_dir = join(base_experiment_dir, "modified")
        NOISY_EXPERIMENTS = {
            join(modified_dir, mod_type, "noisy_problem"): mod_vals
            for mod_type, mod_vals in MODS.items()
        }
        NOISIER_EXPERIMENTS = {
            join(modified_dir, mod_type, "noisier_problem"): 2 * mod_vals
            for mod_type, mod_vals in MODS.items()
        }

        # join all experiments
        return {
            # **base_experiment,
            # **NOISY_EXPERIMENTS,
            **NOISIER_EXPERIMENTS,
        }
    else:
        return base_experiment


def run_experiments(
    base_pyfg: FactorGraphData,
    experiments: Dict[str, np.ndarray],
    config: ExperimentConfigs,
):
    assert all([len(noise_vals) == 2 for exp_dir, noise_vals in experiments.items()])

    # make noisy copies of problem and save
    for exp_dir, noise_mods in experiments.items():
        if not config.run_experiments_with_added_noise and "modified" in exp_dir:
            print(
                f"Skipping {exp_dir} because config.run_experiments_with_added_noise is False"
            )
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
            look_for_cached_cora_solns=config.look_for_cached_cora_solns,
            solve_marginalized_problem=config.solve_marginalized_problem,
        )

    # evaluate results
    for exp_dir in experiments.keys():
        if not config.perform_evaluation:
            continue

        _perform_evaluation(
            exp_dir,
            config.desired_plot_modes,
            use_cached_trajs=config.use_cached_trajs,
            overlay_river_map=config.overlay_river_map,
        )
