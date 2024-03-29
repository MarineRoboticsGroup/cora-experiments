import numpy as np
from py_factor_graph.io.pyfg_text import read_from_pyfg_text

from os.path import join
from evo.tools import plot

from utils.run_experiments import run_experiments, ExperimentConfigs
from utils.paths import DATA_DIR


if __name__ == "__main__":
    run_noisy_experiments = False  # whether to add artificial noise to the problem and also run CORA on the noisy problems

    MARINE_DIR = join(DATA_DIR, "marine")

    BASE_EXPERIMENT = {
        MARINE_DIR: np.array([0.0, 0.0]),
    }

    if run_noisy_experiments:
        # make noisy version of problem
        trans_stddev = 0.01
        rot_stddev = 0.005
        modified_dir = join(MARINE_DIR, "modified")
        MODS = {
            "trans_only": np.array([trans_stddev, 0.0]),
            "rot_only": np.array([0.0, rot_stddev]),
            "trans_and_rot": np.array([trans_stddev, rot_stddev]),
        }

        NOISY_EXPERIMENTS = {
            join(modified_dir, mod_type, "noisy_problem"): mod_vals
            for mod_type, mod_vals in MODS.items()
        }
        NOISIER_EXPERIMENTS = {
            join(modified_dir, mod_type, "noisier_problem"): 2 * mod_vals
            for mod_type, mod_vals in MODS.items()
        }

        # join all experiments
        EXPERIMENTS = {**BASE_EXPERIMENT, **NOISY_EXPERIMENTS, **NOISIER_EXPERIMENTS}
    else:
        EXPERIMENTS = BASE_EXPERIMENT

    marine_experiment_fpath = join(MARINE_DIR, "marine_two_robots.pyfg")
    pyfg = read_from_pyfg_text(marine_experiment_fpath)
    exp_config = ExperimentConfigs(
        run_experiments_with_added_noise=False,
        use_cached_problems=True,
        animate_trajs=False,
        run_cora=False,
        solve_marginalized_problem=True,
        show_solver_animation=True,
        show_gt_cora_animation=True,
        look_for_cached_cora_solns=True,
        perform_evaluation=True,
        use_cached_trajs=True,
        desired_plot_modes=[plot.PlotMode.xy],
        overlay_river_map=True,
    )
    run_experiments(pyfg, EXPERIMENTS, exp_config)
