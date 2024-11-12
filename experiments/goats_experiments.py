import numpy as np
from py_factor_graph.io.pyfg_text import read_from_pyfg_text

from os.path import join
from evo.tools import plot

from utils.run_experiments import run_experiments, ExperimentConfigs
from utils.paths import DATA_DIR


if __name__ == "__main__":
    run_noisy_experiments = False  # whether to add artificial noise to the problem and also run CORA on the noisy problems

    goats_exps = [f"goats_{num}" for num in [14, 15, 16]]

    for exp in goats_exps:

        EXP_DIR = join(DATA_DIR, "goats", exp)

        BASE_EXPERIMENT = {
            EXP_DIR: np.array([0.0, 0.0]),
        }
        EXPERIMENTS = BASE_EXPERIMENT

        marine_experiment_fpath = join(EXP_DIR, "factor_graph.pyfg")
        pyfg = read_from_pyfg_text(marine_experiment_fpath)
        exp_config = ExperimentConfigs(
            run_experiments_with_added_noise=False,
            use_cached_problems=True,
            animate_trajs=False,
            run_cora=False,
            solve_marginalized_problem=True,
            show_solver_animation=True,
            show_gt_cora_animation=True,
            look_for_cached_cora_solns=False,
            perform_evaluation=True,
            use_cached_trajs=False,
            desired_plot_modes=[plot.PlotMode.xy],
            overlay_river_map=False,
        )
        run_experiments(pyfg, EXPERIMENTS, exp_config)