import numpy as np
from py_factor_graph.io.pyfg_text import read_from_pyfg_text
from os.path import join
from evo.tools import plot

from utils.run_experiments import run_experiments, ExperimentConfigs
from utils.paths import DATA_DIR

import logging, coloredlogs

logger = logging.getLogger(__name__)
field_styles = {
    "filename": {"color": "green"},
    "levelname": {"bold": True, "color": "black"},
    "name": {"color": "blue"},
}
coloredlogs.install(
    level="INFO",
    fmt="[%(filename)s:%(lineno)d] %(name)s %(levelname)s - %(message)s",
    field_styles=field_styles,
)


if __name__ == "__main__":
    run_noisy_experiments = False  # whether to add artificial noise to the problem and also run CORA on the noisy problems

    TIERS_EXPERIMENT_DIR = join(DATA_DIR, "tiers")
    BASE_EXPERIMENT = {
        TIERS_EXPERIMENT_DIR: np.array([0.0, 0.0]),
    }

    # make noisy version of problem
    if run_noisy_experiments:
        trans_stddev = 0.01
        rot_stddev = 0.005
        modified_dir = join(TIERS_EXPERIMENT_DIR, "modified")
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

    pyfg = read_from_pyfg_text(join(TIERS_EXPERIMENT_DIR, "factor_graph.pyfg"))

    # configure how we want to run experiments
    exp_config = ExperimentConfigs(
        run_experiments_with_added_noise=True,
        use_cached_problems=True,
        animate_trajs=False,
        run_cora=False,
        show_solver_animation=False,
        show_gt_cora_animation=True,
        look_for_cached_cora_solns=True,
        perform_evaluation=True,
        use_cached_trajs=True,
        desired_plot_modes=[plot.PlotMode.xy],
    )

    # run experiments
    run_experiments(pyfg, EXPERIMENTS, exp_config)
