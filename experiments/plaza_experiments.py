import numpy as np
from os.path import join
from evo.tools import plot
from py_factor_graph.io.pyfg_text import read_from_pyfg_text

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

    PLAZA_DIR = join(DATA_DIR, "plaza")
    PLAZA_EXPERIMENTS = ["plaza1", "plaza2"]
    PLAZA_EXPERIMENT_DIRS = [join(PLAZA_DIR, exp) for exp in PLAZA_EXPERIMENTS]

    for base_experiment_dir in PLAZA_EXPERIMENT_DIRS:
        BASE_EXPERIMENT = {
            base_experiment_dir: np.array([0.0, 0.0]),
        }

        if run_noisy_experiments:
            # make noisy version of problem
            trans_stddev = 0.01
            rot_stddev = 0.005

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
            EXPERIMENTS = {
                **BASE_EXPERIMENT,
                **NOISY_EXPERIMENTS,
                **NOISIER_EXPERIMENTS,
            }
        else:
            EXPERIMENTS = BASE_EXPERIMENT

        # configure how we want to run experiments
        exp_config = ExperimentConfigs(
            run_experiments_with_added_noise=False,
            use_cached_problems=True,
            animate_trajs=False,
            run_cora=True,
            show_solver_animation=True,
            show_gt_cora_animation=True,
            look_for_cached_solns=True,
            perform_evaluation=True,
            desired_plot_modes=[plot.PlotMode.xy],
        )

        # parse the plaza data
        pyfg = read_from_pyfg_text(join(base_experiment_dir, "factor_graph.pyfg"))

        # run experiments
        run_experiments(pyfg, EXPERIMENTS, exp_config)
