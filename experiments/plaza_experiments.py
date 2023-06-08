import numpy as np
from os.path import dirname, abspath, join
from evo.tools import plot
from py_factor_graph.io.plaza_experiments import parse_plaza_files

import sys

sys.path.insert(0, dirname(abspath(__file__)))
from utils.run_experiments import run_experiments, ExperimentConfigs

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

    THIS_DIR = dirname(abspath(__file__))
    DATA_DIR = join(dirname(dirname(THIS_DIR)), "data")
    PLAZA_DIR = join(DATA_DIR, "plaza")
    PLAZA_EXPERIMENTS = ["Plaza1", "Plaza2"]
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
            generate_problems=False,
            animate_trajs=False,
            run_cora=True,
            show_gt_cora_animation=True,
            look_for_cached_solns=True,
            perform_evaluation=False,
            desired_plot_modes=[plot.PlotMode.xy],
        )

        # parse the plaza data
        pyfg = parse_plaza_files(base_experiment_dir)

        # run experiments
        run_experiments(pyfg, EXPERIMENTS, exp_config)
