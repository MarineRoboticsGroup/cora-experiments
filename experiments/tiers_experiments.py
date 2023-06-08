import numpy as np
from py_factor_graph.io.pyfg_text import read_from_pyfg_text
from os.path import dirname, abspath, join
from evo.tools import plot

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
        generate_problems=True,
        animate_trajs=False,
        run_cora=True,
        show_gt_cora_animation=True,
        look_for_cached_solns=True,
        perform_evaluation=True,
        desired_plot_modes=[plot.PlotMode.xy],
    )

    # run experiments
    run_experiments(pyfg, EXPERIMENTS, exp_config)
