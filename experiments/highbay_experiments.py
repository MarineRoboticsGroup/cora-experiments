import numpy as np
from os.path import join
from evo.tools import plot
from py_factor_graph.io.pyfg_text import read_from_pyfg_text

from .utils.run_experiments import run_experiments, ExperimentConfigs
from .utils.paths import DATA_DIR


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

    # the data directory is two levels up from this file
    HIGHBAY_DIR = join(DATA_DIR, "hexcopter_highbay")

    BASE_EXPERIMENT = {
        HIGHBAY_DIR: np.array([0.0, 0.0]),
    }

    if run_noisy_experiments:
        modified_dir = join(HIGHBAY_DIR, "modified")

        # make noisy version of problem
        trans_stddev = 0.01
        rot_stddev = 0.005

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

    # parse the highbay data
    experiment_file = join(HIGHBAY_DIR, "hexcopter_highbay.pyfg")
    pyfg = read_from_pyfg_text(experiment_file)

    # run experiments
    run_experiments(pyfg, EXPERIMENTS, exp_config)
