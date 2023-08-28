from os.path import join
from evo.tools import plot
from py_factor_graph.io.plaza_experiments import parse_plaza_files
from py_factor_graph.modifiers import skip_first_n_poses

from utils.run_experiments import (
    run_experiments,
    ExperimentConfigs,
    get_all_experiments_to_run,
)
from utils.paths import DATA_DIR

if __name__ == "__main__":
    exp_config = ExperimentConfigs(
        run_experiments_with_added_noise=True,
        use_cached_problems=False,
        animate_trajs=True,
        run_cora=True,
        solve_marginalized_problem=False,
        show_solver_animation=False,
        show_gt_cora_animation=True,
        look_for_cached_cora_solns=True,
        perform_evaluation=True,
        use_cached_trajs=False,
        desired_plot_modes=[plot.PlotMode.xy],
    )

    PLAZA_ORIGINAL_DATA_DIRS = {
        "plaza1": "/home/alan/experimental_data/plaza/Plaza1",
        "plaza2": "/home/alan/experimental_data/plaza/Plaza2",
    }

    PLAZA_DIR = join(DATA_DIR, "plaza")
    # PLAZA_EXPERIMENTS = ["plaza1", "plaza2"]
    # PLAZA_EXPERIMENTS = ["plaza1"]
    PLAZA_EXPERIMENTS = ["plaza2"]
    PLAZA_EXPERIMENT_DIRS = [join(PLAZA_DIR, exp) for exp in PLAZA_EXPERIMENTS]

    for base_experiment_dir in PLAZA_EXPERIMENT_DIRS:
        if "plaza1" in base_experiment_dir:
            orig_data_dir = PLAZA_ORIGINAL_DATA_DIRS["plaza1"]
        elif "plaza2" in base_experiment_dir:
            orig_data_dir = PLAZA_ORIGINAL_DATA_DIRS["plaza2"]
        else:
            raise ValueError(f"Unknown experiment dir: {base_experiment_dir}")

        # configure how we want to run experiments
        # parse the plaza data
        pyfg = parse_plaza_files(orig_data_dir)
        if "plaza2" in base_experiment_dir:
            # GT data is a little spurious for first 225 poses
            pyfg = skip_first_n_poses(pyfg, 225)

        # get all the experiments to run and run them
        experiments = get_all_experiments_to_run(base_experiment_dir, exp_config)
        run_experiments(pyfg, experiments, exp_config)
