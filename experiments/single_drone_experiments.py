from os.path import join
from evo.tools import plot
from py_factor_graph.io.pyfg_text import read_from_pyfg_text

from utils.run_experiments import (
    run_experiments,
    ExperimentConfigs,
    get_all_experiments_to_run,
)
from utils.paths import DATA_DIR

if __name__ == "__main__":
    exp_config = ExperimentConfigs(
        run_experiments_with_added_noise=False,
        use_cached_problems=True,
        animate_trajs=False,
        run_cora=True,
        solve_marginalized_problem=True,
        show_solver_animation=False,
        show_gt_cora_animation=True,
        look_for_cached_cora_solns=False,
        perform_evaluation=True,
        use_cached_trajs=True,
        desired_plot_modes=[plot.PlotMode.xy, plot.PlotMode.xyz, plot.PlotMode.xz],
    )

    # parse the single_drone data
    SINGLE_DRONE_DIR = join(DATA_DIR, "single_drone")
    experiment_file = join(SINGLE_DRONE_DIR, "single_drone.pyfg")
    pyfg = read_from_pyfg_text(experiment_file)

    # run the experiments
    experiments = get_all_experiments_to_run(SINGLE_DRONE_DIR, exp_config)
    run_experiments(pyfg, experiments, exp_config)
