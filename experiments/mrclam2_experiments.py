from py_factor_graph.io.pyfg_text import read_from_pyfg_text
from os.path import join
from evo.tools import plot

from utils.run_experiments import (
    run_experiments,
    ExperimentConfigs,
    get_all_experiments_to_run,
)
from utils.paths import DATA_DIR

if __name__ == "__main__":
    # configure how we want to run experiments
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
    )

    # parse the MRCLAM data
    fname = "mrclam6"
    for fname in [
        "mrclam2",
        "mrclam4",
        "mrclam6",
        "mrclam7",
    ]:
        MRCLAM_EXPERIMENT_DIR = join(DATA_DIR, fname)
        pyfg = read_from_pyfg_text(join(MRCLAM_EXPERIMENT_DIR, f"factor_graph.pyfg"))

        # run the experiments
        experiments = get_all_experiments_to_run(MRCLAM_EXPERIMENT_DIR, exp_config)
        run_experiments(pyfg, experiments, exp_config)
