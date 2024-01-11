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
        use_cached_problems=False,
        animate_trajs=True,
        run_cora=True,
        solve_marginalized_problem=False,
        show_solver_animation=True,
        show_gt_cora_animation=True,
        look_for_cached_cora_solns=False,
        perform_evaluation=True,
        use_cached_trajs=False,
        desired_plot_modes=[plot.PlotMode.xy],
    )

    # parse the data
    MRCLAM_DIR = join(DATA_DIR, "mrclam")
    MRCLAM_RANGE_ONLY_DIR = join(MRCLAM_DIR, "range_only")
    MRCLAM_RANGE_AND_RPM_DIR = join(MRCLAM_DIR, "range_and_rpm")
    mrclam_experiments = [
        "mrclam2",
        "mrclam3a",
        "mrclam3b",
        "mrclam4",
        "mrclam5a",
        "mrclam5b",
        "mrclam5c",
        "mrclam6",
        "mrclam7",
    ]
    for main_mrclam_data_dir in [MRCLAM_RANGE_AND_RPM_DIR]:
        for mrclam_exp in mrclam_experiments:
            data_dir = join(main_mrclam_data_dir, mrclam_exp)
            pyfg_fpath = join(data_dir, f"{mrclam_exp}.pyfg")
            pyfg = read_from_pyfg_text(join(data_dir, pyfg_fpath))
            experiments = get_all_experiments_to_run(data_dir, exp_config)
            run_experiments(pyfg, experiments, exp_config)
