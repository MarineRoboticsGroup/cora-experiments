from os.path import join
from utils.generate_manhattan_experiments import (
    MANHATTAN_EXPERIMENTS,
    generate_manhattan_experiments,
)
from utils.evaluate_manhattan_data import make_manhattan_experiment_plots
from utils.solve_manhattan_problems import solve_manhattan_problems_in_dir
from utils.paths import MANHATTAN_DATA_DIR
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
    experiments = MANHATTAN_EXPERIMENTS
    for use_loop_closure, loop_closure_subdir in [
        (True, "default_with_loop_closures"),
        (False, "default_no_loop_closures"),
    ]:
        exp_base_dir = join(MANHATTAN_DATA_DIR, loop_closure_subdir)
        generate_manhattan_experiments(
            exp_base_dir,
            default_use_loop_closures=use_loop_closure,
            experiments=experiments,
            use_cached_experiments=True,
            num_repeats_per_param=10,
        )
        solve_manhattan_problems_in_dir(
            exp_base_dir, use_cached_results=True, show_animations=True
        )
        make_manhattan_experiment_plots(
            base_experiment_dir=exp_base_dir,
            subexperiment_types=experiments,
            use_cached_sweep_results=True,
            use_cached_subexp_results=True,
        )
