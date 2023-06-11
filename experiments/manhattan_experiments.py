from utils.generate_manhattan_experiments import (
    MANHATTAN_EXPERIMENTS,
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
    # generate_manhattan_experiments(MANHATTAN_DATA_DIR, experiments=experiments, use_cached_experiments=True)
    solve_manhattan_problems_in_dir(
        MANHATTAN_DATA_DIR, use_cached_results=True, show_animations=False
    )
    make_manhattan_experiment_plots(
        base_experiment_dir=MANHATTAN_DATA_DIR, subexperiment_types=experiments
    )
