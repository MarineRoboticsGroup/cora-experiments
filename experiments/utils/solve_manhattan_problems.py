from .run_cora_utils import run_cora
from .paths import MANHATTAN_DATA_DIR, get_leaf_dirs


def solve_manhattan_problems_in_dir(
    base_manhattan_data_dir: str = MANHATTAN_DATA_DIR,
    use_cached_results: bool = False,
    show_animations: bool = True,
):
    manhattan_leaf_dirs = get_leaf_dirs(base_manhattan_data_dir)
    for problem_dir in manhattan_leaf_dirs:
        run_cora(
            experiment_dir=problem_dir,
            show_animation=show_animations,
            animation_show_gt=True,
            look_for_cached_cora_solns=use_cached_results,
        )
