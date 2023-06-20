from os.path import join
from typing import List
from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.utils.solver_utils import save_to_tum, save_results_to_file
from ra_slam.solve_mle_gtsam import solve_mle_gtsam
from ra_slam.utils.gtsam_utils import GtsamSolverParams
from ra_slam.utils.solver_utils import LM_SOLVER
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

try:
    from score.solve_score import solve_score
    from score.utils.gurobi_utils import QCQP_RELAXATION

    SCORE_AVAILABLE = True
except ImportError:
    logger.warning(
        "Could not import SCORE (https://github.com/MarineRoboticsGroup/score)"
        "Please install SCORE to get the SCORE initialized solution."
    )
    SCORE_AVAILABLE = False

GTSAM_ODOM_INIT = "gtsam_odom"
GTSAM_SCORE_INIT = "gtsam_score"
GTSAM_RANDOM_INIT = "gtsam_random"


def write_gtsam_optimized_soln_to_tum(
    fg: FactorGraphData, results_dir: str, init_strategy: str
) -> List[str]:
    """Optimize the factor graph using GTSAM and write the optimized solution to
    a TUM file.

    Args:
        fg (FactorGraphData): the factor graph data
        tum_fpath (str): the path to save the TUM file to

    Returns:
        List[str]: the list of TUM files
    """
    logger.info(f"Optimizing with GTSAM using {init_strategy}")

    if init_strategy == GTSAM_ODOM_INIT:
        init = "compose"
        landmark_init = "gt"
        gt_start = True
        custom_init_file = None
    elif init_strategy == GTSAM_SCORE_INIT:
        init = "custom"
        landmark_init = "custom"
        score_result = solve_score(fg, relaxation_type=QCQP_RELAXATION)
        custom_init_file = join(results_dir, "score_init.pickle")
        save_results_to_file(
            score_result, score_result.solver_cost, True, custom_init_file
        )
        gt_start = False
    elif init_strategy == GTSAM_RANDOM_INIT:
        init = "compose"
        landmark_init = "gt"
        gt_start = False
        custom_init_file = None
    else:
        raise ValueError(f"Invalid init strategy: {init_strategy}")

    solver_params = GtsamSolverParams(
        init_technique=init,
        landmark_init=landmark_init,
        custom_init_file=custom_init_file,
        init_translation_perturbation=None,
        init_rotation_perturbation=None,
        start_at_gt=gt_start,
    )
    gtsam_result = solve_mle_gtsam(fg, solver_params, solver=LM_SOLVER)
    # from py_factor_graph.utils.plot_utils import visualize_solution
    # visualize_solution(gtsam_result)
    tum_fpath = join(results_dir, f"{init_strategy}.tum")
    tum_files = save_to_tum(gtsam_result, tum_fpath)
    logger.info(f"Saved TUM files to {tum_files}")
    return tum_files
