from os.path import join
from typing import List, Optional
from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.utils.solver_utils import save_to_tum, save_results_to_file
from ra_slam.solve_mle_gtsam import solve_mle_gtsam
from ra_slam.utils.gtsam_utils import GtsamSolverParams
from ra_slam.utils.solver_utils import LM_SOLVER
from .logging_utils import logger

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

GTSAM_SCORE_INIT = "gtsam_score"
GTSAM_GT_POSE_GT_LAND = "gtsam_gt_pose_gt_landmarks"
GTSAM_GT_POSE_RAND_LAND = "gtsam_gt_pose_random_landmarks"
GTSAM_RAND_POSE_RAND_LAND = "gtsam_random_pose_random_landmarks"
GTSAM_RAND_POSE_GT_LAND = "gtsam_random_pose_gt_landmarks"


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
    logger.debug(f"Optimizing with GTSAM using {init_strategy}")

    init = "compose"
    custom_init_file = None
    if init_strategy == GTSAM_SCORE_INIT:
        init = "custom"
        gt_start = False
        landmark_init = "custom"
        score_result = solve_score(fg, relaxation_type=QCQP_RELAXATION)
        custom_init_file = join(results_dir, "score_init.pickle")
        save_results_to_file(
            score_result, score_result.solver_cost, True, custom_init_file
        )
    elif init_strategy == GTSAM_GT_POSE_GT_LAND:
        landmark_init = "gt"
        gt_start = True
    elif init_strategy == GTSAM_GT_POSE_RAND_LAND:
        gt_start = True
        landmark_init = "random"
    elif init_strategy == GTSAM_RAND_POSE_RAND_LAND:
        gt_start = False
        landmark_init = "random"
    elif init_strategy == GTSAM_RAND_POSE_GT_LAND:
        gt_start = False
        landmark_init = "gt"
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

    # for some experiments, we had to use a fixed random seed just to get a result
    # that could be aligned via Umeyama without requiring a flip
    rand_seed: Optional[int] = None
    if "drone" in results_dir:
        rand_seed = 999
    elif "plaza1" in results_dir:
        rand_seed = 42 * 999999
    elif "tiers" in results_dir:
        if init_strategy == GTSAM_RAND_POSE_GT_LAND:
            rand_seed = 999
        elif init_strategy == GTSAM_RAND_POSE_RAND_LAND:
            rand_seed = 999 * 42
        else:
            rand_seed = None
    else:
        rand_seed = None

    if rand_seed is not None:
        gtsam_result = solve_mle_gtsam(
            fg, solver_params, solver=LM_SOLVER, seed=rand_seed
        )
    else:
        gtsam_result = solve_mle_gtsam(fg, solver_params, solver=LM_SOLVER)

    tum_fpath = join(results_dir, f"{init_strategy}.tum")
    tum_files = save_to_tum(gtsam_result, tum_fpath)
    logger.debug(f"Saved TUM files to {tum_files}")
    return tum_files
