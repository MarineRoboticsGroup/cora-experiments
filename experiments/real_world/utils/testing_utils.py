from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.utils.solver_utils import (
    VariableValues,
    SolverResults,
)
from py_factor_graph.utils.matrix_utils import apply_transformation_matrix_perturbation
import argparse
from ro_slam.solve_mle_gtsam import solve_mle_gtsam
from ro_slam.utils.gtsam_utils import get_cost_at_variable_values, GtsamSolverParams

from os.path import join
import os

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

# import SCORE if using python3.6
try:
    from score.solve_score import solve_mle_qcqp
    from score.utils.solver_utils import QcqpSolverParams
except ImportError:
    import sys

    python_version = sys.version_info
    logger.warning(
        f"SCORE not installed - currently only tested for Python 3.6.x "
        f"Currently using python {python_version.major}.{python_version.minor}.{python_version.micro}"
    )


def get_cost_at_values(
    factor_graph: FactorGraphData,
    variable_values: VariableValues,
) -> float:
    """Gets the cost at the given variable values.

    Args:
        factor_graph (FactorGraphData): the factor graph data.
        variable_values (VariableValues): the variable values.
        solver_params (GtsamSolverParams): the solver parameters.

    Returns:
        float: the cost at the given variable values.
    """
    return get_cost_at_variable_values(factor_graph, variable_values)


def run_odom_gtsam(
    data: FactorGraphData, args: argparse.Namespace, new_problem: bool = True
) -> str:
    """Runs GTSAM with odometry initialization.

    Args:
        data (FactorGraphData): the factor graph data.
        args (argparse.Namespace): the command line arguments.
    """
    odom_gtsam_result_file = "odom_gtsam_result.pickle"
    gtsam_params = GtsamSolverParams(
        verbose=True, save_results=True, init_technique="compose"
    )
    result_filepath = join(args.results_dir, odom_gtsam_result_file)
    if new_problem:
        _ = solve_mle_gtsam(data, gtsam_params, result_filepath)
    else:
        logger.warning(f"Not running solver, using existing result file.")

    return result_filepath


def run_random_gtsam(
    data: FactorGraphData, args: argparse.Namespace, new_problem: bool = True
) -> str:
    """Runs GTSAM with random initialization.

    Args:
        args (argparse.Namespace): the command line arguments.
    """
    gtsam_params = GtsamSolverParams(
        verbose=True, save_results=True, init_technique="random"
    )

    random_gtsam_result_file = "random_gtsam_result.pickle"
    result_filepath = join(args.results_dir, random_gtsam_result_file)

    if new_problem:
        _ = solve_mle_gtsam(data, gtsam_params, result_filepath)
    else:
        logger.warning(f"Not running solver, using existing result file.")

    return result_filepath


def run_gt_gtsam(
    data: FactorGraphData, args: argparse.Namespace, new_problem: bool = True
) -> str:
    """Runs GTSAM with groundtruth initialization.

    Args:
        args (argparse.Namespace): the command line arguments.
    """
    gtsam_params = GtsamSolverParams(
        verbose=True, save_results=True, init_technique="gt"
    )

    gt_gtsam_result_file = "gt_gtsam_result.pickle"
    result_filepath = join(args.results_dir, gt_gtsam_result_file)

    if new_problem:
        solve_mle_gtsam(data, gtsam_params, result_filepath)
        print()
    else:
        logger.warning(f"Not running solver, using existing result file.")

    return result_filepath


def run_score(
    data: FactorGraphData, args: argparse.Namespace, new_problem: bool = True
) -> str:
    """Runs the score algorithm.

    Args:
        data (FactorGraphData): the factor graph data.
        args (argparse.Namespace): the command line arguments.
    """
    score_result_file = "score_result.pickle"
    qcqp_params = QcqpSolverParams(solver="gurobi", verbose=True, save_results=True)
    result_filepath = join(args.results_dir, score_result_file)
    if new_problem:
        solve_mle_qcqp(data, qcqp_params, result_filepath)
        print()
    else:
        logger.warning(f"Not running solver, using existing result file.")

    return result_filepath


def run_score_gtsam(
    data: FactorGraphData, args: argparse.Namespace, new_problem: bool = True
) -> str:
    """Runs GTSAM with odometry initialization.

    Args:
        data (FactorGraphData): the factor graph data.
        args (argparse.Namespace): the command line arguments.
    """
    score_result_filepath = run_score(data, args)
    score_gtsam_result_file = "score_gtsam_result.pickle"
    gtsam_params = GtsamSolverParams(
        verbose=True,
        save_results=True,
        init_technique="custom",
        custom_init_file=score_result_filepath,
    )
    result_filepath = join(args.results_dir, score_gtsam_result_file)

    if new_problem:
        solve_mle_gtsam(data, gtsam_params, result_filepath)
        print()
    else:
        logger.warning(f"Not running solver, using existing result file.")

    return result_filepath


def clear_results(args: argparse.Namespace) -> None:
    """Removes all results files from the results directory.

    Args:
        args (argparse.Namespace): command line arguments.
    """

    print("Clearing previous results")
    for f in os.listdir(args.results_dir):
        os.remove(os.path.join(args.results_dir, f))


def perturb_solution(solution: SolverResults) -> SolverResults:
    """Perturbs the solution by shifting the translations on the poses

    Args:
        solution (SolverResults): the solution.

    Returns:
        SolverResults: the perturbed solution.
    """
    solution_poses = solution.variables.poses
    perturbed_solution_poses = {}
    for key, pose in solution_poses.items():
        new_pose = apply_transformation_matrix_perturbation(
            pose, perturb_magnitude=0.3, perturb_rotation=0.1
        )
        perturbed_solution_poses[key] = new_pose
    perturbed_solution = SolverResults(
        variables=VariableValues(
            dim=solution.variables.dim,
            poses=perturbed_solution_poses,
            landmarks=solution.variables.landmarks,
            distances=solution.variables.distances,
        ),
        total_time=solution.total_time,
        solved=solution.solved,
        pose_chain_names=solution.pose_chain_names,
        solver_cost=solution.solver_cost,
    )
    return perturbed_solution
