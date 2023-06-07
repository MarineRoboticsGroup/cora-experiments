import os
import pickle
import copy
from typing import List, Dict, Tuple

from py_factor_graph.io.pickle_file import parse_pickle_file
from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.utils.solver_utils import save_to_tum
from ro_slam.solve_mle_gtsam import solve_mle_gtsam
from ro_slam.utils.gtsam_utils import GtsamSolverParams

import matplotlib.pyplot as plt
import pandas as pd
from evo.tools import plot, file_interface
from evo.core import metrics
from evo.core.trajectory import PoseTrajectory3D

from attrs import define, field

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


TRAJ_NAMES = ["Ground Truth", "CORA", "GTSAM"]
TRAJ_COLORS = ["gray", "blue", "red"]
PYFG_FILE_ENDING = "seed.pickle"
EXPECTED_FILE_ENDINGS = [
    PYFG_FILE_ENDING,
    "seed.mat",
    "seed_projected_iterates.gif",
    "results.mat",
    "solver_info.mat",
]
VALID_ROBOT_CHARS = [chr(ord("A") + i) for i in range(26)]
VALID_ROBOT_CHARS.remove("L")


@define
class ResultsPoseTrajCollection:
    """The results of the highbay experiment"""

    gt_traj: PoseTrajectory3D = field()
    cora_traj: PoseTrajectory3D = field()
    gtsam_odom_traj: PoseTrajectory3D = field()


def _traj_error_df_validator(instance, attribute, value):
    """Check that the traj error df is valid"""
    expected_indices = TRAJ_NAMES[1:]
    assert isinstance(value, pd.DataFrame)
    assert value.shape[0] == len(expected_indices)
    expected_cols = [
        "rmse",
        "mean",
        "median",
        "std",
        "min",
        "max",
        "sse",
    ]
    assert value.shape[1] == len(expected_cols)
    assert set(value.index) == set(expected_indices)
    assert set(value.columns) == set(expected_cols)


@define
class TrajErrorDfs:
    """pandas DataFrames representing the trajectory error statistics for a
    single experiment. All errors are w.r.t. aligned trajectories

    Sample DF
    ---------
    #               rmse      mean    median       std       min       max          sse
    # CORA      0.536588  0.502224  0.477673  0.188938  0.048637  0.923781   505.023905
    # Odometry  0.969083  0.854065  0.896745  0.457924  0.045657  2.026719  1647.219439
    # GTSAM     0.911305  0.777825  0.661960  0.474831  0.050085  1.906614  1456.655153

    """

    rot_error_df: pd.DataFrame = field(validator=_traj_error_df_validator)
    trans_error_df: pd.DataFrame = field(validator=_traj_error_df_validator)
    pose_error_df: pd.DataFrame = field(validator=_traj_error_df_validator)


ALIGNED_RESULTS_FNAME = "aligned_results.pickle"
JOINED_CORA_TUM_NAME = "joined_cora.tum"
JOINED_GT_TUM_NAME = "joined_gt.tum"
JOINED_GTSAM_ODOM_TUM_NAME = "joined_gtsam_odom.tum"
GT_TRAJ_PICKLE_NAME = "gt_traj.pickle"
CORA_TRAJ_PICKLE_NAME = "cora_traj.pickle"
GTSAM_ODOM_TRAJ_PICKLE_NAME = "gtsam_odom_traj.pickle"
RESULTS_FILES_GENERATED = [
    ALIGNED_RESULTS_FNAME,
    JOINED_CORA_TUM_NAME,
    JOINED_GT_TUM_NAME,
    JOINED_GTSAM_ODOM_TUM_NAME,
    GT_TRAJ_PICKLE_NAME,
    CORA_TRAJ_PICKLE_NAME,
    GTSAM_ODOM_TRAJ_PICKLE_NAME,
]


def _get_gt_tum_files(results_dir: str, num_robots: int) -> List[str]:
    """Get the ground truth tum files

    Args:
        results_dir (str): the directory containing the ground truth tum files
        num_robots (int): the number of robots

    Returns:
        List[str]: the list of ground truth tum files
    """
    gt_tum_files = [
        os.path.join(results_dir, f"gt_traj_{VALID_ROBOT_CHARS[i]}.tum")
        for i in range(num_robots)
    ]
    return gt_tum_files


def _get_gtsam_odom_tum_files(results_dir: str, num_robots: int) -> List[str]:
    """Get the GTSAM odometry tum files

    Args:
        results_dir (str): the directory containing the GTSAM odometry tum files
        num_robots (int): the number of robots

    Returns:
        List[str]: the list of GTSAM odometry tum files
    """
    gtsam_odom_tum_files = [
        os.path.join(results_dir, f"gtsam_odom_{VALID_ROBOT_CHARS[i]}.tum")
        for i in range(num_robots)
    ]
    return gtsam_odom_tum_files


def _get_cora_tum_files(results_dir: str, num_robots: int) -> List[str]:
    """Get the CORA tum files

    Args:
        results_dir (str): the directory containing the CORA tum files
        num_robots (int): the number of robots

    Returns:
        List[str]: the list of CORA tum files
    """
    cora_tum_paths = [
        os.path.join(results_dir, f"robot_{i+1}.tum") for i in range(num_robots)
    ]
    return cora_tum_paths


def evaluate_results(
    results_dir: str, clear_prev_files: bool = True
) -> Tuple[TrajErrorDfs, ResultsPoseTrajCollection]:
    """Performs evaluation of the generated results. Expects the following files
    in the results_dir:

    1) <results_dir>/factor_graph.pickle
    2) <results_dir>/robot_i.tum (for i = 1,...,n)

    Args:
        results_dir (str): path to the directory with the expected files to run the analysis

    """
    check_dir_ready_for_evaluation(results_dir)
    aligned_results_pickle_path = os.path.join(results_dir, ALIGNED_RESULTS_FNAME)
    if os.path.exists(aligned_results_pickle_path) and not clear_prev_files:
        print(f"Loading aligned results from {aligned_results_pickle_path}")
        aligned_results = pickle.load(open(aligned_results_pickle_path, "rb"))
    else:
        print(f"Aligning results in {results_dir}")
        pyfg_file_name = get_pyfg_file_name_in_dir(results_dir)
        fg_path = os.path.join(results_dir, pyfg_file_name)
        # print(f"Loading factor graph from {fg_path}")
        pyfg = parse_pickle_file(fg_path)
        num_robots = pyfg.num_robots

        # clear the previously generated files if requested
        if clear_prev_files:
            print(f"Clearing previously generated files in {results_dir}")
            file_paths_generated = [
                os.path.join(results_dir, fname) for fname in RESULTS_FILES_GENERATED
            ]
            file_paths_generated.extend(_get_gt_tum_files(results_dir, num_robots))
            file_paths_generated.extend(
                _get_gtsam_odom_tum_files(results_dir, num_robots)
            )
            for f in file_paths_generated:
                if os.path.exists(f):
                    os.remove(f)
                    logger.debug(f"Removed file: {f}")

        # cora tum file
        cora_tum_paths = _get_cora_tum_files(results_dir, num_robots)
        joined_cora_tum_path = os.path.join(results_dir, JOINED_CORA_TUM_NAME)
        if not os.path.exists(joined_cora_tum_path) or clear_prev_files:
            join_tum_files(cora_tum_paths, joined_cora_tum_path)
        else:
            print(f"CORA file already exists at {joined_cora_tum_path}.")

        # groundtruth tum file
        gt_tum_files = _get_gt_tum_files(results_dir, num_robots)
        if not all([os.path.exists(f) for f in gt_tum_files]) or clear_prev_files:
            gt_tum_files = pyfg.write_pose_gt_to_tum(results_dir)
        else:
            print("Ground truth files already exist.")

        joined_gt_tum_path = os.path.join(results_dir, JOINED_GT_TUM_NAME)
        if not os.path.exists(joined_gt_tum_path):
            join_tum_files(gt_tum_files, joined_gt_tum_path)
        else:
            print(f"Ground truth file already exists at {joined_gt_tum_path}.")

        # gtsam + odom init tum file
        gtsam_odom_tum_files = _get_gtsam_odom_tum_files(results_dir, num_robots)
        existing_gtsam_odom_files = [os.path.exists(f) for f in gtsam_odom_tum_files]
        if not all(existing_gtsam_odom_files) or clear_prev_files:
            print("Generating GTSAM odometry file...")
            gtsam_odom_fpath = os.path.join(results_dir, "gtsam_odom.tum")
            gtsam_odom_tum_files = write_gtsam_optimized_soln_to_tum(
                pyfg, gtsam_odom_fpath
            )
        else:
            print("GTSAM odometry files already exist.")
        joined_gtsam_odom_tum_path = os.path.join(
            results_dir, JOINED_GTSAM_ODOM_TUM_NAME
        )
        if not os.path.exists(joined_gtsam_odom_tum_path) or clear_prev_files:
            join_tum_files(gtsam_odom_tum_files, joined_gtsam_odom_tum_path)
        else:
            # print(f"Joined GTSAM odometry file already exists: {joined_gtsam_odom_tum_path}")
            pass

        # get the trajectories
        gt_traj_pickle_path = os.path.join(results_dir, GT_TRAJ_PICKLE_NAME)
        cora_traj_pickle_path = os.path.join(results_dir, CORA_TRAJ_PICKLE_NAME)
        gtsam_odom_traj_pickle_path = os.path.join(
            results_dir, GTSAM_ODOM_TRAJ_PICKLE_NAME
        )
        if os.path.exists(gt_traj_pickle_path) and not clear_prev_files:
            print(f"Loading ground truth trajectory from {gt_traj_pickle_path}")
            gt_traj = pickle.load(open(gt_traj_pickle_path, "rb"))
        else:
            print(f"Constructing ground truth trajectory from {joined_gt_tum_path}")
            gt_traj = file_interface.read_tum_trajectory_file(joined_gt_tum_path)
            pickle.dump(gt_traj, open(gt_traj_pickle_path, "wb"))

        if os.path.exists(cora_traj_pickle_path) and not clear_prev_files:
            print(f"Loading CORA trajectory from {cora_traj_pickle_path}")
            cora_traj = pickle.load(open(cora_traj_pickle_path, "rb"))
        else:
            cora_traj = file_interface.read_tum_trajectory_file(joined_cora_tum_path)
            pickle.dump(cora_traj, open(cora_traj_pickle_path, "wb"))

        if os.path.exists(gtsam_odom_traj_pickle_path) and not clear_prev_files:
            gtsam_odom_traj = pickle.load(open(gtsam_odom_traj_pickle_path, "rb"))
        else:
            gtsam_odom_traj = file_interface.read_tum_trajectory_file(
                joined_gtsam_odom_tum_path
            )
            pickle.dump(gtsam_odom_traj, open(gtsam_odom_traj_pickle_path, "wb"))

        # align the trajectories
        cora_traj_aligned = get_aligned_traj(gt_traj, cora_traj)
        gtsam_odom_traj_aligned = get_aligned_traj(gt_traj, gtsam_odom_traj)

        # group the results
        aligned_results = ResultsPoseTrajCollection(
            gt_traj=gt_traj,
            cora_traj=cora_traj_aligned,
            gtsam_odom_traj=gtsam_odom_traj_aligned,
        )
        pickle.dump(aligned_results, open(aligned_results_pickle_path, "wb"))

    ape_error_dfs = get_ape_error_stats(aligned_results)
    return ape_error_dfs, aligned_results


def make_evo_traj_plots(
    aligned_results: ResultsPoseTrajCollection,
    results_dir: str,
    show_plots: bool = False,
    valid_plot_views: List[plot.PlotMode] = [plot.PlotMode.xy],
):
    """Make plots comparing the ground truth, cora, and gtsam trajectories

    Args:
        aligned_results (ResultsPoseTrajCollection): the aligned trajectories
        results_dir (str): the directory to save the plots
        show_plots (bool, optional): whether to show the plots. Defaults to False.
    """
    gt_traj = aligned_results.gt_traj
    cora_traj_aligned = aligned_results.cora_traj
    gtsam_odom_traj_aligned = aligned_results.gtsam_odom_traj

    for plot_mode in valid_plot_views:
        fig = plt.figure(figsize=(16, 14))
        ax = plot.prepare_axis(fig, plot_mode)
        for traj, name, color in zip(
            [gt_traj, cora_traj_aligned, gtsam_odom_traj_aligned],
            TRAJ_NAMES,
            TRAJ_COLORS,
        ):
            plot.traj(ax, plot_mode, traj, "-", color, name)
        ax.legend()
        traj_plot_path = os.path.join(results_dir, f"traj_plot_{plot_mode.name}.png")
        plt.savefig(traj_plot_path)
        plt.savefig(traj_plot_path.replace(".png", ".svg"), format="svg")
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
        logger.warning(f"Saved trajectory plot to {traj_plot_path}")


def make_plots_from_error_dfs(error_dfs: TrajErrorDfs, save_dir: str):
    #               rmse      mean    median       std       min       max          sse
    # CORA      0.536588  0.502224  0.477673  0.188938  0.048637  0.923781   505.023905
    # Odometry  0.969083  0.854065  0.896745  0.457924  0.045657  2.026719  1647.219439
    # GTSAM     0.911305  0.777825  0.661960  0.474831  0.050085  1.906614  1456.655153

    df_trans = error_dfs.trans_error_df
    df_rot = error_dfs.rot_error_df
    df_pose = error_dfs.pose_error_df

    all_stats = df_trans.columns
    df_trans = df_trans.swapaxes("index", "columns")
    df_rot = df_rot.swapaxes("index", "columns")
    df_pose = df_pose.swapaxes("index", "columns")

    # drop all indices but rmse and max
    stats_to_keep = ["rmse", "max"]
    stats_to_drop = [stat for stat in all_stats if stat not in stats_to_keep]
    df_trans.drop(stats_to_drop, axis=0, inplace=True)
    df_rot.drop(stats_to_drop, axis=0, inplace=True)
    df_pose.drop(stats_to_drop, axis=0, inplace=True)

    # make bar plots
    default_figsize = (16, 8)
    fig, axs = plt.subplots(1, 3, figsize=default_figsize)
    df_trans.plot.bar(ax=axs[0], ylabel="Average Translation Error (meters)")
    df_rot.plot.bar(ax=axs[1], ylabel="Average Rotation Error (degrees)")
    df_pose.plot.bar(ax=axs[2], ylabel="Average Pose Error")
    three_bar_plot_path = os.path.join(save_dir, "three_bar_plot.png")
    plt.savefig(three_bar_plot_path)
    plt.savefig(three_bar_plot_path.replace(".png", ".svg"), format="svg")

    # also make single plots for each set of stats (trans/rot/pose)
    df_trans.plot.bar(
        ylabel="Average Translation Error (meters)", figsize=default_figsize
    )
    trans_bar_path = os.path.join(save_dir, "translation_error.png")
    plt.savefig(trans_bar_path)
    plt.savefig(trans_bar_path.replace(".png", ".svg"), format="svg")

    df_rot.plot.bar(ylabel="Average Rotation Error (degrees)", figsize=default_figsize)
    rot_bar_path = os.path.join(save_dir, "rotation_error.png")
    plt.savefig(rot_bar_path)
    plt.savefig(rot_bar_path.replace(".png", ".svg"), format="svg")

    df_pose.plot.bar(ylabel="Average Pose Error", figsize=default_figsize)
    pose_bar_path = os.path.join(save_dir, "pose_error.png")
    plt.savefig(pose_bar_path)
    plt.savefig(pose_bar_path.replace(".png", ".svg"), format="svg")

    print("Saved plots to: ", save_dir)
    # print the file paths
    for f in [three_bar_plot_path, trans_bar_path, rot_bar_path, pose_bar_path]:
        print(f"File: {f}")


def make_evo_ape_plots_from_trajs(
    aligned_results: ResultsPoseTrajCollection, results_dir: str
):
    collected_error_dfs = get_ape_error_stats(aligned_results)
    make_plots_from_error_dfs(collected_error_dfs, results_dir)
    return collected_error_dfs


def get_ape_error_stats(aligned_results: ResultsPoseTrajCollection) -> TrajErrorDfs:
    gt_traj = aligned_results.gt_traj
    cora_traj_aligned = aligned_results.cora_traj
    gtsam_odom_traj_aligned = aligned_results.gtsam_odom_traj

    comparison_trajs = [cora_traj_aligned, gtsam_odom_traj_aligned]

    # get translation error
    translation_errors: List[Dict[str, float]] = []
    for traj in comparison_trajs:
        translation_metric = metrics.APE(metrics.PoseRelation.translation_part)
        translation_metric.process_data((gt_traj, traj))
        # translation_stat = translation_metric.get_statistic(metrics.StatisticsType.rmse)
        translation_stats = translation_metric.get_all_statistics()
        translation_errors.append(translation_stats)

    # get rotation error
    rotation_errors: List[Dict[str, float]] = []
    for traj in comparison_trajs:
        rotation_metric = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
        rotation_metric.process_data((gt_traj, traj))
        # rotation_stat =
        # rotation_metric.get_statistic(metrics.StatisticsType.rmse)
        rotation_stats = rotation_metric.get_all_statistics()
        rotation_errors.append(rotation_stats)

    # get APE stats
    pose_errors: List[Dict[str, float]] = []
    for traj in comparison_trajs:
        pose_metric = metrics.APE(metrics.PoseRelation.full_transformation)
        pose_metric.process_data((gt_traj, traj))
        # pose_stat = pose_metric.get_statistic(metrics.StatisticsType.rmse)
        pose_stats = pose_metric.get_all_statistics()
        pose_errors.append(pose_stats)

    df_trans = pd.DataFrame(
        translation_errors,
        index=TRAJ_NAMES[1:],
    )
    df_rot = pd.DataFrame(
        rotation_errors,
        index=TRAJ_NAMES[1:],
    )
    df_pose = pd.DataFrame(
        pose_errors,
        index=TRAJ_NAMES[1:],
    )

    collected_error_dfs = TrajErrorDfs(
        rot_error_df=df_rot,
        trans_error_df=df_trans,
        pose_error_df=df_pose,
    )
    return collected_error_dfs


def get_aligned_traj(ref_traj: PoseTrajectory3D, traj: PoseTrajectory3D):
    traj_aligned = copy.deepcopy(traj)
    traj_aligned.align(ref_traj, correct_scale=False)
    return traj_aligned


def join_tum_files(tum_file_paths: List[str], save_path: str) -> None:
    """Joins multiple TUM files into one file

    Args:
        tum_file_paths (List[str]): list of paths to TUM files
        save_path (str): path to save the joined TUM file
    """
    with open(save_path, "w") as f:
        for tum_file_path in tum_file_paths:
            with open(tum_file_path, "r") as f_tum:
                for line in f_tum:
                    f.write(line)
    print(f"Saved joined TUM file to: {save_path}")


def get_pyfg_file_name_in_dir(target_dir: str) -> str:
    file_names = os.listdir(target_dir)
    candidate_pyfg_files = [
        f
        for f in file_names
        if f.endswith(PYFG_FILE_ENDING) or f == "factor_graph.pickle"
    ]
    no_pyfg_files_found = len(candidate_pyfg_files) == 0
    if no_pyfg_files_found:
        raise FileNotFoundError(
            f"Could not find PyFG file in {target_dir} - existing files: {file_names}"
        )
    elif len(candidate_pyfg_files) > 1:
        raise FileNotFoundError(
            f"Multiple candidate PyFG files in {target_dir} - existing files: {file_names}"
        )

    return candidate_pyfg_files[0]


def check_dir_ready_for_evaluation(target_dir: str) -> None:
    """Raises errors if the right files cannot be found to run analysis

    Args:
        target_dir (str): the directory containing the right files

    Raises:
        FileNotFoundError: couldn't find factor_graph.pickle
        FileNotFoundError: couldn't find

    Returns:
        _type_: _description_
    """
    pyfg_file_name = get_pyfg_file_name_in_dir(target_dir)
    pyfg_file = os.path.join(target_dir, pyfg_file_name)
    pyfg = parse_pickle_file(pyfg_file)
    num_robots = pyfg.num_robots

    file_names = os.listdir(target_dir)
    for i in range(num_robots):
        expected_tum_file = f"robot_{i+1}.tum"
        if expected_tum_file not in file_names:
            raise FileNotFoundError(
                f"Could not find {expected_tum_file} in {target_dir}"
            )


def get_leaf_dirs(root_dir: str) -> List[str]:
    """Recursively finds all of the leaf directories under the root directory.

    Args:
        root_dir (str): the root directory

    Returns:
        List[str]: the list of leaf directories
    """
    leaf_dirs = []
    assert os.path.isdir(root_dir)
    # print(f"root_dir: {root_dir} contains the following files: {os.listdir(root_dir)}")
    for root, dirs, files in os.walk(root_dir):
        if len(dirs) == 0:
            leaf_dirs.append(root)
    return leaf_dirs


def dir_has_been_organized(target_dir: str) -> bool:
    file_names = os.listdir(target_dir)

    # check that we have the right number of files expected
    num_files_expected = len(EXPECTED_FILE_ENDINGS)
    if not len(file_names) == num_files_expected:
        print(
            f"The directory {target_dir} isn't fully organized. We found {len(file_names)} files but expected {num_files_expected}"
        )
        print(f"Files found: {file_names}")
        return False

    # check that has one file with the right file ending
    for f_ending in EXPECTED_FILE_ENDINGS:
        files_with_ending = [f for f in file_names if f.endswith(f_ending)]
        num_files_with_ending = len(files_with_ending)
        if num_files_with_ending != 1:
            print(
                f"The directory {target_dir} isn't fully organized. We found {files_with_ending} corresponding to the ending {f_ending}"
            )
            return False

    return True


def write_gtsam_optimized_soln_to_tum(fg: FactorGraphData, tum_fpath: str) -> List[str]:
    """Optimize the factor graph using GTSAM and write the optimized solution to
    a TUM file.

    Args:
        fg (FactorGraphData): the factor graph data
        tum_fpath (str): the path to save the TUM file to

    Returns:
        List[str]: the list of TUM files
    """

    solver_params = GtsamSolverParams(
        init_technique="compose",
        landmark_init="gt",
        custom_init_file=None,
        init_translation_perturbation=None,
        init_rotation_perturbation=None,
        start_at_gt=True,
    )
    gtsam_result = solve_mle_gtsam(fg, solver_params)
    tum_files = save_to_tum(gtsam_result, tum_fpath)
    return tum_files
