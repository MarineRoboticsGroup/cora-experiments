from typing import List, Tuple

from os.path import dirname, abspath, expanduser, join, isfile, isdir
import os
import sys
import pickle


sys.path.insert(0, dirname(dirname(abspath(__file__))))
from evaluate_utils import (
    evaluate_results,
    get_leaf_dirs,
    TrajErrorDfs,
)

import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
import seaborn as sns
import pandas as pd


def collect_error_dfs(
    ape_error_df_collection: List[TrajErrorDfs],
) -> pd.DataFrame:
    """Collect the separate runs (each held in an error dataframe) into a single
    dataframe. A single run is represented by a dataframe that looks like:

    #               rmse      mean    median       std       min       max          sse
    # CORA      0.536588  0.502224  0.477673  0.188938  0.048637  0.923781   505.023905
    # GTSAM     0.911305  0.777825  0.661960  0.474831  0.050085  1.906614  1456.655153

    and we want to show the bulk statistics for each index (CORA, GTSAM)

    Args:
        ape_error_df_collection (List[TrajErrorDfs]): _description_

    Returns:
        pd.Dataframe: _description_
    """
    rot_error_dfs = [df.rot_error_df for df in ape_error_df_collection]
    trans_error_dfs = [df.trans_error_df for df in ape_error_df_collection]
    pose_error_dfs = [df.pose_error_df for df in ape_error_df_collection]

    # collect each group into a single dataframe
    stacked_rot_error_dfs = pd.concat(rot_error_dfs, axis=0)
    stacked_trans_error_dfs = pd.concat(trans_error_dfs, axis=0)
    stacked_pose_error_dfs = pd.concat(pose_error_dfs, axis=0)

    # perform a lot of operations to get the data into the right format
    cols_to_keep = ["rmse", "max"]

    def _get_as_rmse_and_max_dfs_with_indices_diff_experiments(
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # swap the index and columns and group by the index
        df = df[cols_to_keep]
        df = df.swapaxes("index", "columns")

        # split the indices into two separate dataframes
        df_rmse = df.loc["rmse"]
        df_max = df.loc["max"]

        # group by the indices and make into separate columns stacked on top of each other
        df_rmse = (
            df_rmse.groupby(df_rmse.index)
            .agg(list)
            .apply(pd.Series)
            .swapaxes("index", "columns")
        )
        df_max = (
            df_max.groupby(df_max.index)
            .agg(list)
            .apply(pd.Series)
            .swapaxes("index", "columns")
        )

        return df_rmse, df_max

    rot_rmse_df, rot_max_df = _get_as_rmse_and_max_dfs_with_indices_diff_experiments(
        stacked_rot_error_dfs
    )
    (
        trans_rmse_df,
        trans_max_df,
    ) = _get_as_rmse_and_max_dfs_with_indices_diff_experiments(stacked_trans_error_dfs)
    pose_rmse_df, pose_max_df = _get_as_rmse_and_max_dfs_with_indices_diff_experiments(
        stacked_pose_error_dfs
    )

    # add the separate dataframes together, grouped under "RMSE" and "Max"
    groupings = ["RMSE", "Max"]
    collected_rot_errors_df = pd.concat(
        [rot_rmse_df, rot_max_df], axis=1, keys=groupings
    )
    collected_trans_errors_df = pd.concat(
        [trans_rmse_df, trans_max_df], axis=1, keys=groupings
    )
    collected_pose_errors_df = pd.concat(
        [pose_rmse_df, pose_max_df], axis=1, keys=groupings
    )

    # add the separate dataframes together, grouped under "Rotation",
    # "Translation", and "Pose"
    all_errors_collected_df = pd.concat(
        [collected_rot_errors_df, collected_trans_errors_df, collected_pose_errors_df],
        axis=1,
        keys=["Rotation", "Translation", "Pose"],
    )

    return all_errors_collected_df


def make_box_and_whisker_error_plots(
    ape_error_df_collection: List[TrajErrorDfs],
    save_dir: str,
    show_plots: bool = False,
):
    collected_dfs = collect_error_dfs(ape_error_df_collection)

    """ there should be 3 major groupings: Rotation, Translation, Pose
        each of these groupings should have 2 subgroups: RMSE, Max
        each of these subgroups should have 2 columns: CORA, GTSAM
        for each major grouping, we want to make a series of box and
        whisker plots comparing the two columns in each subgroup
    """

    def _get_plot_label(group: str, subgroup: str) -> str:
        group_options = ["Rotation", "Translation", "Pose"]
        subgroup_options = ["RMSE", "Max"]
        if group not in group_options:
            raise ValueError(f"Invalid group: {group}")
        if subgroup not in subgroup_options:
            raise ValueError(f"Invalid subgroup: {subgroup}")

        if group == "Rotation" and subgroup == "RMSE":
            return "Rotation RMSE (degrees)"
        elif group == "Rotation" and subgroup == "Max":
            return "Rotation Max Error (degrees)"
        elif group == "Translation" and subgroup == "RMSE":
            return "Translation RMSE (meters)"
        elif group == "Translation" and subgroup == "Max":
            return "Translation Max Error (meters)"
        elif group == "Pose" and subgroup == "RMSE":
            return "Absolute Pose Error RMSE"
        elif group == "Pose" and subgroup == "Max":
            return "Max Absolute Pose Error"
        else:
            raise ValueError(f"Invalid group: {group}, subgroup: {subgroup}")

    major_groupings = collected_dfs.columns.get_level_values(0).unique()
    subgroups = collected_dfs.columns.get_level_values(1).unique()
    # main_fig, main_ax = plt.subplots(1, 6, figsize=(18, 12))
    # main_fig_cnt = 0
    for group in major_groupings:
        assert isinstance(group, str)
        fig, ax = plt.subplots(1, 2, figsize=(18, 12))
        for i, subgroup in enumerate(subgroups):
            df = collected_dfs[group][subgroup]

            # plot the individual comparison
            sns.boxplot(data=df, ax=ax[i])
            ax[i].set_xlabel(_get_plot_label(group, subgroup), fontsize=16, labelpad=10)

        save_path = join(save_dir, f"{group.lower()}_error_box_and_whisker_plot.png")
        plt.savefig(save_path)
        plt.savefig(save_path.replace(".png", ".svg"), format="svg")
        print(f"Saved plot to {save_path}")
        if show_plots:
            plt.show()

        # close the individual figure
        plt.close(fig)
    # plt.show()


def _get_num_poses(dir_path: str) -> int:
    """Get the number of poses from the directory name"""
    # /home/alan/data/manhattan/cert/no_loop_closures/sweep_num_poses/factor_graph_4robots_0.5rangeStddev_4500poses_500ranges_0loopClosures_29997seed...
    assert "sweep_num_poses" in dir_path, f"Trying to get num poses from {dir_path}"
    leading_str = "rangeStddev_"
    trailing_str = "poses_"
    num_poses = int(
        dir_path[
            dir_path.index(leading_str)
            + len(leading_str) : dir_path.index(trailing_str)
        ]
    )
    return num_poses


def _get_num_robots(dir_path: str) -> int:
    assert "sweep_num_robots" in dir_path, f"Trying to get num robots from {dir_path}"
    leading_str = "factor_graph_"
    trailing_str = "robots_"
    num_robots = int(
        dir_path[
            dir_path.index(leading_str)
            + len(leading_str) : dir_path.index(trailing_str)
        ]
    )
    return num_robots


def _get_num_ranges(dir_path: str) -> int:
    assert "sweep_num_ranges" in dir_path, f"Trying to get num ranges from {dir_path}"
    leading_str = "poses_"
    trailing_str = "ranges_"
    num_ranges = int(
        dir_path[
            dir_path.index(leading_str)
            + len(leading_str) : dir_path.index(trailing_str)
        ]
    )
    return num_ranges


def _get_range_stddev(dir_path: str) -> float:
    assert "sweep_range_cov" in dir_path, f"Trying to get range stddev from {dir_path}"
    leading_str = "robots_"
    trailing_str = "rangeStddev_"
    range_stddev = float(
        dir_path[
            dir_path.index(leading_str)
            + len(leading_str) : dir_path.index(trailing_str)
        ]
    )
    return range_stddev


def _get_param(exp_type: str, dir_path: str):
    if exp_type == "sweep_num_poses":
        return _get_num_poses(dir_path)
    elif exp_type == "sweep_num_robots":
        return _get_num_robots(dir_path)
    elif exp_type == "sweep_num_ranges":
        return _get_num_ranges(dir_path)
    elif exp_type == "sweep_range_cov":
        return _get_range_stddev(dir_path)
    else:
        raise ValueError(f"Invalid exp_type: {exp_type}")


def _sort_according_to_params(
    params: List,
    rot_error_dfs: List[pd.DataFrame],
    pose_error_dfs: List[pd.DataFrame],
):
    """Sort the error dfs according to the params"""
    # sort the params in increasing order and sort the error dfs accordingly
    sorted_params = sorted(params)
    sorted_rot_error_dfs = []
    sorted_pose_error_dfs = []
    for param in sorted_params:
        idx = params.index(param)
        sorted_rot_error_dfs.append(rot_error_dfs[idx])
        sorted_pose_error_dfs.append(pose_error_dfs[idx])
    return sorted_params, sorted_rot_error_dfs, sorted_pose_error_dfs


def make_error_plots_vs_params(
    params: List,
    rot_error_dfs: List[pd.DataFrame],
    pose_error_dfs: List[pd.DataFrame],
    param_sweep_type: str,
    save_dir: str,
    show_plots: bool = False,
):
    #            rmse      mean    median       std       min       max         sse
    # CORA   0.371115  0.331680  0.309012  0.166476  0.012585  0.923779  275.452125
    # GTSAM  0.512161  0.456579  0.424164  0.232044  0.018938  1.244727  524.617574

    if not isdir(save_dir):
        os.makedirs(save_dir)

    # if we are sweeping covariance we need to square the values, which are
    # actually the std devs
    if param_sweep_type == "sweep_range_cov":
        params = [p**2 for p in params]

    # sort the params in increasing order and sort the error dfs accordingly
    params, rot_error_dfs, pose_error_dfs = _sort_according_to_params(
        params, rot_error_dfs, pose_error_dfs
    )

    gtsam_color = "red"
    cora_color = "blue"

    exp_params = {
        "sweep_num_poses": r"\textbf{\# poses}",
        "sweep_num_ranges": r"\textbf{\# range measurements}",
        "sweep_num_robots": r"\textbf{\# robots}",
        "sweep_range_cov": r"\textbf{$\sigma_{ij}^2$}",
    }

    # we want rmse and max
    cora_rot_rmse_errors = []
    cora_rot_max_errors = []
    gtsam_rot_rmse_errors = []
    gtsam_rot_max_errors = []
    for rot_error_df in rot_error_dfs:
        cora_rot_rmse_errors.append(rot_error_df.loc["CORA", "rmse"])
        cora_rot_max_errors.append(rot_error_df.loc["CORA", "max"])
        gtsam_rot_rmse_errors.append(rot_error_df.loc["GTSAM", "rmse"])
        gtsam_rot_max_errors.append(rot_error_df.loc["GTSAM", "max"])

    cora_pose_rmse_errors = []
    cora_pose_max_errors = []
    gtsam_pose_rmse_errors = []
    gtsam_pose_max_errors = []
    for pose_error_df in pose_error_dfs:
        cora_pose_rmse_errors.append(pose_error_df.loc["CORA", "rmse"])
        cora_pose_max_errors.append(pose_error_df.loc["CORA", "max"])
        gtsam_pose_rmse_errors.append(pose_error_df.loc["GTSAM", "rmse"])
        gtsam_pose_max_errors.append(pose_error_df.loc["GTSAM", "max"])

    # plot rotation errors
    fig, axs = plt.subplots(1, 2, figsize=(16, 10))
    axs[0].plot(params, cora_rot_rmse_errors, color=cora_color, label="CORA")
    axs[0].plot(params, gtsam_rot_rmse_errors, color=gtsam_color, label="GTSAM")
    axs[0].set_title("Rotation RMSE")
    axs[0].set_xlabel(exp_params[param_sweep_type])
    axs[0].set_ylabel("Rotation Error (degrees)")
    axs[0].legend()

    axs[1].plot(params, cora_rot_max_errors, color=cora_color, label="CORA")
    axs[1].plot(params, gtsam_rot_max_errors, color=gtsam_color, label="GTSAM")
    axs[1].set_title("Rotation Max Error")
    axs[1].set_xlabel(exp_params[param_sweep_type])
    axs[1].set_ylabel("Rotation Error (degrees)")
    axs[1].legend()

    # plt.ylabel("Rotation Error (degrees)")

    plt.savefig(join(save_dir, "rotation_errors_vs_params.png"))
    plt.savefig(join(save_dir, "rotation_errors_vs_params.svg"), format="svg")
    print(f"Saved rotation errors vs params plot to {save_dir}")
    if show_plots:
        plt.show()
    plt.close()

    # plot pose errors
    fig, axs = plt.subplots(1, 2, figsize=(16, 10))
    axs[0].plot(params, cora_pose_rmse_errors, color=cora_color, label="CORA")
    axs[0].plot(params, gtsam_pose_rmse_errors, color=gtsam_color, label="GTSAM")
    axs[0].set_title("Absolute Pose RMSE")
    axs[0].set_xlabel(exp_params[param_sweep_type])
    axs[0].set_ylabel("Absolute Pose Error")
    axs[0].legend()

    axs[1].plot(params, cora_pose_max_errors, color=cora_color, label="CORA")
    axs[1].plot(params, gtsam_pose_max_errors, color=gtsam_color, label="GTSAM")
    axs[1].set_title("Absolute Pose Max Error")
    axs[1].set_xlabel(exp_params[param_sweep_type])
    axs[1].set_ylabel("Absolute Pose Error")
    axs[1].legend()
    plt.savefig(join(save_dir, "pose_errors_vs_params.png"))
    plt.savefig(join(save_dir, "pose_errors_vs_params.svg"), format="svg")
    print(f"Saved pose errors vs params plot to {save_dir}")
    if show_plots:
        plt.show()
    plt.close()


if __name__ == "__main__":
    cora_manhattan_base_dir = expanduser("~/data/manhattan/cert")
    exp_options = ["no_loop_closures", "100loop_closures"]
    exp_suboptions = [
        # "sweep_num_poses",
        # "sweep_num_ranges",
        # "sweep_num_robots",
        "sweep_range_cov",
    ]

    for base_opt in exp_options:
        for sub_opt in exp_suboptions:
            cora_manhattan_dir = join(cora_manhattan_base_dir, base_opt, sub_opt)

            exp_subdirs = get_leaf_dirs(cora_manhattan_dir)
            ape_error_df_collection_path = join(
                cora_manhattan_dir, "ape_error_df_collection.pickle"
            )
            param_list = []
            rot_error_dfs = []
            pose_error_dfs = []
            param_list_pickle_path = join(cora_manhattan_dir, "param_list.pickle")
            rot_error_dfs_pickle_path = join(cora_manhattan_dir, "rot_error_dfs.pickle")
            pose_error_dfs_pickle_path = join(
                cora_manhattan_dir, "pose_error_dfs.pickle"
            )
            if not (
                isfile(param_list_pickle_path)
                and isfile(rot_error_dfs_pickle_path)
                and isfile(pose_error_dfs_pickle_path)
            ):
                # if not os.path.isfile(ape_error_df_collection_path) or True:
                # ape_error_df_collection = []
                for exp_dir in exp_subdirs:
                    print(f"\nProcessing {exp_dir}")
                    try:
                        ape_error_dfs, aligned_trajs = evaluate_results(exp_dir)
                        param_list.append(_get_param(sub_opt, exp_dir))
                        rot_error_dfs.append(ape_error_dfs.rot_error_df)
                        pose_error_dfs.append(ape_error_dfs.pose_error_df)

                        # ape_error_df_collection.append(ape_error_dfs)
                    except FileNotFoundError:
                        print(f"Could not find results in {exp_dir}, skipping...")

                # pickle.dump(ape_error_df_collection, open(ape_error_df_collection_path, "wb"))

                pickle.dump(param_list, open(param_list_pickle_path, "wb"))
                pickle.dump(rot_error_dfs, open(rot_error_dfs_pickle_path, "wb"))
                pickle.dump(pose_error_dfs, open(pose_error_dfs_pickle_path, "wb"))
            else:
                param_list = pickle.load(open(param_list_pickle_path, "rb"))
                rot_error_dfs = pickle.load(open(rot_error_dfs_pickle_path, "rb"))
                pose_error_dfs = pickle.load(open(pose_error_dfs_pickle_path, "rb"))

            error_plot_save_dir = join(
                "/home/alan/rss23-cora-author-feedback/imgs/multirobot",
                base_opt,
                sub_opt,
            )
            make_error_plots_vs_params(
                param_list,
                rot_error_dfs,
                pose_error_dfs,
                sub_opt,
                error_plot_save_dir,
                show_plots=False,
            )

            # ape_error_df_collection = pickle.load(open(ape_error_df_collection_path, "rb"))
            # make_box_and_whisker_error_plots(ape_error_df_collection, cora_manhattan_dir)
            # make_evo_traj_plots(aligned_trajs, cora_manhattan_dir, show_plots=True)
