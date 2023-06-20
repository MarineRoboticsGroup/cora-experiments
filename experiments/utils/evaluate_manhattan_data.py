from typing import List, Tuple, Optional
from os.path import join, isfile, isdir
import os
import pickle
from attrs import define, field
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
import seaborn as sns
import pandas as pd

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

from .evaluate_utils import (
    get_aligned_traj_results_in_dir,
    get_ape_error_stats_from_aligned_trajs,
    TrajErrorDfs,
    ResultsPoseTrajCollection,
)
from .paths import DATA_DIR, get_leaf_dirs
from .generate_manhattan_experiments import (
    SWEEP_NUM_ROBOTS,
    SWEEP_NUM_BEACONS,
    SWEEP_PCT_LOOP_CLOSURES,
    SWEEP_NUM_POSES,
    SWEEP_NUM_RANGES,
    SWEEP_RANGE_COV,
    MANHATTAN_EXPERIMENTS,
    EXPERIMENT_TRAILING_STR,
)

MANHATTAN_DATA_DIR = join(DATA_DIR, "manhattan")
SUBEXP_FNAME = "subexperiment_results.pickle"


@define
class SubExperimentResults:
    param_list: List = field(default=[])
    traj_error_df_collection: List[TrajErrorDfs] = field(default=[])
    traj_collections: List[ResultsPoseTrajCollection] = field(default=[])

    def add_experimental_result(
        self,
        param,
        traj_error_df: TrajErrorDfs,
        traj_collection: ResultsPoseTrajCollection,
    ):
        self.param_list.append(param)
        self.traj_error_df_collection.append(traj_error_df)
        self.traj_collections.append(traj_collection)

    def sort_according_to_params(
        self,
    ):
        """Sort the error dfs according to the params"""
        sorted_params, sorted_ape_error_df_collection, sorted_traj_collections = (
            list(t)
            for t in zip(
                *sorted(
                    zip(
                        self.param_list,
                        self.traj_error_df_collection,
                        self.traj_collections,
                    ),  # sort according to the params
                    key=lambda x: x[0],
                )
            )
        )
        self.param_list = sorted_params
        self.traj_error_df_collection = sorted_ape_error_df_collection
        self.traj_collections = sorted_traj_collections


def make_box_and_whisker_error_plots(
    subexp_results: SubExperimentResults,
    save_dir: str,
    show_plots: bool = False,
):
    def _prep_error_dfs_for_manhattan_box_whisker(
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

        (
            rot_rmse_df,
            rot_max_df,
        ) = _get_as_rmse_and_max_dfs_with_indices_diff_experiments(
            stacked_rot_error_dfs
        )
        (
            trans_rmse_df,
            trans_max_df,
        ) = _get_as_rmse_and_max_dfs_with_indices_diff_experiments(
            stacked_trans_error_dfs
        )
        (
            pose_rmse_df,
            pose_max_df,
        ) = _get_as_rmse_and_max_dfs_with_indices_diff_experiments(
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
            [
                collected_rot_errors_df,
                collected_trans_errors_df,
                collected_pose_errors_df,
            ],
            axis=1,
            keys=["Rotation", "Translation", "Pose"],
        )

        return all_errors_collected_df

    ape_error_df_collection = subexp_results.traj_error_df_collection
    collected_dfs = _prep_error_dfs_for_manhattan_box_whisker(ape_error_df_collection)

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
        logger.info(f"Saved plot to {save_path}")
        if show_plots:
            plt.show()

        # close the individual figure
        plt.close(fig)
    # plt.show()


def make_error_plots_vs_params(
    subexp_results: SubExperimentResults,
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
    params = subexp_results.param_list
    if param_sweep_type == "sweep_range_cov":
        params = [p**2 for p in params]

    # sort the params in increasing order and sort the error dfs accordingly
    subexp_results.sort_according_to_params()
    params, ape_error_collections = (
        subexp_results.param_list,
        subexp_results.traj_error_df_collection,
    )

    gtsam_color = "red"
    cora_color = "blue"

    exp_params = {
        "sweep_num_poses": r"\textbf{\# poses}",
        "sweep_num_ranges": r"\textbf{\# range measurements}",
        "sweep_num_robots": r"\textbf{\# robots}",
        "sweep_range_cov": r"\textbf{$\sigma_{ij}^2$}",
        "sweep_num_beacons": r"\textbf{\# beacons}",
        "sweep_pct_loop_closures": r"\textbf{\% loop closures}",
    }

    exp_params = {
        "sweep_num_poses": "number of poses",
        "sweep_num_ranges": "number of range measurements",
        "sweep_num_robots": "number of robots",
        "sweep_range_cov": r"{$\sigma_{ij}^2$}",
        "sweep_num_beacons": "number of beacons",
        "sweep_pct_loop_closures": "\% loop closures",
    }

    # we want rmse and max
    cora_rot_rmse_errors = []
    cora_rot_max_errors = []
    gtsam_rot_rmse_errors = []
    gtsam_rot_max_errors = []
    cora_trans_rmse_errors = []
    cora_trans_max_errors = []
    gtsam_trans_rmse_errors = []
    gtsam_trans_max_errors = []
    cora_pose_rmse_errors = []
    cora_pose_max_errors = []
    gtsam_pose_rmse_errors = []
    gtsam_pose_max_errors = []
    for ape_error_collection in ape_error_collections:
        # collect rotation error dfs
        rot_error_df = ape_error_collection.rot_error_df
        cora_rot_rmse_errors.append(rot_error_df.loc["CORA", "rmse"])
        cora_rot_max_errors.append(rot_error_df.loc["CORA", "max"])
        gtsam_rot_rmse_errors.append(rot_error_df.loc["GTSAM", "rmse"])
        gtsam_rot_max_errors.append(rot_error_df.loc["GTSAM", "max"])

        # collect translation error dfs
        trans_error_df = ape_error_collection.trans_error_df
        cora_trans_rmse_errors.append(trans_error_df.loc["CORA", "rmse"])
        cora_trans_max_errors.append(trans_error_df.loc["CORA", "max"])
        gtsam_trans_rmse_errors.append(trans_error_df.loc["GTSAM", "rmse"])
        gtsam_trans_max_errors.append(trans_error_df.loc["GTSAM", "max"])

        # collect pose_error dfs
        pose_error_df = ape_error_collection.pose_error_df
        cora_pose_rmse_errors.append(pose_error_df.loc["CORA", "rmse"])
        cora_pose_max_errors.append(pose_error_df.loc["CORA", "max"])
        gtsam_pose_rmse_errors.append(pose_error_df.loc["GTSAM", "rmse"])
        gtsam_pose_max_errors.append(pose_error_df.loc["GTSAM", "max"])

    rot_errors = (
        cora_rot_rmse_errors,
        cora_rot_max_errors,
        gtsam_rot_rmse_errors,
        gtsam_rot_max_errors,
    )
    trans_errors = (
        cora_trans_rmse_errors,
        cora_trans_max_errors,
        gtsam_trans_rmse_errors,
        gtsam_trans_max_errors,
    )
    pose_errors = (
        cora_pose_rmse_errors,
        cora_pose_max_errors,
        gtsam_pose_rmse_errors,
        gtsam_pose_max_errors,
    )

    # define error type as Tuple[List[float], List[float], List[float], List[float]]
    ErrorContainer = Tuple[List[float], List[float], List[float], List[float]]

    def _line_plot_helper(
        errors: ErrorContainer,
        units: Optional[str],
        title: str,
        ylabel: str,
        fname: str,
        show_confidence_intervals: bool = False,
    ):
        cora_rmse, cora_max, gtsam_rmse, gtsam_max = errors
        fig, axs = plt.subplots(1, 2, figsize=(16, 10))

        fig.subplots_adjust(wspace=0.3)
        plt.rcParams["lines.linewidth"] = 4
        plt.rcParams["axes.labelsize"] = 30
        plt.rcParams["xtick.labelsize"] = 30
        plt.rcParams["ytick.labelsize"] = 30
        plt.rcParams["legend.fontsize"] = 25

        if show_confidence_intervals:
            sns.lineplot(
                x=params,
                y=cora_rmse,
                color=cora_color,
                label="CORA",
                ax=axs[0],
                errorbar="sd",
                err_style="band",
            )
            sns.lineplot(
                x=params,
                y=gtsam_rmse,
                color=gtsam_color,
                label="GTSAM",
                ax=axs[0],
                errorbar="sd",
                err_style="band",
            )
        else:
            axs[0].plot(params, cora_rmse, color=cora_color, label="CORA")
            axs[0].plot(params, gtsam_rmse, color=gtsam_color, label="GTSAM")
        if units is not None:
            axs[0].set_ylabel(f"{title} RMSE ({units})")
        else:
            axs[0].set_ylabel(f"{title} RMSE")
        axs[0].set_xlabel(exp_params[param_sweep_type])
        axs[0].legend()

        xmin, xmax = min(params), max(params)
        axs[0].set_xlim(xmin, xmax)

        if show_confidence_intervals:
            sns.lineplot(
                x=params,
                y=cora_max,
                color=cora_color,
                label="CORA",
                ax=axs[1],
                errorbar="sd",
                err_style="band",
            )
            sns.lineplot(
                x=params,
                y=gtsam_max,
                color=gtsam_color,
                label="GTSAM",
                ax=axs[1],
                errorbar="sd",
                err_style="band",
            )
        else:
            axs[1].plot(params, cora_max, color=cora_color, label="CORA")
            axs[1].plot(params, gtsam_max, color=gtsam_color, label="GTSAM")
        if units is not None:
            axs[1].set_ylabel(f"{title} Max Error ({units})")
        else:
            axs[1].set_ylabel(f"{title} Max Error")
        axs[1].set_xlabel(exp_params[param_sweep_type])
        axs[1].set_xlim(xmin, xmax)
        axs[1].legend()

        # set both axes to have the same y limits
        ymin0, ymax0 = axs[0].get_ylim()
        ymin1, ymax1 = axs[1].get_ylim()
        ymin, ymax = min(ymin0, ymin1), max(ymax0, ymax1)

        for ax in axs:
            ax.set_ylim(ymin, ymax)

        # set the grid color to light gray and the facecolor to transparent
        for ax in axs:
            ax.grid(color="gainsboro")
            ax.set_facecolor("w")

        plt.savefig(join(save_dir, fname))
        plt.savefig(join(save_dir, fname.replace(".png", ".svg")), format="svg")
        logger.info(f"Saved {fname} plot to {join(save_dir, fname)}")
        if show_plots:
            plt.show()
        plt.close()

    _line_plot_helper(
        rot_errors,
        "degrees",
        "Rotation",
        "Rotation Error (degrees)",
        "rotation_errors_vs_params.png",
        show_confidence_intervals=True,
    )
    _line_plot_helper(
        trans_errors,
        "meters",
        "Translation",
        "Translation Error (meters)",
        "translation_errors_vs_params.png",
        show_confidence_intervals=True,
    )
    _line_plot_helper(
        pose_errors,
        None,
        "Pose",
        "Absolute Pose Error",
        "pose_errors_vs_params.png",
        show_confidence_intervals=True,
    )


def _get_subexperiment_results(
    subexperiment_base_dir: str,
    subexperiment_type: str,
    use_cached_sweep_results: bool,
    use_cached_subexp_results: bool,
) -> SubExperimentResults:
    subexperiment_dir = subexperiment_base_dir
    assert isdir(subexperiment_dir), f"Couldn't find directory: {subexperiment_dir}"

    def _get_param(exp_type: str, dir_path: str):
        assert (
            exp_type in MANHATTAN_EXPERIMENTS
        ), f"Invalid experiment type: {exp_type}, expected something from {MANHATTAN_EXPERIMENTS}"

        dir_path_components = dir_path.split("/")
        param_type_subdirs = [d for d in dir_path_components if d.startswith("sweep")]
        assert (
            len(param_type_subdirs) == 1
        ), f"Found multiple param type subdirs: {param_type_subdirs}"
        assert (
            param_type_subdirs[0] == exp_type
        ), f"Trying to extract param type {exp_type} from experiment directory {dir_path}"

        # get the part of the path we expect to hold the info
        trailing_str = EXPERIMENT_TRAILING_STR[exp_type]
        param_subdirs = [
            d
            for d in dir_path_components
            if d.endswith(trailing_str) and not d.startswith("sweep")
        ]
        assert len(param_subdirs) == 1, f"Found multiple param subdirs: {param_subdirs}"
        param_relevant_subdirs = param_subdirs[0]
        trailing_str_idx = param_relevant_subdirs.index(trailing_str)

        # extract the value of the param and cast to the appropriate type
        experiment_to_param_type = {
            SWEEP_NUM_ROBOTS: int,
            SWEEP_NUM_BEACONS: int,
            SWEEP_PCT_LOOP_CLOSURES: float,
            SWEEP_NUM_POSES: int,
            SWEEP_NUM_RANGES: int,
            SWEEP_RANGE_COV: float,
        }
        param_value = param_relevant_subdirs[:trailing_str_idx]
        param_type = experiment_to_param_type[exp_type]

        # cast the value to correct type and return
        try:
            return param_type(param_value)
        except ValueError:
            raise ValueError(
                f"Could not cast {param_value} to type {param_type} for experiment type {exp_type}"
            )

    # the files that cached results are saved to
    subexp_fpath = join(subexperiment_dir, SUBEXP_FNAME)

    # if the cached files exist and we want to use them, just load them
    if use_cached_sweep_results and isfile(subexp_fpath):
        logger.warning(
            f"Loading cached results for all {subexperiment_type} experiments"
        )
        subexp_results = pickle.load(open(subexp_fpath, "rb"))
    else:
        leaf_subdirs = get_leaf_dirs(subexperiment_dir)
        subexp_results = SubExperimentResults([], [], [])
        from time import perf_counter

        for exp_dirpath in leaf_subdirs:
            single_subexp_start = perf_counter()
            logger.info(f"\nProcessing {exp_dirpath}")
            try:
                param = _get_param(subexperiment_type, exp_dirpath)
                aligned_trajs = get_aligned_traj_results_in_dir(
                    exp_dirpath, use_cached_results=use_cached_subexp_results
                )
                ape_error_dfs = get_ape_error_stats_from_aligned_trajs(
                    aligned_trajs, exp_dirpath, use_cached_subexp_results
                )
                subexp_results.add_experimental_result(
                    param, ape_error_dfs, aligned_trajs
                )
                print()
            except FileNotFoundError as e:
                logger.error(f"Could not find reslts in {exp_dirpath}: {e}")
                raise e

            single_subexp_end = perf_counter()
            logger.info(
                f"Processed {exp_dirpath} in {single_subexp_end - single_subexp_start:.2f} seconds"
            )

        # sort the results according to the params
        subexp_results.sort_according_to_params()

        # cache the results
        pickle.dump(subexp_results, open(subexp_fpath, "wb"))

    return subexp_results


import gc


def make_manhattan_experiment_plots(
    base_experiment_dir: str = MANHATTAN_DATA_DIR,
    subexperiment_types: List[str] = MANHATTAN_EXPERIMENTS,
    use_cached_sweep_results: bool = False,
    use_cached_subexp_results: bool = False,
):
    assert isdir(
        base_experiment_dir
    ), f"Base experimental Manhattan dir not found: {base_experiment_dir}"
    for sub_opt in subexperiment_types:
        subexperiment_dir = join(base_experiment_dir, sub_opt)
        assert isdir(
            subexperiment_dir
        ), f"Could not find subexperiment dir: {subexperiment_dir}"

        subexp_results = _get_subexperiment_results(
            subexperiment_dir,
            sub_opt,
            use_cached_sweep_results=use_cached_sweep_results,
            use_cached_subexp_results=use_cached_subexp_results,
        )
        make_error_plots_vs_params(
            subexp_results,
            sub_opt,
            subexperiment_dir,
            show_plots=False,
        )
        # make_box_and_whisker_error_plots(
        #     subexp_results, subexperiment_dir, show_plots=False
        # )
        gc.collect()
