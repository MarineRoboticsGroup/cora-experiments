from typing import List, Tuple, Optional, Any
from os.path import join, isfile, isdir
import os
import pickle
from attrs import define, field, validators
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams["text.usetex"] = True
import seaborn as sns
import pandas as pd

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
from .logging_utils import logger

MANHATTAN_DATA_DIR = join(DATA_DIR, "manhattan")
SUBEXP_FNAME = "subexperiment_results.pickle"


def get_param_from_dirpath(exp_type: str, dir_path: str):
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


@define
class SubExperimentTrajResults:
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


def _make_box_and_whisker_error_plots(
    subexp_results: SubExperimentTrajResults,
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


def _make_error_plots_vs_params(
    subexp_results: SubExperimentTrajResults,
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


def _get_subexperiment_traj_results(
    subexperiment_base_dir: str,
    subexperiment_type: str,
    use_cached_sweep_results: bool,
    use_cached_subexp_results: bool,
) -> SubExperimentTrajResults:
    subexperiment_dir = subexperiment_base_dir
    assert isdir(subexperiment_dir), f"Couldn't find directory: {subexperiment_dir}"

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
        subexp_results = SubExperimentTrajResults([], [], [])
        from time import perf_counter

        for exp_dirpath in leaf_subdirs:
            single_subexp_start = perf_counter()
            logger.info(f"\nProcessing {exp_dirpath}")
            try:
                param = get_param_from_dirpath(subexperiment_type, exp_dirpath)
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


def make_manhattan_rmse_plots(
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

        subexp_results = _get_subexperiment_traj_results(
            subexperiment_dir,
            sub_opt,
            use_cached_sweep_results=use_cached_sweep_results,
            use_cached_subexp_results=use_cached_subexp_results,
        )
        _make_error_plots_vs_params(
            subexp_results,
            sub_opt,
            subexperiment_dir,
            show_plots=False,
        )
        _make_box_and_whisker_error_plots(
            subexp_results, subexperiment_dir, show_plots=False
        )


def get_subopt_results_in_dir(exp_dir: str) -> Tuple[float, float]:
    results_fpath = join(exp_dir, "factor_graph_results.mat")
    if not isfile(results_fpath):
        raise FileNotFoundError(f"Could not find results file: {results_fpath}")

    results = loadmat(results_fpath)
    certified_lower_bound = results["results"]["certified_lower_bound"][0][0].item()
    final_cost = results["results"]["final_soln_cost"][0][0].item()
    return certified_lower_bound, final_cost


SUBOPT_GAP_LABEL = "Optimality Gap (\%)"
FINAL_COST_LABEL = "Final Cost"
LOWER_BOUND_LABEL = "Certified Lower Bound"


@define
class SubExperimentSuboptResults:
    with_loop_closures: bool = field(validator=validators.instance_of(bool))
    param_sweep_type: str = field(
        validator=[validators.instance_of(str), validators.in_(MANHATTAN_EXPERIMENTS)]
    )
    param_list: List = field(default=[])
    certified_lower_bounds: List[float] = field(default=[])
    final_costs: List[float] = field(default=[])

    @property
    def suboptimality_gaps(self) -> List[float]:
        gaps = [
            (final_cost - certified_lower_bound) / certified_lower_bound * 100
            for final_cost, certified_lower_bound in zip(
                self.final_costs, self.certified_lower_bounds
            )
        ]

        gaps = [max(0, gap) for gap in gaps]
        return gaps

    def num_zero_gap_experiments(self, zero_eps=1e-4) -> int:
        zero_gaps = [gap / 100 < zero_eps for gap in self.suboptimality_gaps]
        zero_gap_cnt = sum(zero_gaps)
        assert isinstance(zero_gap_cnt, int)
        return zero_gap_cnt

    @property
    def num_experiments(self) -> int:
        n = len(self.param_list)
        assert isinstance(n, int)
        return n

    def add_experimental_result(
        self, param: Any, certified_lower_bound: float, final_cost: float
    ):
        self.param_list.append(param)
        self.certified_lower_bounds.append(certified_lower_bound)
        self.final_costs.append(final_cost)

    def results_to_df(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                LOWER_BOUND_LABEL: self.certified_lower_bounds,
                FINAL_COST_LABEL: self.final_costs,
                SUBOPT_GAP_LABEL: self.suboptimality_gaps,
            },
            index=self.param_list,
        )
        df.index.name = self.param_sweep_type
        return df


from .generate_manhattan_experiments import (
    USE_LOOP_CLOSURE,
    NO_LOOP_CLOSURE,
    LOOP_CLOSURE_OPTIONS,
    EXPERIMENT_PLOT_TITLES,
)
from scipy.io import loadmat


LOOP_CLOSURE_TITLES = {
    USE_LOOP_CLOSURE: "With Loop Closures",
    NO_LOOP_CLOSURE: "No Loop Closures",
}


def _plot_subopt_results(
    subopt_results: SubExperimentSuboptResults,
    loop_closure_status: str,
    subexperiment_type: str,
    ax: Optional[plt.Axes] = None,
):
    print(
        f"Plotting {subexperiment_type} results with loop closures: {loop_closure_status}"
    )

    zero_eps = 1e-5
    num_zero_gap_experiments = subopt_results.num_zero_gap_experiments(
        zero_eps=zero_eps
    )
    num_experiments = subopt_results.num_experiments
    assert isinstance(num_zero_gap_experiments, int) and isinstance(
        num_experiments, int
    ), f"Invalid number of experiments: {num_zero_gap_experiments}, {num_experiments}"
    assert num_zero_gap_experiments <= num_experiments
    print()
    print(f"Computing tightness of optimality gap with zero eps: {zero_eps}")
    print(
        f"Number of experiments with zero gap: {num_zero_gap_experiments} / {num_experiments}"
    )
    print(
        f"Percentage of experiments with zero gap: {num_zero_gap_experiments / num_experiments * 100:.2f}%"
    )
    print()

    results_df = subopt_results.results_to_df()

    # Set the style of seaborn for better visualization
    sns.set(style="whitegrid")

    # Create the box-and-whisker plot
    using_outside_ax = ax is not None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    # y label is suboptimality gap (percentage)
    # sns.boxplot(x=results_df.index, y=SUBOPT_GAP_LABEL, data=results_df)

    # plot as line plot with shading of confidence interval
    # sns.lineplot(
    #     x=results_df.index,
    #     y=SUBOPT_GAP_LABEL,
    #     data=results_df,
    #     ax=ax,
    # )

    # plot as violin plots, with separate violing for each index
    # all colors should be the same
    sns.violinplot(
        x=results_df.index,
        y=SUBOPT_GAP_LABEL,
        data=results_df,
        ax=ax,
        # hue="smoker",
        palette="muted",
        split=True,
        cut=0,
        # color="tab:violet",
    )

    # if results_df.index == "sweep_num_beacons":
    # print(results_df.head())

    # xticks should be all integers
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # if we are doing "Number of Ranges", lets rotate the xticks
    # if subexperiment_type == "sweep_num_ranges":
    #     ax.xaxis.set_tick_params(rotation=45)

    # increase fontsize for x and y ticks
    ax.tick_params(axis="both", which="major", labelsize=18)

    # set the xlimits tight to the data
    # xmin, xmax = min(results_df.index), max(results_df.index)
    # ax.set_xlim(xmin, xmax)

    # Set title and labels for the plot
    if using_outside_ax:
        plt.xlabel("")
        plt.ylabel("")
    else:
        x_label = EXPERIMENT_PLOT_TITLES[subexperiment_type]
        # plt.title(
        #     f"Suboptimality Gap vs. {x_label} ({LOOP_CLOSURE_TITLES[loop_closure_status]})"
        # )
        plt.xlabel(x_label)
        plt.ylim(0, 7)

        plt.tight_layout()
        plt.show()


def make_manhattan_subopt_plots(
    base_experiment_dir: str = MANHATTAN_DATA_DIR,
    loop_closure_options: List[str] = LOOP_CLOSURE_OPTIONS,
    subexperiment_types: List[str] = MANHATTAN_EXPERIMENTS,
):
    assert isdir(
        base_experiment_dir
    ), f"Base experimental Manhattan dir not found: {base_experiment_dir}"

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    from itertools import product

    axes_row_idxs = [0, 1]
    axes_col_idxs = [0, 1, 2]
    axes_idxs = list(product(axes_row_idxs, axes_col_idxs))
    axes_cnt = 0
    for loop_closure_option in loop_closure_options:
        assert loop_closure_option in LOOP_CLOSURE_OPTIONS
        has_loop_closures = loop_closure_option == USE_LOOP_CLOSURE

        for sub_exp in subexperiment_types:
            subexperiment_dir = join(base_experiment_dir, loop_closure_option, sub_exp)
            assert isdir(
                subexperiment_dir
            ), f"Could not find parameter sweep dir: {subexperiment_dir}"

            leaf_subdirs = get_leaf_dirs(subexperiment_dir)
            subexp_results = SubExperimentSuboptResults(
                has_loop_closures, sub_exp, [], [], []
            )

            for exp_dirpath in leaf_subdirs:
                try:
                    param = get_param_from_dirpath(sub_exp, exp_dirpath)
                    certified_lower_bound, final_cost = get_subopt_results_in_dir(
                        exp_dirpath
                    )
                    opt_gap = final_cost - certified_lower_bound
                    opt_gap_epsilon = 5e-1
                    if opt_gap <= -opt_gap_epsilon:
                        msg = f"Optimality gap should be non-negative: {opt_gap}, final cost: {final_cost}, lb: {certified_lower_bound}"
                        raise ValueError(msg)
                    subexp_results.add_experimental_result(
                        param, certified_lower_bound, final_cost
                    )
                except FileNotFoundError as e:
                    logger.info(f"Could not find results in {exp_dirpath}: {e}")
                    # raise e
                except ValueError as e:
                    logger.info(e)

            if subexp_results.num_experiments == 0:
                raise ValueError(
                    f"No experiments found for {loop_closure_option} {sub_exp}"
                )
            ax_idx = axes_idxs[axes_cnt]
            print(f"Plotting {loop_closure_option} {sub_exp} on {ax_idx}")
            _plot_subopt_results(
                subexp_results, loop_closure_option, sub_exp, ax=axes[ax_idx]
            )
            axes_cnt += 1

            ax_row, ax_col = ax_idx
            if ax_row == 1:
                axes[ax_idx].set_xlabel(
                    EXPERIMENT_PLOT_TITLES[sub_exp], fontsize=26, labelpad=20
                )
            else:
                axes[ax_idx].set_xlabel("")
            if ax_col == 0:
                # space the y label out a bit
                axes[ax_idx].set_ylabel(SUBOPT_GAP_LABEL, fontsize=20, labelpad=20)
            else:
                axes[ax_idx].set_ylabel("")

    # set the heights constant within each row
    for row_idx in axes_row_idxs:
        max_ylim_in_row = max(
            [axes[(row_idx, col_idx)].get_ylim()[1] for col_idx in axes_col_idxs]
        )
        for col_idx in axes_col_idxs:
            axes[(row_idx, col_idx)].set_ylim(0, max_ylim_in_row)

    # set the background to white and the grid to light gray
    for ax in axes.flatten():
        ax.set_facecolor("w")
        ax.grid(color="gainsboro")

        # add lines on the bottom and left of the plot
        ax.spines["bottom"].set_color("black")
        ax.spines["left"].set_color("black")

    # to the left of the first row, add a label for the loop closure status
    for row_idx in axes_row_idxs:
        axes[row_idx, 0].text(
            -0.25,
            0.5,
            LOOP_CLOSURE_TITLES[loop_closure_options[row_idx]],
            horizontalalignment="center",
            verticalalignment="center",
            rotation=90,
            transform=axes[(row_idx, 0)].transAxes,
            fontsize=26,
        )

    savedir = "/home/alan/rss23-ra-slam-certification/figures/experiments/manhattan"
    fpath = join(savedir, "suboptimality_gap_vs_params_violin.png")
    # plt.savefig(fpath)
    print(f"Saved suboptimality plots to {fpath}")

    plt.show()
