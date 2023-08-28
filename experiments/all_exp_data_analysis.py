MARINE_DIRNAME = "marine"
PLAZA1_DIRNAME = "plaza/plaza1"
PLAZA2_DIRNAME = "plaza/plaza2"
SINGLE_DRONE_DIRNAME = "single_drone"
TIERS_DIRNAME = "tiers"
MULTI_ROBOT_EXPERIMENTS = [TIERS_DIRNAME]
EXPERIMENT_NAMING = {
    MARINE_DIRNAME: "Marine",
    PLAZA1_DIRNAME: "Plaza 1",
    PLAZA2_DIRNAME: "Plaza 2",
    SINGLE_DRONE_DIRNAME: "Single Drone",
    TIERS_DIRNAME: "TIERS",
}

import itertools
import numpy as np
import pandas as pd
from os.path import join
from typing import List, Tuple, Optional
from utils.evaluate_utils import (
    get_aligned_traj_results_in_dir,
    clean_df_collection,
    get_ape_error_stats_from_aligned_trajs,
    TrajErrorDfs,
    PREFERRED_LABEL_ORDERING,
)
from utils.paths import DATA_DIR
from attrs import define, field
import matplotlib.pyplot as plt
from matplotlib import ticker


def _reorder_cols_according_to_preferred_ordering(df: pd.DataFrame) -> pd.DataFrame:
    preferred_ordering = PREFERRED_LABEL_ORDERING
    cols_to_reorder = [col for col in preferred_ordering if col in df.columns]
    return df[cols_to_reorder]


def _set_grid_color(ax: plt.Axes):
    ax.grid(color="black", linestyle="-", linewidth=1)

    # remove vertical gridlines
    ax.grid(which="major", axis="x", linestyle="-", linewidth=0)


@define
class NamedResults:
    name: str = field()
    results: TrajErrorDfs = field()

    def __str__(self) -> str:
        return f"{self.name}:\n{self.results}"

    def plot_trans(self, target_ax: plt.Axes):
        assert not self.results.trajs_are_indices
        indices_to_plot = self.results.rot_error_df.index.drop("Max")
        plot_df = self.results.trans_error_df.loc[indices_to_plot]
        plot_df.plot.bar(
            ax=target_ax, ylabel="Translation RMSE (meters)", color=self.results.cmap
        )
        ylim_top_min = 3.0
        if target_ax.get_ylim()[1] < ylim_top_min:
            target_ax.set_ylim(0, ylim_top_min)
        target_ax.set_xticklabels(target_ax.get_xticklabels(), rotation=0)
        target_ax.set_facecolor("white")
        target_ax.legend(frameon=True, loc="upper left")
        _set_grid_color(target_ax)

    def plot_rot(self, target_ax: plt.Axes):
        assert not self.results.trajs_are_indices
        indices_to_plot = self.results.rot_error_df.index.drop("Max")
        plot_df = self.results.rot_error_df.loc[indices_to_plot]
        plot_df.plot.bar(
            ax=target_ax,
            ylabel="Rotation RMSE (deg)",
            color=self.results.cmap,
        )
        target_ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        target_ax.legend(frameon=True, loc="upper left")

        target_ax.set_xticklabels(target_ax.get_xticklabels(), rotation=0)
        target_ax.set_facecolor("white")
        _set_grid_color(target_ax)


def _get_experiment_named_results(
    exp_list: List[str], use_cached_trajs: bool, use_cached_error_dfs: bool
) -> List[NamedResults]:
    named_error_results = []
    for exp in exp_list:
        eval_dir = join(DATA_DIR, exp)
        aligned_trajs = get_aligned_traj_results_in_dir(
            eval_dir, use_cached_results=use_cached_trajs
        )
        exp_collected_error_dfs = get_ape_error_stats_from_aligned_trajs(
            aligned_trajs, eval_dir, use_cached_error_dfs
        )

        cleaned_collection = clean_df_collection(exp_collected_error_dfs)
        exp_name = EXPERIMENT_NAMING[exp]
        named_error_results.append(NamedResults(exp_name, cleaned_collection))

    return named_error_results


# only have y labels on the leftmost column and x labels on the bottom row
def _clean_axis_labels(axes: np.ndarray[plt.Axes]):
    num_rows, num_results = axes.shape
    assert num_rows == 2
    for i in range(num_results):
        top_ax = axes[0, i]
        assert isinstance(top_ax, plt.Axes)

        # none of the top axes should have x labels
        top_ax.set_xlabel("")
        top_ax.set_xticklabels([])

        # if this is not the leftmost column, remove the y labels
        is_leftmost_col = i == 0
        if not is_leftmost_col:
            top_ax.set_ylabel("")
            top_ax.set_yticklabels([])

            bottom_ax = axes[1, i]
            assert isinstance(bottom_ax, plt.Axes)
            bottom_ax.set_ylabel("")
            bottom_ax.set_yticklabels([])


def _set_ylim_the_same(axes: np.ndarray[plt.Axes]):
    # set the ylim of each row to be the max ylim
    num_rows, num_results = axes.shape
    top_row_ylim_max = 0
    bottom_row_ylim_max = 0
    for col in range(num_results):
        top_ax = axes[0, col]
        bottom_ax = axes[1, col]
        assert isinstance(top_ax, plt.Axes)
        assert isinstance(bottom_ax, plt.Axes)

        top_row_ylim_max = max(top_row_ylim_max, top_ax.get_ylim()[1])
        bottom_row_ylim_max = max(bottom_row_ylim_max, bottom_ax.get_ylim()[1])

    for col in range(num_results):
        top_ax = axes[0, col]
        bottom_ax = axes[1, col]
        assert isinstance(top_ax, plt.Axes)
        assert isinstance(bottom_ax, plt.Axes)

        top_ax.set_ylim(0, top_row_ylim_max)
        bottom_ax.set_ylim(0, bottom_row_ylim_max)


def _add_titles_above_top_row(
    axes: np.ndarray[plt.Axes], named_results: List[NamedResults]
):
    num_rows, num_results = axes.shape
    for col in range(num_results):
        top_ax = axes[0, col]
        assert isinstance(top_ax, plt.Axes)

        # make title bold, larger, and space it out from the plot
        top_ax.set_title(
            named_results[col].name, fontweight="bold", fontsize=16, pad=20
        )


def _remove_all_vertical_gridlines(axes: np.ndarray[plt.Axes]):
    try:
        num_rows, num_results = axes.shape
    except ValueError:
        num_results = 1
        num_rows = None
    for col in range(num_results):
        if num_rows is None:
            top_ax = axes[0]
            bottom_ax = axes[1]
        else:
            top_ax = axes[0, col]
            bottom_ax = axes[1, col]
        assert isinstance(top_ax, plt.Axes)
        assert isinstance(bottom_ax, plt.Axes)

        top_ax.grid(visible=None, axis="x")
        bottom_ax.grid(visible=None, axis="x")


def _clear_legends_except_desired_results(
    axes: np.ndarray[plt.Axes],
    named_results: List[NamedResults],
    names_to_keep: List[str],
):
    num_rows, num_results = axes.shape
    for col in range(num_results):
        top_ax = axes[0, col]
        bottom_ax = axes[1, col]
        assert isinstance(top_ax, plt.Axes)
        assert isinstance(bottom_ax, plt.Axes)

        if named_results[col].name.lower() not in names_to_keep:
            top_ax.get_legend().remove()
            bottom_ax.get_legend().remove()


def _rotate_xticklabels(axes: np.ndarray[plt.Axes]):
    try:
        num_rows, num_results = axes.shape
    except ValueError:
        num_results = 1
        num_rows = None

    for col in range(num_results):
        if num_rows is None:
            top_ax = axes[0]
            bottom_ax = axes[1]
        else:
            top_ax = axes[0, col]
            bottom_ax = axes[1, col]
        assert isinstance(top_ax, plt.Axes)
        assert isinstance(bottom_ax, plt.Axes)
        # make the x tick labels bold, larger, and space them out from the plot
        top_ax.tick_params(
            axis="x",
            which="major",
            labelsize=16,
            labelrotation=0,
        )
        bottom_ax.tick_params(
            axis="x",
            which="major",
            labelsize=16,
            labelrotation=0,
        )


def _replace_xticklabels_with_experiment_names(
    axes: np.ndarray[plt.Axes], named_results: List[NamedResults]
):
    num_rows, num_results = axes.shape
    for col in range(num_results):
        top_ax = axes[0, col]
        bottom_ax = axes[1, col]
        assert isinstance(top_ax, plt.Axes)
        assert isinstance(bottom_ax, plt.Axes)

        top_ax.set_xticklabels([named_results[col].name])
        bottom_ax.set_xticklabels([named_results[col].name])


def _join_results(
    named_results: List[NamedResults],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # make a df such that the indices are the experiment names and the columns
    # are the same columns as the error dfs

    # first, make sure all of the error dfs have the same columns
    first_cols = named_results[0].results.expected_cols
    assert all(
        [result.results.expected_cols == first_cols for result in named_results]
    ), (
        "Not all error dfs have the same columns. "
        " This is likely an error due to some results being cached. "
        " To be safe you can delete the cached results and re-run the experiments."
    )

    # now, make a df with the same columns as the error dfs
    rot_error_df = pd.DataFrame(
        columns=first_cols, index=[result.name for result in named_results]
    )
    trans_error_df = pd.DataFrame(
        columns=first_cols, index=[result.name for result in named_results]
    )
    target_idx = "RMSE"
    for result in named_results:
        res_name = result.name
        # fill each index with the corresponding RMSE values
        rot_error_df.loc[res_name] = result.results.rot_error_df.loc[target_idx]
        trans_error_df.loc[res_name] = result.results.trans_error_df.loc[target_idx]

    return rot_error_df, trans_error_df


def _plot_on_separate_axes(named_results: List[NamedResults]):
    # plot rotation error on top, translation error on bottom
    num_rows = 2
    num_cols = len(named_results)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))

    for i, named_result in enumerate(named_results):
        top_ax = axes[0, i]
        bottom_ax = axes[1, i]
        assert isinstance(top_ax, plt.Axes)
        assert isinstance(bottom_ax, plt.Axes)

        named_result.plot_rot(top_ax)
        named_result.plot_trans(bottom_ax)

    _set_ylim_the_same(axes)
    _clean_axis_labels(axes)
    # _add_titles_above_top_row(axes, named_results)
    _replace_xticklabels_with_experiment_names(axes, named_results)
    _rotate_xticklabels(axes)
    _remove_all_vertical_gridlines(axes)

    legends_to_keep = [MARINE_DIRNAME, TIERS_DIRNAME]
    _clear_legends_except_desired_results(axes, named_results, legends_to_keep)

    # remove any horizontal space between the plots
    plt.subplots_adjust(wspace=-0.1)

    # remove padding around bars in each axis
    for ax in axes.flatten():
        assert isinstance(ax, plt.Axes)
        ax.margins(x=-0.4)

        # reduce xlim
        cur_xlim = ax.get_xlim()
        shrink_factor = 0.0
        ax.set_xlim(
            cur_xlim[0] + shrink_factor * (cur_xlim[1] - cur_xlim[0]),
            cur_xlim[1] - shrink_factor * (cur_xlim[1] - cur_xlim[0]),
        )

    plt.show()


def _plot_named_results_on_same_axis(
    named_results: List[NamedResults],
    ylog: bool = False,
    savepath: Optional[str] = None,
    show: bool = True,
):
    joined_results = _join_results(named_results)
    joined_rot_rmse = _reorder_cols_according_to_preferred_ordering(joined_results[0])
    joined_trans_rmse = _reorder_cols_according_to_preferred_ordering(joined_results[1])

    # make a bar plot of the rotation and translation error
    # fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    fig, axes = plt.subplots(2, 1)
    colormap = named_results[0].results.cmap
    joined_rot_rmse.plot.bar(ax=axes[0], ylabel="Rotation RMSE (deg)", color=colormap)
    joined_trans_rmse.plot.bar(
        ax=axes[1], ylabel="Translation RMSE (meter)", color=colormap
    )
    # _add_titles_above_top_row(axes, named_results)
    _remove_all_vertical_gridlines(axes)
    _rotate_xticklabels(axes)

    for ax in axes:
        assert isinstance(ax, plt.Axes)

        # facecolor to white
        ax.set_facecolor("white")

        # grid formatting
        _set_grid_color(ax)

        # y-axis log scale
        if ylog:
            ax.set_yscale("log")

        # tight layout
        fig.tight_layout()

    # make extra room at the top for the legend
    for ax in axes:
        assert isinstance(ax, plt.Axes)
        ax_ymin, ax_ymax = ax.get_ylim()
        ax.set_ylim(ax_ymin, ax_ymax * 7.0)

        # move the legend up and break it into two rows
        num_legend_rows = 2
        num_entries = len(ax.get_legend().get_texts())
        num_cols = int(np.ceil(num_entries / num_legend_rows))

        def flip(items, ncol):
            return itertools.chain(*[items[i::ncol] for i in range(ncol)])

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            flip(handles, num_cols),
            flip(labels, num_cols),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=num_cols,
        )

    # add some padding between the plots
    plt.subplots_adjust(hspace=0.5)

    # make sure y-axis labels are aligned
    fig.align_ylabels(axes)

    if savepath is not None:
        fig.savefig(savepath, dpi=300)
        print(f"Saved plot to {savepath}")

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    use_cached_trajs = True
    use_cached_error_dfs = True
    show_plots = False
    single_robot_exps = [
        SINGLE_DRONE_DIRNAME,
        # MARINE_DIRNAME,
        PLAZA1_DIRNAME,
        PLAZA2_DIRNAME,
    ]
    single_robot_results = _get_experiment_named_results(
        exp_list=single_robot_exps,
        use_cached_trajs=use_cached_trajs,
        use_cached_error_dfs=use_cached_error_dfs,
    )
    single_robot_plot_fpath = join(DATA_DIR, "single_robot_results.png")
    _plot_named_results_on_same_axis(
        single_robot_results,
        ylog=True,
        savepath=single_robot_plot_fpath,
        show=show_plots,
    )

    multi_robot_results = _get_experiment_named_results(
        exp_list=MULTI_ROBOT_EXPERIMENTS,
        use_cached_trajs=use_cached_trajs,
        use_cached_error_dfs=use_cached_error_dfs,
    )
    multi_robot_plot_fpath = join(DATA_DIR, "multi_robot_results.png")
    _plot_named_results_on_same_axis(
        multi_robot_results, ylog=True, savepath=multi_robot_plot_fpath, show=show_plots
    )
