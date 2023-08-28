from utils.cora_iterates_evaluation_utils import plot_cora_cost_over_iters
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import pandas as pd

DIR_OPTIONS = {
    "block_cholesky": "block-diag Cholesky (Ours)",
    "ichol": "incomplete Cholesky",
    "block_jacobi": "block Jacobi",
    "jacobi": "Jacobi",
    "regularized_cholesky": "regularized incomplete Cholesky",
}


def plot_on_separate_axes(exp_dir: str):
    fig, axs = plt.subplots(2, 2, figsize=(16, 8))

    for dir_option, exp_name, axs_idx in zip(
        DIR_OPTIONS.keys(), DIR_OPTIONS.values(), [(0, 0), (0, 1), (1, 0), (1, 1)]
    ):
        dirpath = join(exp_dir, dir_option)
        ax = axs[axs_idx]
        try:
            plot_cora_cost_over_iters(dirpath, exp_name, ax)
        except FileNotFoundError:
            print(f"Could not find {dirpath}")

        row_idx, col_idx = axs_idx
        if row_idx == 1:
            ax.set_xlabel("Iterations")
        if col_idx == 0:
            ax.set_ylabel("Cost")

    # set all x-limits and y-limits to be the same
    max_x_limit = 0
    max_y_limit = 0
    min_y_limit = np.inf
    for ax in axs.flatten():
        max_x_limit = max(max_x_limit, ax.get_xlim()[1])
        max_y_limit = max(max_y_limit, ax.get_ylim()[1])
        min_y_limit = min(min_y_limit, ax.get_ylim()[0])

    for ax in axs.flatten():
        ax.set_xlim((-1, max_x_limit))
        ax.set_ylim((min_y_limit, max_y_limit))

    fpath = join(exp_dir, "multi_plot_cost_vs_iters.png")
    plt.savefig(fpath, dpi=300)
    print(f"Saved figure to {fpath}")

    plt.show()


def load_results(exp_dir: str) -> pd.DataFrame:
    # iterates_mat_path = join(exp_dir, ITERATES_MAT_FNAME)
    # iterates_mat = sio.loadmat(iterates_mat_path)

    # # get the data
    # info = iterates_mat[ITERATE_INFO_KEY]
    # assert isinstance(info, np.ndarray)
    # assert info.shape[0] == 1

    # # get the field names
    # field_names = info.dtype.names
    # assert isinstance(field_names, tuple)
    # assert all([isinstance(name, str) for name in field_names])

    # fields_vals = {name: info[0][name] for name in desired_fields}
    # cleaned_fields_vals: Dict[str, np.ndarray] = {
    #     name: np.array([float(val[i][0, 0]) for i in range(len(val))])
    #     for name, val in fields_vals.items()
    # }
    # return cleaned_fields_vals

    lower_bound_key = "certified_lower_bound"
    soln_cost_key = "final_soln_cost"
    solve_time_key = "solve_time"
    exp_names = []
    data = []
    for dir_opt, exp_name in DIR_OPTIONS.items():
        results_fpath = join(exp_dir, dir_opt, "factor_graph_results.mat")

        try:
            results = sio.loadmat(results_fpath)["results"][0][0]
        except FileNotFoundError:
            print(f"Could not find {results_fpath}")
            continue

        field_names = results.dtype.names
        expected_fields = ["X", lower_bound_key, soln_cost_key, solve_time_key]
        if set(field_names) != set(expected_fields):
            print(f"Expected fields: {expected_fields}, got: {field_names}")
            continue

        lower_bound_idx = field_names.index(lower_bound_key)
        final_cost_idx = field_names.index(soln_cost_key)
        solve_time_idx = field_names.index(solve_time_key)

        lower_bound = results[lower_bound_idx][0, 0]
        final_cost = results[final_cost_idx][0, 0]
        solve_time = results[solve_time_idx][0, 0]
        data.append([lower_bound, final_cost, solve_time])
        exp_names.append(exp_name)

    df = pd.DataFrame(
        data, index=exp_names, columns=[lower_bound_key, soln_cost_key, solve_time_key]
    )

    # sort df by solve time
    df = df.sort_values(by=solve_time_key)
    return df


def plot_on_same_axes(exp_dir: str, x_ub: float = 20):
    fig, ax = plt.subplots(figsize=(10, 10))

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    legend_handles = []
    zorder_cnt = 10
    for dir_option, exp_name in DIR_OPTIONS.items():
        dirpath = join(exp_dir, dir_option)
        try:
            plot_handle = plot_cora_cost_over_iters(
                dirpath,
                exp_name,
                ax,
                draw_vlines=False,
                color=colors.pop(),
                zorder=zorder_cnt,
                show_refinement_steps=False,
            )
            legend_handles.append(plot_handle)
            zorder_cnt -= 1

        except FileNotFoundError:
            print(f"Could not find {dirpath}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cost")

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_xlim((-1, x_ub))
    # ax.set_ylim((ymin/10, ymin*1e3))

    # clear any title
    ax.set_title("")

    # set legend as the exp_names
    ax.legend(legend_handles)

    # show the legend
    ax.legend()

    # increase the font size of the legend, axes labels, and axes ticks
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(30)

    # set the font size of the legend
    ax.legend(fontsize=24)

    fpath = join(exp_dir, "single_plot_cost_vs_iters.png")
    plt.savefig(fpath, dpi=300)
    print(f"Saved figure to {fpath}")

    # plt.show()


if __name__ == "__main__":
    exp_dirs = {
        "plaza1": "/home/alan/range-only-slam-mission-control/cora-experiments/data/plaza/plaza1",
        "plaza2": "/home/alan/range-only-slam-mission-control/cora-experiments/data/plaza/plaza2",
        "single_drone": "/home/alan/range-only-slam-mission-control/cora-experiments/data/single_drone",
        "tiers": "/home/alan/range-only-slam-mission-control/cora-experiments/data/tiers",
    }
    # exp_dir = exp_dirs["single_drone"]
    # exp_dir = exp_dirs["plaza1"]
    for exp_dir in exp_dirs.values():
        # plot_on_separate_axes(exp_dir)
        print(exp_dir)
        df = load_results(exp_dir)
        print(df)
        print()

        # x_ub is the solve_time of the first method
        x_ub = df.iloc[0]["solve_time"] / 2
        print(f"Setting x_ub to {x_ub}")
        plot_on_same_axes(exp_dir, x_ub=x_ub)
    # exp_dir = exp_dirs["tiers"]
    # plot_on_separate_axes(exp_dir)
    # plot_on_same_axes(exp_dir)
