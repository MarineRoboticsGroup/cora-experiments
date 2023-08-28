import scipy.io as sio
from os.path import join
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

ITERATES_MAT_FNAME = "factor_graph_cora_iterates_info.mat"
HEADER_KEY = "__header__"
ITERATE_INFO_KEY = "cora_iterates_info"


def _left_shift_idxs_following_pts(
    idxs: np.ndarray, left_shift_pts: np.ndarray
) -> None:
    # for each reset, shift the idxs after the reset by the value of the idx before the reset
    for i in left_shift_pts:
        idxs[i:] -= 1


def get_solver_info_as_dict(
    exp_dir: str, desired_fields: List[str] = ["iter", "cost", "gradnorm", "time"]
) -> Dict[str, np.ndarray]:
    """Reads the iterates info from the matlab file and returns it as a dictionary

    possible fields are:
        - iter
        - cost
        - gradnorm
        - Delta
        - time
        - rho
        - rhonum
        - rhoden
        - accepted
        - stepsize
        - limitedbyTR
        - numinner
        - hessvecevals
        - memorytCG_MB
        - Xvals
        - hooked

    Args:
        exp_dir (str): the experiment directory
        desired_fields (List[str], optional): _description_. Defaults to ["iter", "cost", "gradnorm", "time"].

    Returns:
        Dict[str, np.ndarray]: the dictionary of the desired fields
    """

    iterates_mat_path = join(exp_dir, ITERATES_MAT_FNAME)
    iterates_mat = sio.loadmat(iterates_mat_path)

    # get the data
    info = iterates_mat[ITERATE_INFO_KEY]
    assert isinstance(info, np.ndarray)
    assert info.shape[0] == 1

    # get the field names
    field_names = info.dtype.names
    assert isinstance(field_names, tuple)
    assert all([isinstance(name, str) for name in field_names])

    fields_vals = {name: info[0][name] for name in desired_fields}
    cleaned_fields_vals: Dict[str, np.ndarray] = {
        name: np.array([float(val[i][0, 0]) for i in range(len(val))])
        for name, val in fields_vals.items()
    }
    return cleaned_fields_vals


def get_certified_lower_bound_and_final_cost(
    exp_dir: str,
) -> Tuple[float, float]:
    """Gets the certified lower bound and final cost from the matlab file

    Args:
        exp_dir (str): the experiment directory

    Returns:
        Tuple[float, float]: the certified lower bound and final cost
    """
    info = get_solver_info_as_dict(exp_dir, desired_fields=["cost", "iter"])
    cora_reset_idxs = np.array([i for i, val in enumerate(info["iter"]) if val == 0])

    final_cost = info["cost"][-1]
    certified_lower_bound = info["cost"][cora_reset_idxs[-1] - 1]
    print(f"final cost: {final_cost}")
    print(f"certified lower bound: {certified_lower_bound}")

    return certified_lower_bound, final_cost


def plot_cora_cost_over_iters(
    exp_dir: str,
    plot_title: str = None,
    ax: plt.Axes = None,
    draw_vlines: bool = True,
    color: str = "tab:blue",
    show_refinement_steps: bool = False,
    zorder: Optional[int] = None,
) -> None:
    cleaned_fields_vals = get_solver_info_as_dict(exp_dir)
    cora_reset_idxs = np.array(
        [i for i, val in enumerate(cleaned_fields_vals["iter"]) if val == 0]
    )

    # for iter and time, we need to correct for the resets
    for name in ["iter"]:
        vals_before_reset = cleaned_fields_vals[name][cora_reset_idxs[1:] - 1]
        nonfirst_reset_idxs = cora_reset_idxs[1:]
        for v, idx in zip(vals_before_reset, nonfirst_reset_idxs):
            cleaned_fields_vals[name][idx:] += v + 1

    for name in ["time"]:
        vals_before_reset = cleaned_fields_vals[name][cora_reset_idxs[1:] - 1]
        nonfirst_reset_idxs = cora_reset_idxs[1:]
        for v, idx in zip(vals_before_reset, nonfirst_reset_idxs):
            cleaned_fields_vals[name][idx:] += v

    # check that the iterates are indeed the same as the indices
    assert (
        cleaned_fields_vals["iter"] == np.arange(len(cleaned_fields_vals["iter"]))
    ).all()

    # check that time is strictly increasing
    assert np.all(np.diff(cleaned_fields_vals["time"]) > 0)

    # now lets drop the iter field
    cleaned_fields_vals.pop("iter")
    cleaned_fields_vals.pop("gradnorm")
    # cleaned_fields_vals.pop("time")

    # close all plots and reset plot settings to default
    using_outside_plot = ax is not None
    if not using_outside_plot:
        fig, ax = plt.subplots(figsize=(10, 10))

    # set the scatter size to be smaller
    plt.rcParams["scatter.marker"] = "."

    # set the linewidth to be larger
    plt.rcParams["lines.linewidth"] = 6

    def _plot_cost(
        ax: plt.Axes, x_idxs: np.ndarray, reset_idxs: np.ndarray, cost_vals: np.ndarray
    ):
        # plot the cost as two separate line plots, separated by the last reset
        if len(reset_idxs) > 0:
            last_reset_idx = reset_idxs[-1]
        else:
            last_reset_idx = len(x_idxs) - 1

        first_line_idxs = x_idxs[:last_reset_idx]
        first_cost_vals = cost_vals[:last_reset_idx]

        second_line_idxs = x_idxs[last_reset_idx:]
        second_cost_vals = cost_vals[last_reset_idx:]

        plot_handle = ax.plot(
            first_line_idxs,
            first_cost_vals,
            color=color,
            label=plot_title,
            zorder=zorder,
        )  # first line
        ax.plot(
            second_line_idxs, second_cost_vals, color=color, zorder=zorder
        )  # second line

        # ax.scatter(first_line_idxs[-1], first_cost_vals[-1], color="red", zorder=2)
        # ax.scatter(second_line_idxs[0], second_cost_vals[0], color="red", zorder=2)
        ax.set_yscale("log")
        # ax.set_xscale("log")

        # turn on horizontal grid lines
        ax.yaxis.grid(True)
        return plot_handle

    # extract the info, as the data is stored as a numpy array of 2-d arrays
    val = cleaned_fields_vals["cost"]
    x_idxs = cleaned_fields_vals["time"]

    experiment_name = exp_dir.split("/")[-2]
    precon_name = exp_dir.split("/")[-1]
    print(
        f"experiment: {experiment_name}, precon: {precon_name}, max_time: {x_idxs[-1]}"
    )

    if not show_refinement_steps:
        val = val[: cora_reset_idxs[-1]]
        x_idxs = x_idxs[: cora_reset_idxs[-1]]
        cora_reset_idxs = cora_reset_idxs[:-1]

    # x_idxs = np.arange(len(val))
    # _left_shift_idxs_following_pts(x_idxs, cora_reset_idxs)

    assert isinstance(ax, plt.Axes)
    if plot_title is None:
        ax.set_title(f"{exp_dir}")
    else:
        ax.set_title(plot_title)

    # ax.set_xlabel("iteration")
    # ax.set_ylabel(name)
    # ax.set_xlim(0, len(val))

    # draw a vertical line at each reset
    for idx in cora_reset_idxs:
        if not draw_vlines:
            continue
        # red dashed line
        ax.axvline(x_idxs[idx], color="gray", linestyle="--")

        # make the vline below all other lines and markers
        ax.set_zorder(0)

    plot_handle = _plot_cost(ax, x_idxs, cora_reset_idxs, val)

    if not using_outside_plot:
        plt.show()

    return plot_handle
