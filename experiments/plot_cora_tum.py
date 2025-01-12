from typing import List, Optional
import os
from evo.tools import file_interface
import evo.core.trajectory as et
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


def get_xy_from_tum(
    tum_file: str, align_traj: Optional[et.PoseTrajectory3D] = None
) -> np.ndarray:
    traj = file_interface.read_tum_trajectory_file(tum_file)
    if align_traj is not None:
        traj.align(align_traj)
        # traj.align_origin(align_traj)

    xyz = traj.positions_xyz
    x, y, z = zip(*xyz)
    return np.array([x, y]).T


def visualize_solution(
    solution_iterates: List[np.ndarray],
    gt_traj: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Visualizes the solution.

    Args:
        solution (SolverResults): the solution.
        gt_traj (str): the path to the groundtruth trajectory.
    """
    import cartopy.crs as ccrs
    import cartopy.io.img_tiles as cimgt
    import pymap3d as pm

    base_lon = -71.08699035644531
    base_lat = 42.35807418823242
    tiler_style = "satellite"
    tiler_style = "terrain"
    tiler_style = "street"
    # tiler_style = "only_streets"
    tiler = cimgt.GoogleTiles(style=tiler_style, cache=False)
    tile_zoom = 20

    # draw the x/y positions of the robot
    def convert_latlon_to_xy(lat, lon):
        return pm.geodetic2enu(lat, lon, 0.0, base_lat, base_lon, 0.0)

    def convert_xy_to_latlon(x, y):
        return pm.enu2geodetic(x, y, 0.0, base_lat, base_lon, 0.0)

    # make figure the full screen size
    fig, ax = plt.subplots(
        figsize=(20, 10),
        subplot_kw={"projection": tiler.crs},
    )
    ax.add_image(tiler, tile_zoom)
    ax.set_extent(
        [
            -71.08999107262918,
            -71.08631488034362,
            42.356811248659724,
            42.35857418823244,
        ]
    )

    gt_lat, gt_lon, _ = convert_xy_to_latlon(gt_traj[:, 0], gt_traj[:, 1])
    ax.plot(
        gt_lon,
        gt_lat,
        color="gray",
        linestyle="dashed",
        transform=ccrs.PlateCarree(),
        label="Groundtruth",
        linewidth=4,
    )

    traj_latlon = [
        convert_xy_to_latlon(traj[:, 0], traj[:, 1]) for traj in solution_iterates[:-1]
    ]
    traj_line = ax.plot(
        [],
        [],
        color="lime",
        transform=ccrs.PlateCarree(),
        label="CORA",
        linewidth=4,
    )[0]

    ax.legend()

    # that the lines are drawn in the order they are added
    # colors = ["blue", "lime", "red"]
    # line_cnt = 0
    # for plt_handle in plt.gca().get_lines()[1:2]:
    #     plt_handle.set_color(colors[line_cnt])
    #     plt_handle.set_zorder(len(colors) - line_cnt)
    #     line_cnt += 1

    # hide the grid
    plt.grid(False)

    # set the background to white
    plt.gca().set_facecolor("white")

    def update(num):
        traj_line.set_data(traj_latlon[num][1], traj_latlon[num][0])
        return [traj_line]

    ani = animation.FuncAnimation(
        fig, update, frames=len(traj_latlon), interval=2, blit=True, repeat=False
    )

    # plt.show()

    # save as a gif
    # ani.save(save_path, writer="imagemagick", fps=10, dpi=300)
    ani.save(save_path, writer="pillow", fps=10, dpi=100)

    print(f"Saved to {save_path}")

    plt.close(fig)


if __name__ == "__main__":
    from os.path import join

    soln_dir = "/tmp/_iterates/"
    # get every subdirectory in the directory
    # subdirs = [f.path for f in os.scandir(soln_dir) if f.is_dir()]
    subdirs = [join(soln_dir, f"cora_{i}") for i in range(1, 196)]
    subdirs = subdirs[::2]
    # subdirs = subdirs[-3:]

    # skip the last 10 subdirs except the last one
    subdirs = subdirs[:-10] + subdirs[-1:]

    # copy subdirs onto itself so it loops
    # subdirs = subdirs + subdirs

    print(f"Found {len(subdirs)} subdirectories")
    gt_file = "/home/alan/range-only-slam-mission-control/cora-experiments/data/marine/gt_traj_A.tum"
    gt_xy_vals = get_xy_from_tum(gt_file)
    gt_traj = file_interface.read_tum_trajectory_file(gt_file)

    traj_files = [os.path.join(subdir, "cora_0.tum") for subdir in subdirs]
    xy_vals = [get_xy_from_tum(f, align_traj=gt_traj) for f in traj_files]

    visualize_solution(
        xy_vals,
        gt_xy_vals,
        save_path="/tmp/cora_new2.gif",
    )
