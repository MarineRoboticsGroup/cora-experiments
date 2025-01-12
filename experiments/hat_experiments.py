import numpy as np
from py_factor_graph.io.pyfg_text import read_from_pyfg_text
from py_factor_graph.factor_graph import FactorGraphData

from os.path import join
from evo.tools import plot

from utils.run_experiments import run_experiments, ExperimentConfigs
from utils.paths import DATA_DIR

from utils.logging_utils import get_logger

logger = get_logger(__name__)


def draw_groundtruth(pyfg: FactorGraphData):
    # draw groundtruth

    import cartopy.crs as ccrs
    import cartopy.io.img_tiles as cimgt
    import pymap3d as pm
    import matplotlib.pyplot as plt

    ##### pretty params
    plt.rcParams["lines.linewidth"] = 4
    plt.rcParams["axes.labelsize"] = 30
    plt.rcParams["xtick.labelsize"] = 30
    plt.rcParams["ytick.labelsize"] = 30
    plt.rcParams["legend.fontsize"] = 20
    plt.rcParams["legend.loc"] = "lower right"
    plt.rcParams["grid.color"] = "gray"
    plt.rcParams["grid.color"] = "lightgray"
    plt.rcParams["figure.subplot.hspace"] = 0.01
    plt.rcParams["axes.edgecolor"] = "gray"

    ##### set up plot with satellite overlay
    tiler_style = "satellite"
    tiler = cimgt.GoogleTiles(style=tiler_style, cache=False)
    tile_zoom = 20
    fig, ax = plt.subplots(
        figsize=(20, 10),
        subplot_kw={"projection": tiler.crs},
    )
    ax.set_extent(
        [
            -71.08999107262918,
            -71.08631488034362,
            42.356811248659724,
            42.35857418823244,
        ]
    )
    ax.add_image(tiler, tile_zoom)

    #### draw the actual information
    base_lon = -71.08699035644531
    base_lat = 42.35807418823242

    def convert_latlon_to_xy(lat, lon):
        return pm.geodetic2enu(lat, lon, 0.0, base_lat, base_lon, 0.0)

    def convert_xy_to_latlon(x, y):
        return pm.enu2geodetic(x, y, 0.0, base_lat, base_lon, 0.0)

    # robot position
    positions = [pose.true_position for pose in pyfg.pose_variables[0]]
    latlons = [convert_xy_to_latlon(pos[0], pos[1]) for pos in positions]
    rob_lats, rob_lons, _ = zip(*latlons)

    # landmark positions
    landmark_positions = [var.true_position for var in pyfg.landmark_variables]
    latlons = [convert_xy_to_latlon(pos[0], pos[1]) for pos in landmark_positions]
    lats, lons, _ = zip(*latlons)
    ax.plot(
        lons,
        lats,
        color="orange",
        label="Ground Truth (GPS Vehicle)",
        transform=ccrs.PlateCarree(),
        linewidth=1,
    )

    ax.plot(
        rob_lons,
        rob_lats,
        color="green",
        label="Ground Truth (Dead Reckoning Vehicle)",
        transform=ccrs.PlateCarree(),
    )

    ax.legend()

    # make the scatter plot bigger in the legend
    for handle in ax.get_legend().legendHandles:
        handle._sizes = [200]

    plt.show()


if __name__ == "__main__":
    run_noisy_experiments = False  # whether to add artificial noise to the problem and also run CORA on the noisy problems

    MARINE_DIR = join(DATA_DIR, "marine")

    BASE_EXPERIMENT = {
        MARINE_DIR: np.array([0.0, 0.0]),
    }
    EXPERIMENTS = BASE_EXPERIMENT

    marine_experiment_fpath = join(MARINE_DIR, "factor_graph.pyfg")
    pyfg = read_from_pyfg_text(marine_experiment_fpath)

    # draw_groundtruth(pyfg)

    # for range_measure in pyfg.range_measurements:
    #     precision = 4.0
    #     range_measure.stddev = 1.0 / np.sqrt(precision)
    #     range_measure.dist *= (1.0)
    #     range_measure.dist += 1.0

    # for odom_chain in pyfg.odom_measurements:
    #     for odom_measure in odom_chain:
    #         odom_measure.translation_precision = 10.0
    #         odom_measure.rotation_precision = 20.0

    # save_to_pyfg_text(pyfg, join(MARINE_DIR, "factor_graph.pyfg"))

    exp_config = ExperimentConfigs(
        run_experiments_with_added_noise=False,
        use_cached_problems=True,
        animate_trajs=False,
        run_cora=False,
        solve_marginalized_problem=True,
        show_solver_animation=True,
        show_gt_cora_animation=True,
        look_for_cached_cora_solns=True,
        perform_evaluation=True,
        use_cached_trajs=True,
        desired_plot_modes=[plot.PlotMode.xy],
        overlay_river_map=True,
    )
    run_experiments(pyfg, EXPERIMENTS, exp_config)
