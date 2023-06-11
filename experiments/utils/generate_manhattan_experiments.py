import math
from os import makedirs
from os.path import join, isdir, isfile
from typing import List
from py_factor_graph.io.pickle_file import parse_pickle_file
from manhattan.simulator.simulator import ManhattanSimulator, SimulationParams
from attrs import field, define
from itertools import product
import numpy as np

from .paths import REPO_BASE_DIR

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


# insert the pyfg_to_matlab directory into the path
import sys

sys.path.insert(0, REPO_BASE_DIR)
from pyfg_to_matlab.matlab_interfaces import export_fg_to_matlab_cora_format

MANHATTAN_DEFAULT_EXP = "default"
SWEEP_NUM_ROBOTS = "sweep_num_robots"
SWEEP_RANGE_COV = "sweep_range_cov"
SWEEP_NUM_RANGES = "sweep_num_ranges"
SWEEP_NUM_POSES = "sweep_num_poses"
SWEEP_NUM_BEACONS = "sweep_num_beacons"
SWEEP_NUM_LOOP_CLOSURES = "sweep_num_loop_closures"

MANHATTAN_EXPERIMENTS = [
    SWEEP_NUM_ROBOTS,
    SWEEP_RANGE_COV,
    SWEEP_NUM_RANGES,
    SWEEP_NUM_POSES,
    SWEEP_NUM_BEACONS,
    SWEEP_NUM_LOOP_CLOSURES,
]

EXPERIMENT_TRAILING_STR = {
    SWEEP_NUM_ROBOTS: "robots",
    SWEEP_NUM_BEACONS: "beacons",
    SWEEP_NUM_LOOP_CLOSURES: "loopClosures",
    SWEEP_NUM_POSES: "poses",
    SWEEP_NUM_RANGES: "ranges",
    SWEEP_RANGE_COV: "rangeStddev",
}


@define
class ManhattanExpParam:
    num_robots: int = field()
    num_beacons: int = field()
    total_num_poses: int = field()
    num_range_measurements: int = field()
    num_loop_closures: int = field()
    range_cov: float = field()

    def get_experiment_param_as_string(self, exp_name: str) -> str:
        assert exp_name in MANHATTAN_EXPERIMENTS
        trailing_string = EXPERIMENT_TRAILING_STR[exp_name]
        if exp_name == "default":
            return "default"
        elif exp_name == "sweep_num_robots":
            return f"{self.num_robots}{trailing_string}"
        elif exp_name == "sweep_range_cov":
            return f"{np.sqrt(self.range_cov):.3}{trailing_string}"
        elif exp_name == "sweep_num_ranges":
            return f"{self.num_range_measurements}{trailing_string}"
        elif exp_name == "sweep_num_poses":
            return f"{self.total_num_poses}{trailing_string}"
        elif exp_name == "sweep_num_beacons":
            return f"{self.num_beacons}{trailing_string}"
        elif exp_name == "sweep_num_loop_closures":
            return f"{self.num_loop_closures}{trailing_string}"
        else:
            raise ValueError(f"Unknown experiment: {exp_name}")


def run_manhattan_simulator(exp_save_dir: str, exp_params: ManhattanExpParam) -> str:
    grid_len = 100

    num_robots = exp_params.num_robots
    num_beacons = exp_params.num_beacons
    total_num_poses = exp_params.total_num_poses
    num_range_measurements = exp_params.num_range_measurements
    num_loop_closures = exp_params.num_loop_closures
    range_cov = exp_params.range_cov

    # set the range probability such that the expected (in probabilistic sense)
    # number of ranges is 'num_range_measurements'
    if num_range_measurements > 0:
        total_range_opportunities = (num_robots - 1) * num_robots / 2
        average_range_opportunities = total_range_opportunities / num_robots
        range_prob = min(
            1.0, num_range_measurements / total_num_poses / average_range_opportunities
        )
    else:
        range_prob = 0.0

    loop_closure_prob = min(float(num_loop_closures) / total_num_poses, 1.0)

    dist_stddev = math.sqrt(range_cov)
    pos_stddev = 0.05
    theta_stddev = 0.0005
    seed_cnt = 2
    seed = (seed_cnt + 1) * 9999
    num_timesteps = int(total_num_poses / num_robots) - 1
    use_gt_measurements = False

    sim_args = SimulationParams(
        num_robots=num_robots,
        num_beacons=num_beacons,
        grid_shape=(grid_len, grid_len),
        y_steps_to_intersection=2,
        x_steps_to_intersection=5,
        cell_scale=1.0,
        range_sensing_prob=range_prob,
        range_sensing_radius=1000.0,
        false_range_data_association_prob=0.0,
        outlier_prob=0.0,
        max_num_loop_closures=99999,
        loop_closure_prob=loop_closure_prob,
        loop_closure_radius=1000.0,
        false_loop_closure_prob=0.0,
        range_stddev=dist_stddev,
        odom_x_stddev=pos_stddev,
        odom_y_stddev=pos_stddev,
        odom_theta_stddev=theta_stddev,
        loop_x_stddev=pos_stddev,
        loop_y_stddev=pos_stddev,
        loop_theta_stddev=theta_stddev,
        debug_mode=False,
        seed_num=seed,
        groundtruth_measurements=use_gt_measurements,
    )
    sim = ManhattanSimulator(sim_args)

    show_animation = False
    if show_animation:
        # sim.plot_grid()
        sim.plot_beacons()

    for _ in range(num_timesteps):
        sim.random_step()

        if show_animation:
            sim.plot_robot_states()
            sim.show_plot(animation=True)

    if show_animation:
        sim.close_plot()

    filename = "factor_graph"
    save_filepath = sim.save_simulation_data(
        exp_save_dir, format="pickle", filename=filename
    )
    logger.info(f"Saved simulation data to {save_filepath}")

    return save_filepath


def generate_manhattan_experiments(
    base_dir: str,
    experiments: List[str] = MANHATTAN_EXPERIMENTS,
    use_cached_experiments: bool = False,
):
    if not isdir(base_dir):
        makedirs(base_dir)

    for experiment in experiments:
        # default values
        num_robots_list = [4]
        range_cov_list = [(0.5**2)]
        num_ranges_list = [500]
        num_poses_list = [4000]
        num_beacons_list = [2]
        num_loop_closures_list = [
            int(num_poses_list[0] * 0.05)
        ]  # loop closures by % of poses

        if experiment == "default":
            logger.info("Skipping default experiment")
            continue
        elif experiment == "sweep_num_robots":
            # sweep from 2 to 20 robots
            min_num_robots = 2
            max_num_robots = 20
            step_size = 2
            num_robots_list = list(range(min_num_robots, max_num_robots + 1, step_size))
        elif experiment == "sweep_range_cov":
            range_stddev_lower_bound = 0.2
            range_stddev_upper_bound = 2
            range_stddev_step_size = 0.2
            range_cov_array = np.arange(
                range_stddev_lower_bound,
                range_stddev_upper_bound + range_stddev_step_size,
                range_stddev_step_size,
            )
            range_cov_list = [(x**2) for x in range_cov_array]
        elif experiment == "sweep_num_ranges":
            num_ranges_lower_bound = 100
            num_ranges_upper_bound = 2000
            num_ranges_step_size = (
                num_ranges_upper_bound - num_ranges_lower_bound
            ) // 10
            num_ranges_list = list(
                range(
                    num_ranges_lower_bound,
                    num_ranges_upper_bound + 1,
                    num_ranges_step_size,
                )
            )
        elif experiment == "sweep_num_poses":
            min_num_poses = 1000
            max_num_poses = 10000
            step_size = 500
            num_poses_list = list(range(min_num_poses, max_num_poses + 1, step_size))
        elif experiment == "sweep_num_beacons":
            min_num_beacons = 0
            max_num_beacons = 10
            step_size = 2
            num_beacons_list = list(
                range(min_num_beacons, max_num_beacons + step_size, step_size)
            )
        elif experiment == "sweep_num_loop_closures":
            min_fraction_loop_closures = 0.0
            max_fraction_loop_closures = 0.15
            fraction_step_size = 0.03
            num_loop_closures_list = list(
                np.arange(
                    min_fraction_loop_closures,
                    max_fraction_loop_closures + fraction_step_size,
                    fraction_step_size,
                )
            )
        else:
            raise ValueError(f"Unknown experiment: {experiment}")

        param_sweep_base_dir = join(base_dir, experiment)

        for params in product(
            num_robots_list,
            num_beacons_list,
            num_poses_list,
            num_ranges_list,
            num_loop_closures_list,
            range_cov_list,
        ):
            (
                num_robots,
                num_beacons,
                total_num_poses,
                num_ranges,
                num_loop_closures,
                range_cov,
            ) = params
            packaged_params = ManhattanExpParam(
                num_robots=num_robots,
                num_beacons=num_beacons,
                total_num_poses=total_num_poses,
                num_range_measurements=num_ranges,
                num_loop_closures=num_loop_closures,
                range_cov=range_cov,
            )

            # get the specific directory name for this sub-experiment
            sweep_subexp_dirname = packaged_params.get_experiment_param_as_string(
                experiment
            )
            subexp_dir = join(param_sweep_base_dir, sweep_subexp_dirname)

            # run the simulator and save the factor graph to a pickle file
            expected_saved_fg_fpath = join(subexp_dir, "factor_graph.pickle")
            if use_cached_experiments and isfile(expected_saved_fg_fpath):
                logger.warning(
                    "Found cached manhattan experiment - not generating a new one!"
                )
                factor_graph_file = expected_saved_fg_fpath
            else:
                factor_graph_file = run_manhattan_simulator(subexp_dir, packaged_params)

            # convert the data to our matlab format
            fg = parse_pickle_file(factor_graph_file)
            expected_saved_matlab_fpath = join(subexp_dir, "factor_graph.mat")
            if use_cached_experiments and isfile(expected_saved_matlab_fpath):
                logger.warning(
                    "Found cached MATLAB version of experiment - not generating a new one!"
                )
            else:
                mat_file = factor_graph_file.replace(".pickle", ".mat")
                export_fg_to_matlab_cora_format(fg, matlab_filepath=mat_file)
            print()


if __name__ == "__main__":
    from .paths import REPO_BASE_DIR

    manhattan_data_dir = join(REPO_BASE_DIR, "data", "manhattan")
    generate_manhattan_experiments(manhattan_data_dir)
