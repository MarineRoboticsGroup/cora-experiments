import math
from os import makedirs
from os.path import join, isdir, isfile
from typing import List
from py_factor_graph.io.pyfg_text import save_to_pyfg_text
from manhattan.simulator.simulator import ManhattanSimulator, SimulationParams
from attrs import field, define
from itertools import product
import numpy as np

from .logging_utils import logger

# insert the pyfg_to_matlab directory into the path

MANHATTAN_DEFAULT_EXP = "default"
SWEEP_NUM_ROBOTS = "sweep_num_robots"
SWEEP_RANGE_COV = "sweep_range_cov"
SWEEP_NUM_RANGES = "sweep_num_ranges"
SWEEP_NUM_POSES = "sweep_num_poses"
SWEEP_NUM_BEACONS = "sweep_num_beacons"
SWEEP_PCT_LOOP_CLOSURES = "sweep_pct_loop_closures"

USE_LOOP_CLOSURE = "default_with_loop_closures"
NO_LOOP_CLOSURE = "default_no_loop_closures"
LOOP_CLOSURE_OPTIONS = [USE_LOOP_CLOSURE, NO_LOOP_CLOSURE]

MANHATTAN_EXPERIMENTS = [
    SWEEP_NUM_ROBOTS,
    # SWEEP_RANGE_COV,
    SWEEP_NUM_RANGES,
    # SWEEP_NUM_POSES,
    SWEEP_NUM_BEACONS,
    # SWEEP_PCT_LOOP_CLOSURES,
]

EXPERIMENT_PLOT_TITLES = {
    SWEEP_NUM_ROBOTS: "Number of Robots",
    SWEEP_RANGE_COV: "Range Covariance",
    SWEEP_NUM_RANGES: "Number of Ranges",
    SWEEP_NUM_POSES: "Number of Poses",
    SWEEP_NUM_BEACONS: "Number of Beacons",
    SWEEP_PCT_LOOP_CLOSURES: "Loop Closure Probability",
}


EXPERIMENT_TRAILING_STR = {
    SWEEP_NUM_ROBOTS: "robots",
    SWEEP_NUM_BEACONS: "beacons",
    SWEEP_PCT_LOOP_CLOSURES: "loopClosures",
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
    pct_loop_closures: float = field()
    range_cov: float = field()
    seed: int = field(default=42)

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
        elif exp_name == "sweep_pct_loop_closures":
            return f"{self.pct_loop_closures}{trailing_string}"
        else:
            raise ValueError(f"Unknown experiment: {exp_name}")


def run_manhattan_simulator(exp_save_dir: str, exp_params: ManhattanExpParam) -> str:
    grid_len = 100

    num_robots = exp_params.num_robots
    num_beacons = exp_params.num_beacons
    total_num_poses = exp_params.total_num_poses
    num_range_measurements = exp_params.num_range_measurements
    pct_loop_closures = exp_params.pct_loop_closures
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

    loop_closure_prob = float(pct_loop_closures)

    dist_stddev = math.sqrt(range_cov)
    pos_stddev = 0.05
    theta_stddev = 0.0005
    seed = (exp_params.seed + 1) * 9999
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

    filename = "factor_graph.pyfg"
    if not isdir(exp_save_dir):
        makedirs(exp_save_dir)
    save_filepath = join(exp_save_dir, filename)
    save_to_pyfg_text(sim._factor_graph, save_filepath)

    return save_filepath


from multiprocessing import Pool, cpu_count


def process_experiment(args):
    params, trial_num, param_sweep_base_dir, experiment, use_cached_experiments = args
    (
        num_robots,
        num_beacons,
        total_num_poses,
        num_ranges,
        pct_loop_closures,
        range_cov,
    ) = params
    packaged_params = ManhattanExpParam(
        num_robots=num_robots,
        num_beacons=num_beacons,
        total_num_poses=total_num_poses,
        num_range_measurements=num_ranges,
        pct_loop_closures=pct_loop_closures,
        range_cov=range_cov,
        seed=trial_num * 999,
    )
    sweep_subexp_dirname = packaged_params.get_experiment_param_as_string(experiment)
    subexp_dir = join(param_sweep_base_dir, sweep_subexp_dirname, f"trial{trial_num}")
    expected_saved_fg_fpath = join(subexp_dir, "factor_graph.pyfg")
    if use_cached_experiments and isfile(expected_saved_fg_fpath):
        logger.warning(f"Using cached experiment file - {expected_saved_fg_fpath}")
    else:
        run_manhattan_simulator(subexp_dir, packaged_params)


def generate_manhattan_experiments(
    base_dir: str,
    default_use_loop_closures: bool = True,
    experiments: List[str] = MANHATTAN_EXPERIMENTS,
    use_cached_experiments: bool = False,
    num_repeats_per_param: int = 1,
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
        if default_use_loop_closures:
            pct_loop_closures_list = [0.05]  # loop closures by % of poses
        else:
            pct_loop_closures_list = [0.0]

        if experiment == "default":
            logger.info("Skipping default experiment")
            continue
        elif experiment == "sweep_num_robots":
            # sweep from 1 to 20 robots
            min_num_robots = 2
            max_num_robots = 20
            step_size = 3
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
        elif experiment == "sweep_pct_loop_closures":
            min_fraction_loop_closures = 0.0
            max_fraction_loop_closures = 0.15
            fraction_step_size = 0.03
            pct_loop_closures_list = list(
                np.arange(
                    min_fraction_loop_closures,
                    max_fraction_loop_closures + fraction_step_size,
                    fraction_step_size,
                )
            )
        else:
            raise ValueError(f"Unknown experiment: {experiment}")

        param_sweep_base_dir = join(base_dir, experiment)

        args_list = [
            (
                params,
                trial_num,
                param_sweep_base_dir,
                experiment,
                use_cached_experiments,
            )
            for params in product(
                num_robots_list,
                num_beacons_list,
                num_poses_list,
                num_ranges_list,
                pct_loop_closures_list,
                range_cov_list,
            )
            for trial_num in range(num_repeats_per_param)
        ]
        with Pool(cpu_count() - 2) as p:
            p.map(process_experiment, args_list)

        # for params in product(
        #     num_robots_list,
        #     num_beacons_list,
        #     num_poses_list,
        #     num_ranges_list,
        #     pct_loop_closures_list,
        #     range_cov_list,
        # ):
        #     (
        #         num_robots,
        #         num_beacons,
        #         total_num_poses,
        #         num_ranges,
        #         pct_loop_closures,
        #         range_cov,
        #     ) = params
        #     for trial_num in range(num_repeats_per_param):
        #         packaged_params = ManhattanExpParam(
        #             num_robots=num_robots,
        #             num_beacons=num_beacons,
        #             total_num_poses=total_num_poses,
        #             num_range_measurements=num_ranges,
        #             pct_loop_closures=pct_loop_closures,
        #             range_cov=range_cov,
        #             seed=trial_num * 999,
        #         )

        #         # get the specific directory name for this sub-experiment (e.g., "4robots")
        #         sweep_subexp_dirname = packaged_params.get_experiment_param_as_string(
        #             experiment
        #         )

        #         subexp_dir = join(
        #             param_sweep_base_dir, sweep_subexp_dirname, f"trial{trial_num}"
        #         )

        #         # run the simulator and save the factor graph to a pickle file
        #         expected_saved_fg_fpath = join(subexp_dir, "factor_graph.pyfg")
        #         if use_cached_experiments and isfile(expected_saved_fg_fpath):
        #             logger.warning(f"Using cached experiment file - {expected_saved_fg_fpath}")
        #         else:
        #             run_manhattan_simulator(
        #                 subexp_dir, packaged_params
        #             )
