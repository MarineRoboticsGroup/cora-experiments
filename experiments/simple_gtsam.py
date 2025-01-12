import os
from utils.paths import DATA_DIR

from threadpoolctl import threadpool_limits

# Set environment variables to limit the number of threads to 1
os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP threads
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS threads
os.environ["MKL_NUM_THREADS"] = "1"  # MKL threads
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NumExpr threads
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # Accelerate framework threads (macOS)
os.environ["OPENMP"] = "1"

# Additionally, set the maximum number of parallel processes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["GOMP_CPU_AFFINITY"] = "0"  # GNU OpenMP
os.environ["KMP_AFFINITY"] = "norespect,granularity=fine,compact,1,0"  # Intel OpenMP
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_WARNINGS"] = "1"


# Use threadpoolctl to limit threads during the execution
with threadpool_limits(limits=1):

    from os.path import join
    from py_factor_graph.io.pyfg_text import read_from_pyfg_text
    from ra_slam.solve_mle_gtsam import solve_mle_gtsam
    from ra_slam.utils.gtsam_utils import GtsamSolverParams
    from ra_slam.utils.solver_utils import LM_SOLVER

    GTSAM_GT_POSE_GT_LAND = "gtsam_gt_pose_gt_landmarks"
    GTSAM_GT_POSE_RAND_LAND = "gtsam_gt_pose_random_landmarks"
    GTSAM_RAND_POSE_RAND_LAND = "gtsam_random_pose_random_landmarks"
    GTSAM_RAND_POSE_GT_LAND = "gtsam_random_pose_gt_landmarks"
    STRATEGIES = [
        GTSAM_GT_POSE_GT_LAND,
        GTSAM_GT_POSE_RAND_LAND,
        GTSAM_RAND_POSE_RAND_LAND,
        GTSAM_RAND_POSE_GT_LAND,
    ]

    if __name__ == "__main__":

        standard_exp_subdirs = [
            # "plaza/plaza1",
            # "plaza/plaza2",
            # "single_drone",
            # "tiers",
            # "marine",
            "outfinite",
        ]
        mrclam_subdirs = [
            "mrclam2",
            "mrclam4",
            "mrclam6",
            "mrclam7",
        ]

        strategies = [GTSAM_GT_POSE_RAND_LAND]
        strategies = [GTSAM_RAND_POSE_GT_LAND]
        strategies = [GTSAM_GT_POSE_GT_LAND]
        # strategies = STRATEGIES

        fnames = mrclam_subdirs
        fnames = standard_exp_subdirs

        for fname in fnames:

            fpath = join(DATA_DIR, fname, "factor_graph.pyfg")
            # gt_files = [join(DATA_DIR, fname, f"gt_traj_{letter}.tum") for letter in "ABC"]
            gt_files = None
            pyfg = read_from_pyfg_text(fpath)

            assert pyfg.all_variables_have_factors(), "All variables must have factors"

            for init_strategy in strategies:
                init = "compose"
                custom_init_file = None
                if init_strategy == GTSAM_GT_POSE_GT_LAND:
                    landmark_init = "gt"
                    gt_start = True
                elif init_strategy == GTSAM_GT_POSE_RAND_LAND:
                    gt_start = True
                    landmark_init = "random"
                elif init_strategy == GTSAM_RAND_POSE_RAND_LAND:
                    gt_start = False
                    landmark_init = "random"
                elif init_strategy == GTSAM_RAND_POSE_GT_LAND:
                    gt_start = False
                    landmark_init = "gt"
                else:
                    raise ValueError(f"Invalid init strategy: {init_strategy}")

                solver_params = GtsamSolverParams(
                    init_technique=init,
                    landmark_init=landmark_init,
                    custom_init_file=custom_init_file,
                    init_translation_perturbation=None,
                    init_rotation_perturbation=None,
                    start_at_gt=gt_start,
                )

                gtsam_result = solve_mle_gtsam(
                    pyfg, solver_params, solver=LM_SOLVER, return_all_iterates=True
                )
                if len(gtsam_result) > 1:
                    print(
                        f"Problem: {fname}, strategy: {init_strategy}, time: {gtsam_result[0].total_time:.2f} secs"
                    )
                    from py_factor_graph.utils.plot_utils import visualize_solution

                    assert isinstance(gtsam_result, list)

                    # save gtsam_result
                    limits = [res.limits for res in gtsam_result]
                    xlims = [lim[0] for lim in limits]
                    ylims = [lim[1] for lim in limits]
                    xmin = min([lim[0] for lim in xlims])
                    xmax = max([lim[1] for lim in xlims])
                    ymin = min([lim[0] for lim in ylims])
                    ymax = max([lim[1] for lim in ylims])

                    gtsam_result = gtsam_result[::2]

                    for i, res in enumerate(gtsam_result):
                        fname = f"iter_{i:03d}.png"
                        if solver_params.start_at_gt:
                            start_name = "gt_start"
                        else:
                            start_name = "random_start"
                        visualize_solution(
                            res,
                            gt_files=gt_files,
                            xlim=(xmin, xmax),
                            ylim=(ymin, ymax),
                            save_path=f"/tmp/gtsam/{init_strategy}/{fname}",
                            show=False,
                        )
                else:
                    print(
                        f"Problem: {fname}, strategy: {init_strategy}, time: {gtsam_result.total_time:.2f} secs"
                    )
