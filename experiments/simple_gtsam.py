import os
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

    from os.path import join, isfile
    from typing import List, Optional
    from py_factor_graph.factor_graph import FactorGraphData
    from py_factor_graph.utils.solver_utils import save_to_tum, save_results_to_file
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

        fname = "single_drone.pyfg"
        fname = "plaza1.pyfg"
        fname = "plaza2.pyfg"
        fname = "tiers.pyfg"

        standard_fnames = [
            # "plaza1.pyfg",
            # "plaza2.pyfg",
            # "single_drone.pyfg",
            "tiers.pyfg",
        ]
        mrclam_exps = [
            "mrclam2",
            # "mrclam3a",
            # "mrclam3b",
            "mrclam4",
            # "mrclam5a",
            # "mrclam5b",
            # "mrclam5c",
            "mrclam6",
            "mrclam7",
        ]
        mrclam_fnames = [f"mrclam/range_and_rpm/{exp}/{exp}.pyfg" for exp in mrclam_exps]
        fnames = mrclam_fnames
        fnames = standard_fnames

        strategies = [GTSAM_GT_POSE_RAND_LAND]
        strategies = [GTSAM_RAND_POSE_GT_LAND]
        # strategies = STRATEGIES

        for fname in fnames:

            data_dir = "/home/alan/cora/examples/data"
            fpath = join(data_dir, fname)
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

                gtsam_result = solve_mle_gtsam(pyfg, solver_params, solver=LM_SOLVER)
                print(
                    f"Problem: {fname}, strategy: {init_strategy}, time: {gtsam_result.total_time:.2f} secs"
                )
