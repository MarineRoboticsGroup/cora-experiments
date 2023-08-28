from os.path import join
from utils.generate_manhattan_experiments import (
    MANHATTAN_EXPERIMENTS,
    USE_LOOP_CLOSURE,
    LOOP_CLOSURE_OPTIONS,
)
from utils.evaluate_manhattan_data import (
    make_manhattan_subopt_plots,
)
from utils.paths import MANHATTAN_DATA_DIR

if __name__ == "__main__":

    experiments = MANHATTAN_EXPERIMENTS

    # hardrive_data = "/media/alan/2aa68396-ccc1-40e6-8355-ccfb4d4e7c35/manhattan-experiments"
    # hardrive_data = "/media/alan/2aa68396-ccc1-40e6-8355-ccfb4d4e7c35/manhattan-experiments-fully-marginalized"
    hardrive_data = "/media/alan/2aa68396-ccc1-40e6-8355-ccfb4d4e7c35/manhattan-experiments-unmarginalized"
    make_manhattan_subopt_plots(base_experiment_dir=hardrive_data)

    use_all_caches = True
    for loop_closure_subdir in LOOP_CLOSURE_OPTIONS[::1]:
        use_loop_closure = loop_closure_subdir == USE_LOOP_CLOSURE
        exp_base_dir = join(MANHATTAN_DATA_DIR, loop_closure_subdir)
        # generate_manhattan_experiments(
        #     exp_base_dir,
        #     default_use_loop_closures=use_loop_closure,
        #     experiments=experiments,
        #     use_cached_experiments=True or use_all_caches,
        #     num_repeats_per_param=20,
        # )
        # gc.collect()
        # for e in experiments[::1]:
        #     problem_dir = join(exp_base_dir, e)
        #     try:
        #         solve_manhattan_problems_in_dir(
        #             problem_dir,
        #             use_cached_results=True or use_all_caches,
        #             show_animations=False,
        #         )
        #     except Exception as e:
        #         logger.error(f"Error solving problems in {problem_dir}: {e}")
        # gc.collect()
        # make_manhattan_rmse_plots(
        #     base_experiment_dir=exp_base_dir,
        #     subexperiment_types=experiments,
        #     use_cached_sweep_results=True or use_all_caches,
        #     use_cached_subexp_results=True or use_all_caches,
        # )
