import pandas as pd
import matplotlib.pyplot as plt
import os

file_dir = "/home/alan/cora/build"
# get the .csv files in the directory
files = [f for f in os.listdir(file_dir) if f.endswith(".csv")]

fstart = "single_drone_"
NAME_MAP = {
    "block_chol.csv": "block-diag Cholesky",
    "ichol.csv": "incomplete Cholesky",
    "jacobi.csv": "Jacobi",
    "block_jacobi.csv": "block Jacobi",
    "reg_chol.csv": "regularized Cholesky"
}
NAME_MAP = {fstart + k: v for k, v in NAME_MAP.items()}

min_runtime = float("inf")
plt.figure(figsize=(10, 6))
for file in files:
    file_path = os.path.join(file_dir, file)
    data = pd.read_csv(file_path)

    # the elapsed time resets periodically, we need to find the indices where it resets
    elapsed_time = data["elapsed_time"]
    elapsed_time_diff = elapsed_time.diff()
    reset_indices = elapsed_time_diff[elapsed_time_diff < 0].index

    for idx in reset_indices[::-1]:
        print(f"Resetting at index {idx}")
        prev_val = data.loc[idx - 1, "elapsed_time"]
        # make sure prev val is greater than val at idx
        assert prev_val > data.loc[idx, "elapsed_time"]
        data.loc[idx:, "elapsed_time"] += data.loc[idx - 1, "elapsed_time"]

    # Plot cost vs. runtime with y-axis in log scale
    exp_name = NAME_MAP[file]
    plt.plot(
        data["elapsed_time"],
        data["fx"],
        # marker="o",
        label=exp_name
    )

    # find the minimum runtime
    min_runtime = min(min_runtime, data["elapsed_time"].max())

# set x-axis limit as the minimum runtime
plt.xlim(0, min_runtime-1)

# thicken the lines
for line in plt.gca().lines:
    line.set_linewidth(4)

plt.xlabel("Runtime (seconds)")
plt.ylabel("Cost (fx)")
plt.yscale("log")
plt.title("Single Drone")
plt.grid(True, which="major", linestyle="--")
plt.legend()

plt.show()
