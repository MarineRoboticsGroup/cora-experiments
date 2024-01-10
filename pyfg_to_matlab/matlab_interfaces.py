import scipy.io as sio
import os
import numpy as np
from data_matrix import get_full_data_matrix
from matrix_utils import get_var_idxs_from_pyfg
from constraints import get_constraints
from matrix_type_conversions import convert_torch_tensor_to_scipy_sparse
from pymanopt_helpers import (
    get_odometric_initialization,
    get_ground_truth_initialization,
    get_X_from_manifold_numpy,
)
from py_factor_graph.factor_graph import FactorGraphData
from typing import List

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


def _get_timestamps(fg: FactorGraphData) -> List[float]:
    tstamps = []
    for pose_chain in fg.pose_variables:
        for pose in pose_chain:
            tstamps.append(pose.timestamp)
    return tstamps


def export_fg_to_matlab_cora_format(fg: FactorGraphData, matlab_filepath: str) -> None:
    """Exports a factor graph to MATLAB format. This was used for generating the
    problems we worked on in CORA.

    Args:
        fg (FactorGraphData): Factor graph to export.
        matlab_filepath (str): Path to save the MATLAB file to.
    """

    file_dir = os.path.dirname(matlab_filepath)
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)

    # get Q and indices info and save to MATLAB format
    Q = get_full_data_matrix(fg)
    Q = convert_torch_tensor_to_scipy_sparse(Q)
    dim = fg.dimension
    X_odom = get_X_from_manifold_numpy(*get_odometric_initialization(fg))
    X_gt = get_X_from_manifold_numpy(*get_ground_truth_initialization(fg))
    var_idxs = get_var_idxs_from_pyfg(fg)
    rot_idxs, tran_idxs, beacon_idxs, range_idxs = var_idxs
    print(f"rot_idxs: {rot_idxs}")
    print(f"tran_idxs: {tran_idxs}")
    print(f"beacon_idxs: {beacon_idxs}")
    print(f"range_idxs: {range_idxs}")
    stacked_constraints = get_constraints(fg)
    stacked_constraints = convert_torch_tensor_to_scipy_sparse(stacked_constraints)
    timestamps = np.array(_get_timestamps(fg))
    assert all([not tstamp is None for tstamp in timestamps])
    mats = {
        "Q": Q,
        "stacked_constraints": stacked_constraints,
        "rot_idxs": rot_idxs,
        "tran_idxs": tran_idxs,
        "beacon_idxs": beacon_idxs,
        "range_idxs": range_idxs,
        "num_poses": fg.num_poses + int(fg.has_priors),
        "num_loop_closures": fg.num_loop_closures,
        "num_landmarks": fg.num_landmarks,
        "num_range_measurements": fg.num_range_measurements,
        "num_pose_priors": fg.num_pose_priors,
        "num_beacon_priors": fg.num_landmark_priors,
        "num_robots": fg.num_robots,
        "dim": dim,
        "X_odom": X_odom,
        "X_gt": X_gt,
        "timestamps": timestamps,
    }
    sio.savemat(matlab_filepath, mats)
    logger.info(f"Saved data to {matlab_filepath}")
