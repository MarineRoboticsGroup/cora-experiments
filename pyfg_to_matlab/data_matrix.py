from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.measurements import (
    POSE_MEASUREMENT_TYPES,
    PoseMeasurement2D,
    PoseMeasurement3D,
    POSE_LANDMARK_MEASUREMENT_TYPES,
    PoseToLandmarkMeasurement2D,
    PoseToLandmarkMeasurement3D,
)
from py_factor_graph.priors import (
    PosePrior2D,
    PosePrior3D,
    LandmarkPrior2D,
    LandmarkPrior3D,
)
from py_factor_graph.utils.solver_utils import VariableValues
import numpy as np
from typing import List, Dict, Tuple

from matrix_utils import (
    _get_beacon_variable_idxs,
    _get_pose_variable_idxs,
    _get_dist_variable_idxs,
    PoseVariableIndices,
)


import torch
import torch.sparse

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


def get_gt_variable_values(fg: FactorGraphData) -> VariableValues:
    """Returns the ground-truth variable values for the given problem.

    Args:
        fg (FactorGraphData): the factor graph data describing the problem

    Returns:
        VariableValues: the ground-truth variable values
    """
    pose_vals = {}
    for pose_chain in fg.pose_variables:
        for pose_var in pose_chain:
            pose_vals[pose_var.name] = pose_var.transformation_matrix

    beacon_vals = {}
    for beacon_var in fg.landmark_variables:
        beacon_vals[beacon_var.name] = np.array(beacon_var.true_position)

    dist_vals = {}
    dim = fg.dimension
    for dist_measure in fg.range_measurements:
        var1_name, var2_name = dist_measure.association
        pose1 = pose_vals[var1_name]
        t1 = pose1[:dim, dim]

        if var2_name in pose_vals:
            pose2 = pose_vals[var2_name]
            t2 = pose2[:dim, dim]
        else:
            t2 = beacon_vals[var2_name]

        diff = t1 - t2
        dist_vals[dist_measure.association] = diff / np.linalg.norm(diff)

    gt_var_vals = VariableValues(fg.dimension, pose_vals, beacon_vals, dist_vals)
    return gt_var_vals


def get_full_data_matrix(fg: FactorGraphData) -> torch.Tensor:
    """Computes the full data matrix for the given problem.

    Q = block_diag(Q_pose, Q_range)

    Args:
        fg (FactorGraphData): the factor graph data describing the problem

    Returns:
        SparseMatrixTypes: the full data matrix
    """
    # get the pose data matrix
    Q_pose: torch.Tensor = _get_pose_data_matrix(fg)
    Q_pose_vals = Q_pose.values()
    Q_pose_idxs = Q_pose.indices()

    # get the range data matrix
    Q_range_vals, Q_range_rows, Q_range_cols, Q_range_shape = _get_range_data_matrix(
        fg, idx_offset=Q_pose.shape[0]
    )
    Q_range_idxs = torch.LongTensor([Q_range_rows, Q_range_cols])

    # combine the pose and range data matrices
    Q_idxs = torch.cat([Q_pose_idxs, Q_range_idxs], dim=1)
    Q = torch.sparse_coo_tensor(
        indices=Q_idxs,
        values=torch.cat([Q_pose_vals, torch.DoubleTensor(Q_range_vals)]),
        size=Q_range_shape,
    )

    full_matrix_dim = _get_full_data_matrix_dim(fg)
    assert Q.shape == (
        full_matrix_dim,
        full_matrix_dim,
    ), f"Q.shape = {Q.shape} but should be {(full_matrix_dim, full_matrix_dim)}"

    return Q


###### HELPER FUNCTIONS FOR THE PUBLIC FUNCTIONS ######


def _get_pose_data_matrix_dim(fg: FactorGraphData) -> int:
    num_poses = fg.num_poses

    # if there are priors, then we will introduce an auxiliary pose to represent
    # the priors in the frame of this pose
    if fg.has_priors:
        num_poses += 1

    dim = fg.dimension
    pose_dims = dim + 1
    num_landmarks = fg.num_landmarks
    mat_dim = (pose_dims * num_poses) + num_landmarks
    return mat_dim


def _get_full_data_matrix_dim(fg: FactorGraphData) -> int:
    """Returns the dimension of the full data matrix for the given problem.

    Args:
        fg (FactorGraphData): the factor graph data describing the problem

    Returns:
        int: the dimension of the full data matrix
    """
    variables_dim = _get_pose_data_matrix_dim(fg) + fg.num_range_measurements
    full_matrix_dim = variables_dim
    return full_matrix_dim


def _get_dim_from_pose_measure_type(pose_measure: POSE_MEASUREMENT_TYPES) -> int:
    if isinstance(pose_measure, PoseMeasurement2D):
        dim = 2
    elif isinstance(pose_measure, PoseMeasurement3D):
        dim = 3
    else:
        raise ValueError(f"Invalid measurement type: {type(pose_measure)}")
    return dim


def _get_dim_from_pose_to_landmark_measure_type(
    measure: POSE_LANDMARK_MEASUREMENT_TYPES,
) -> int:
    if isinstance(measure, PoseToLandmarkMeasurement2D):
        dim = 2
    elif isinstance(measure, PoseToLandmarkMeasurement3D):
        dim = 3
    else:
        raise ValueError(f"Invalid measurement type: {type(measure)}")
    return dim


def _get_Vij(
    pose_var_idxs: Dict[str, PoseVariableIndices],
    pose_measure: POSE_MEASUREMENT_TYPES,
) -> Tuple[List[float], List[int], List[int]]:
    """Returns the V_ij matrix for the given odom measurement.

    See Cartan-Sync supplement Eq 6.

    Args:
        pose_var_idxs (Dict[str, PoseVariableIndices]): the corresponding
        indices for each pose variable
        pose_measure (POSE_MEASUREMENT_TYPES): the odom measurement
        mat_dim (int): the dimension of the data matrix

    Returns:
        data (List[float]): the data values for the sparse matrix
        row_idxs (List[int]): the row indices for the sparse matrix
        col_idxs (List[int]): the column indices for the sparse matrix

    """
    assert isinstance(pose_measure, (PoseMeasurement2D, PoseMeasurement3D))
    dim = _get_dim_from_pose_measure_type(pose_measure)

    # block-column vector
    # V_ij = LilMatrix((mat_dim, dim + 1))
    row_vals = []
    col_vals = []
    data_vals = []

    # V_i = - T_ij^{measured}
    neg_homogenous_rel_pose = -pose_measure.transformation_matrix
    i_idx_start = pose_var_idxs[pose_measure.base_pose].rot_start
    dim_se = dim + 1
    for row in range(i_idx_start, i_idx_start + dim_se):
        for col in range(dim + 1):
            v = neg_homogenous_rel_pose[row - i_idx_start, col]
            row_vals.append(row)
            col_vals.append(col)
            data_vals.append(v)

    # V_j = I
    j_idx_start = pose_var_idxs[pose_measure.to_pose].rot_start
    v = 1
    col = 0
    for row in range(j_idx_start, j_idx_start + dim_se):
        row_vals.append(row)
        col_vals.append(col)
        col += 1
        data_vals.append(v)

    return data_vals, row_vals, col_vals


def _get_Vij_pose_to_landmark(
    pose_var_idxs: Dict[str, PoseVariableIndices],
    landmark_var_idxs: Dict[str, int],
    pose_to_landmark_measure: POSE_LANDMARK_MEASUREMENT_TYPES,
) -> Tuple[List[float], List[int], List[int]]:
    """Returns the V_ij matrix for the given odom measurement.

    See Cartan-Sync supplement Eq 6.

    Args:
        pose_var_idxs (Dict[str, PoseVariableIndices]): the corresponding
        indices for each pose variable
        landmark_var_idxs (Dict[str, int]): the corresponding indices for
        each landmark variable
        pose_to_landmark_measure (POSE_MEASUREMENT_TYPES): the odom measurement
        mat_dim (int): the dimension of the data matrix

    Returns:
        data (List[float]): the data values for the sparse matrix
        row_idxs (List[int]): the row indices for the sparse matrix
        col_idxs (List[int]): the column indices for the sparse matrix

    """
    assert isinstance(
        pose_to_landmark_measure,
        (PoseToLandmarkMeasurement2D, PoseToLandmarkMeasurement3D),
    )
    dim = _get_dim_from_pose_to_landmark_measure_type(pose_to_landmark_measure)

    # block-column vector
    row_vals = []
    col_vals = []
    data_vals = []

    # this is slightly different because the landmark is just a translation
    # variable so instead of needing Vij to be width d+1, it is just width 1
    # (i.e. the update is rank-1 instead of rank-(d+1))

    # this is the same as the relative pose case but stripping out the
    # relationship between the rotations

    # V_i = - t_ij^{prior} (homogenized)
    neg_homogenous_rel_trans = -np.ones(dim + 1)  # make ones so it's homogenized
    neg_homogenous_rel_trans[:dim] = -pose_to_landmark_measure.translation_vector
    i_idx_start = pose_var_idxs[pose_to_landmark_measure.pose_name].rot_start
    dim_se = dim + 1
    col = 0
    cnt = 0
    for row in range(i_idx_start, i_idx_start + dim_se):
        v = neg_homogenous_rel_trans[cnt]
        row_vals.append(row)
        col_vals.append(col)
        data_vals.append(v)
        cnt += 1

    # V_j = 1
    j_idx = landmark_var_idxs[pose_to_landmark_measure.landmark_name]
    v = 1
    row = j_idx
    col = 0
    row_vals.append(row)
    col_vals.append(col)
    data_vals.append(v)
    return data_vals, row_vals, col_vals


def _get_pose_data_matrix(fg: FactorGraphData) -> torch.Tensor:
    """Computes the relative-pose component of the data matrix for the given
    problem (this is based purely on the measurements). Note that the syntax of
    this code follows the supplement from Cartan-Sync (Briales et al.)

    Q_pose \in \symmetric{R}^{(d+1) \times n}

    From Cartan-Sync: Q = A @ Omega @ A.T

    A = [V_{e_1} ... V_{e_m}]
    Omega = block_diag(Omega_1, ... Omega_m)

    Args:
        fg (FactorGraphData): the factor graph data describing the problem

    Returns:
        torch.Tensor: the data matrix for the relative-pose component of the
    """
    d = fg.dimension
    V_ij_height = _get_pose_data_matrix_dim(fg)
    V_ij_shape = (V_ij_height, d + 1)

    # V_ij_list = []
    V_vals = []
    V_rows = []
    V_cols = []
    Omega_vals = []
    pose_var_idxs = _get_pose_variable_idxs(fg)

    # add up odometry measurements
    rel_pose_measure_cnt = 0
    for odom_chain in fg.odom_measurements:
        for odom_idx in range(len(odom_chain)):

            # get the odom measurement
            odom_measure = odom_chain[odom_idx]

            # get the V_ij matrix for this odom measurement
            V_ij_data, V_ij_row, V_ij_col = _get_Vij(pose_var_idxs, odom_measure)

            # offset the column indices of Vij to account for hstack
            col_offset = rel_pose_measure_cnt * V_ij_shape[1]
            V_ij_col = [c + col_offset for c in V_ij_col]
            V_vals.extend(V_ij_data)
            V_rows.extend(V_ij_row)
            V_cols.extend(V_ij_col)
            rel_pose_measure_cnt += 1

            # each Omega_ij is a diagonal matrix of values:
            #  [rot_weight] * dim + [trans_weight]
            Omega_vals.extend([odom_measure.rotation_precision] * d)
            Omega_vals.append(odom_measure.translation_precision)

    # add up loop closure measurements
    for loop_measure in fg.loop_closure_measurements:

        # get the V_ij matrix for this odom measurement
        V_ij_data, V_ij_row, V_ij_col = _get_Vij(pose_var_idxs, loop_measure)

        # offset the column indices of Vij to account for hstack
        col_offset = rel_pose_measure_cnt * V_ij_shape[1]
        V_ij_col = [c + col_offset for c in V_ij_col]
        V_vals.extend(V_ij_data)
        V_rows.extend(V_ij_row)
        V_cols.extend(V_ij_col)
        rel_pose_measure_cnt += 1

        # each Omega_ij is a diagonal matrix of values:
        #  [rot_weight] * dim + [trans_weight]
        Omega_vals.extend([odom_measure.rotation_precision] * d)
        Omega_vals.append(odom_measure.translation_precision)

    # add up any priors on the poses
    origin_pose_name = "origin"
    if len(fg.pose_priors) > 0:
        logger.warning("Adding pose priors to the data matrix")
    for pose_prior in fg.pose_priors:
        # get the V_ij matrix for this odom measurement
        if fg.dimension == 2:
            assert isinstance(pose_prior, PosePrior2D)
            prior_pose_measure = PoseMeasurement2D(
                base_pose=origin_pose_name,
                to_pose=pose_prior.name,
                x=pose_prior.x,
                y=pose_prior.y,
                theta=pose_prior.theta,
                translation_precision=pose_prior.translation_precision,
                rotation_precision=pose_prior.rotation_precision,
                timestamp=pose_prior.timestamp,
            )
        elif fg.dimension == 3:
            assert isinstance(pose_prior, PosePrior3D)
            prior_pose_measure = PoseMeasurement3D(
                base_pose=origin_pose_name,
                to_pose=pose_prior.name,
                translation=pose_prior.translation_vector,
                rotation=pose_prior.rotation,
                translation_precision=pose_prior.translation_precision,
                rotation_precision=pose_prior.rotation_precision,
                timestamp=pose_prior.timestamp,
            )

        V_ij_data, V_ij_row, V_ij_col = _get_Vij(pose_var_idxs, prior_pose_measure)
        col_offset = rel_pose_measure_cnt * V_ij_shape[1]
        V_ij_col = [c + col_offset for c in V_ij_col]
        V_vals.extend(V_ij_data)
        V_rows.extend(V_ij_row)
        V_cols.extend(V_ij_col)
        rel_pose_measure_cnt += 1

        # each Omega_ij is a diagonal matrix of values:
        #  [rot_weight] * dim + [trans_weight]
        Omega_vals.extend([prior_pose_measure.rotation_precision] * d)
        Omega_vals.append(prior_pose_measure.translation_precision)

    # add up priors on the landmarks
    landmark_var_idxs = _get_beacon_variable_idxs(fg)
    landmark_cnt = 0
    if len(fg.landmark_priors) > 0:
        logger.warning("Adding landmark priors to the data matrix")

    for landmark_prior in fg.landmark_priors:
        if fg.dimension == 2:
            assert isinstance(landmark_prior, LandmarkPrior2D)
            prior_landmark_measure = PoseToLandmarkMeasurement2D(
                pose_name=origin_pose_name,
                landmark_name=landmark_prior.name,
                x=landmark_prior.x,
                y=landmark_prior.y,
                translation_precision=landmark_prior.translation_precision,
                timestamp=landmark_prior.timestamp,
            )
        elif fg.dimension == 3:
            assert isinstance(landmark_prior, LandmarkPrior3D)
            prior_landmark_measure = PoseToLandmarkMeasurement3D(
                pose_name=origin_pose_name,
                landmark_name=landmark_prior.name,
                x=landmark_prior.x,
                y=landmark_prior.y,
                z=landmark_prior.z,
                translation_precision=landmark_prior.translation_precision,
                timestamp=landmark_prior.timestamp,
            )

        V_ij_data, V_ij_row, V_ij_col = _get_Vij_pose_to_landmark(
            pose_var_idxs, landmark_var_idxs, prior_landmark_measure
        )
        col_offset = rel_pose_measure_cnt * V_ij_shape[1] + landmark_cnt
        V_ij_col = [c + col_offset for c in V_ij_col]
        V_vals.extend(V_ij_data)
        V_rows.extend(V_ij_row)
        V_cols.extend(V_ij_col)
        landmark_cnt += 1

        # each Omega_ij is a diagonal matrix of values:
        Omega_vals.append(prior_landmark_measure.translation_precision)

    # stitch V_rows and V_cols together as a list of 2-tuples
    V_idxs = list(zip(V_rows, V_cols))
    V_idx_set = set()
    for idx in V_idxs:
        assert idx not in V_idx_set, f"Duplicate index {idx}"
        V_idx_set.add(idx)

    # the Omega_ij matrices are all diagonal, so we can infer the indices
    Omega_idxs = np.arange(len(Omega_vals), dtype=np.int64)
    Omega_idxs = np.vstack((Omega_idxs, Omega_idxs))

    Omega_shape = (len(Omega_vals), len(Omega_vals))
    Omega = torch.sparse_coo_tensor(
        indices=torch.LongTensor(Omega_idxs),
        values=torch.DoubleTensor(Omega_vals),
        size=Omega_shape,
        dtype=torch.double,
    )

    V_shape = (V_ij_height, rel_pose_measure_cnt * (d + 1) + landmark_cnt)
    V = torch.sparse_coo_tensor(
        indices=torch.LongTensor([V_rows, V_cols]),
        values=torch.DoubleTensor(V_vals),
        size=V_shape,
        dtype=torch.double,
    )

    # Q_pose = V @ Omega @ V.T
    Q_pose = torch.sparse.mm(V, Omega)  # V @ Omega
    Q_pose = torch.sparse.mm(Q_pose, V.t()).coalesce()  # Q_pose @ V.T

    return Q_pose


def _get_range_data_matrix(
    fg: FactorGraphData, idx_offset: int
) -> Tuple[List[float], List[int], List[int], Tuple[int, int]]:
    """Computes the range component of the data matrix for the given problem
    (this is based purely on the range measurements).

    Args:
        fg (FactorGraphData): the factor graph data describing the problem
        idx_offset (int): the offset to add to the variable indices

    Returns:
        vals (List[float]): the values of the data matrix
        rows (List[int]): the row indices of the data matrix
        cols (List[int]): the column indices of the data matrix
        shape (Tuple[int, int]): the shape of the data matrix
    """
    num_ranges = fg.num_range_measurements

    mat_dim = num_ranges + idx_offset
    mat_shape = (mat_dim, mat_dim)

    # make a sparse square matrix of zeros
    Q_range_rows = []
    Q_range_cols = []
    Q_range_vals = []

    def _add_to_Q_range(row, col, val):
        Q_range_rows.append(row)
        Q_range_cols.append(col)
        Q_range_vals.append(val)

    pose_var_idxs = _get_pose_variable_idxs(fg)
    beacon_idxs = _get_beacon_variable_idxs(fg)
    dist_idxs = _get_dist_variable_idxs(fg)
    for range_idx, range_measure in enumerate(fg.range_measurements):
        precision = range_measure.precision
        dist = range_measure.dist
        key1, key2 = range_measure.association

        t1_idx = pose_var_idxs[key1].translation
        if key2 in pose_var_idxs:
            t2_idx = pose_var_idxs[key2].translation
        elif key2 in beacon_idxs:
            t2_idx = beacon_idxs[key2]
        else:
            raise ValueError(f"Unknown key {key2}")

        dist_idx = dist_idxs[range_measure.association].dist_var_idx

        # optimal when dist_ij == (ti - tj) / norm(ti - tj)
        # cost_ij = precision * || ti - tj - dist * dist_ij ||^2
        # K(4,4) = 1; % t1
        _add_to_Q_range(t1_idx, t1_idx, precision)

        # K(8,8) = 1; % t2
        _add_to_Q_range(t2_idx, t2_idx, precision)

        # K(4,8) = -1; % t1/t2
        # K(8,4) = -1; % t2/t1
        _add_to_Q_range(t1_idx, t2_idx, -precision)
        _add_to_Q_range(t2_idx, t1_idx, -precision)

        weighted_dist = precision * dist
        # K(9,9) = d*d; % d12
        _add_to_Q_range(dist_idx, dist_idx, weighted_dist * dist)

        # K(4,9) = -d; % t1/d12
        # K(9,4) = -d; % d12/t1
        _add_to_Q_range(t1_idx, dist_idx, -weighted_dist)
        _add_to_Q_range(dist_idx, t1_idx, -weighted_dist)

        # K(8,9) = d;  % t2/d12
        # K(9,8) = d;  % d12/t2
        _add_to_Q_range(t2_idx, dist_idx, weighted_dist)
        _add_to_Q_range(dist_idx, t2_idx, weighted_dist)

    return Q_range_vals, Q_range_rows, Q_range_cols, mat_shape
