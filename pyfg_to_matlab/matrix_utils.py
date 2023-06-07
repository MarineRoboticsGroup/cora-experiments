from py_factor_graph.utils.solver_utils import VariableValues
from py_factor_graph.factor_graph import FactorGraphData
import numpy as np
import torch
from attrs import define, field
from typing import Tuple, List, Dict, Union
import scipy.sparse.csr as csr
import scipy.sparse.csc as csc
import scipy.sparse.coo as coo
import scipy.sparse.lil as lil


SparseMatrixTypes = Union[
    lil.lil_matrix, csr.csr_matrix, csc.csc_matrix, coo.coo_matrix
]
LilMatrix = lil.lil_matrix
CsrMatrix = csr.csr_matrix
CscMatrix = csc.csc_matrix
CooMatrix = coo.coo_matrix


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


@define
class PoseVariableIndices:
    """Class for storing the indices of the pose variables in the data matrix."""

    var_name: str = field()
    rot_start: int = field()
    rot_end: int = field()
    translation: int = field()


@define
class DistanceVariableIndices:
    """Class for storing the indices of the distance variables in the data matrix."""

    association_name: Tuple[str, str] = field()
    first_translation_idx: int = field()
    second_translation_idx: int = field()
    dist_var_idx: int = field()


def get_var_idxs(
    dim: int,
    num_poses: int,
    num_beacons: int,
    num_range_measurements: int,
    has_priors: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns the indices for each set of variables:
    - rotations
    - translations
    - beacons
    - range measurements


    Args:
        dim (int): dimension of the problem
        num_poses (int): number of poses
        num_beacons (int): number of beacons
        num_range_measurements (int): number of range measurements
        has_priors (bool): whether or not the factor graph has priors

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: _description_
    """
    beacon_var_start_idx = (dim + 1) * num_poses
    range_var_start_idx = beacon_var_start_idx + num_beacons
    full_mat_len = range_var_start_idx + num_range_measurements

    beacon_idxs = np.arange(beacon_var_start_idx, range_var_start_idx)
    assert len(beacon_idxs) == num_beacons
    range_idxs = np.arange(range_var_start_idx, full_mat_len)
    assert len(range_idxs) == num_range_measurements

    trans_col_idxs = []
    rot_col_idxs = []
    last_pose_idx = beacon_var_start_idx - 1
    for i in range(last_pose_idx + 1):
        col_num = i + 1
        is_tran_col = col_num % (dim + 1) == 0
        if is_tran_col:
            trans_col_idxs.append(i)
        else:
            rot_col_idxs.append(i)

    rot_idxs = np.array(rot_col_idxs)
    tran_idxs = np.array(trans_col_idxs)
    assert len(rot_idxs) == num_poses * dim
    assert len(tran_idxs) == num_poses

    return rot_idxs, tran_idxs, beacon_idxs, range_idxs


def get_var_idxs_from_pyfg(
    fg: FactorGraphData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return get_var_idxs(
        fg.dimension,
        fg.num_poses + int(fg.has_priors),
        fg.num_landmarks,
        fg.num_range_measurements,
        fg.has_priors,
    )


def get_stacked_mat_from_var_vals(var_vals: VariableValues) -> torch.Tensor:
    poses = [mat[:-1, :] for mat in var_vals.poses.values()]

    # get the dimension so we can properly reshape all of the vectors
    dim = poses[0].shape[0]
    beacons = [vec.reshape(dim, 1) for vec in var_vals.landmarks.values()]

    # get the distance variables
    translation_values = var_vals.translations
    dist_vars: List[np.ndarray] = []
    for association, dist in var_vals.distances.items():
        trans1 = translation_values[association[0]]
        trans2 = translation_values[association[1]]
        trans_diff = trans1 - trans2
        trans_diff_direction = trans_diff / np.linalg.norm(trans_diff)

        trans_diff_direction_norm = np.linalg.norm(trans_diff_direction)
        assert np.isclose(trans_diff_direction_norm, 1.0), (
            f"the direction should be a unit vector, received"
            f" {trans_diff_direction_norm} from {trans_diff_direction}"
        )

        dist_vars.append(trans_diff_direction.reshape(dim, 1))

    # stack the poses, beacons, and dist vars
    stacked_mat = np.transpose(np.hstack(poses + beacons + dist_vars))

    # convert to torch tensor
    stacked_mat = torch.from_numpy(stacked_mat).double()

    return stacked_mat


def convert_stacked_block_matrix_to_column_vector_matrix(
    stacked_block_mat: torch.Tensor, block_height: int
) -> torch.Tensor:
    """Converts the stacked block matrix to a column vector matrix.

    E.g. X = [A1; A2; A3] -> X_vec = [vec(A1), vec(A2), vec(A3)]
    where ';' denotes concatenation and vec() denotes vectorization.

    Args:
        stacked_block_mat (torch.Tensor): the stacked block matrix
        block_height (int): the height of each block

    Returns:
        torch.Tensor: the column vector matrix
    """
    stacked_height, width = stacked_block_mat.shape
    assert (
        stacked_height % block_height == 0
    ), "stacked_height must be divisible by block_height"
    num_blocks = int(stacked_height / block_height)
    # shape of the vectorized column matrix
    num_columns = num_blocks
    vector_height = block_height * width
    vectorized_shape = (vector_height, num_columns)

    if not stacked_block_mat.is_sparse:
        raise NotImplementedError(
            "Only implemented for sparse matrices, please convert to sparse"
        )
        stacked_block_mat = stacked_block_mat.reshape(num_blocks, block_height, width)
        vectorized_column_matrix = stacked_block_mat.T.reshape(
            num_columns, block_height * width
        ).T
    else:
        indices = stacked_block_mat._indices()

        # rows are the row of the block + the offset due to stacking the
        # columns of the block
        block_row_indices = indices[0] % block_height
        column_offset = indices[1] * block_height
        rows = block_row_indices + column_offset

        # columns correspond to the block indices (i.e. the first block is
        # column 0, the second is column 1, etc.)
        columns = indices[0] // block_height

        # create the sparse tensor
        vectorized_column_matrix = torch.sparse_coo_tensor(
            torch.stack([rows, columns]), stacked_block_mat._values(), vectorized_shape
        )

    assert vectorized_column_matrix.shape == vectorized_shape, (
        f"vectorized_column_matrix.shape = {vectorized_column_matrix.shape}, "
        f"vectorized_shape = {vectorized_shape}"
    )

    return vectorized_column_matrix


def vectorize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Vectorizes the given tensor.

    Args:
        tensor (torch.Tensor): the tensor to vectorize

    Returns:
        torch.Tensor: the vectorized tensor
    """
    tensor_height, tensor_width = tensor.shape
    vectorized_tensor = convert_stacked_block_matrix_to_column_vector_matrix(
        tensor, tensor_height
    )
    return vectorized_tensor


def vectorize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Vectorizes the given matrix from m x n to (m*n) x 1.

    Args:
        matrix (np.ndarray): the matrix to vectorize

    Returns:
        np.ndarray: the vectorized matrix
    """
    v = np.reshape(matrix, (-1, 1), order="F")
    return v


def unpack_vectorized_matrix(vec: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Unpacks the given vectorized matrix into the original shape.

    Args:
        vec (np.ndarray): the vectorized matrix
        shape (Tuple[int, int]): the original shape

    Returns:
        np.ndarray: the unpacked matrix
    """
    assert vec.shape[0] == shape[0] * shape[1], (
        f"Vectorized matrix has length {vec.shape}, "
        f"but expected shape {shape[0] * shape[1]}"
    )
    height, width = shape

    if isinstance(vec, torch.Tensor):
        # print the sparsity of the matrix
        # nnz = torch.nonzero(vec).shape[0]
        # numel = vec.numel()
        unpacked_vec = convert_stacked_block_matrix_to_column_vector_matrix(
            vec.to_sparse(), height
        )
        return unpacked_vec
    elif isinstance(vec, np.ndarray):
        return np.reshape(vec, shape, order="F")
    else:
        raise NotImplementedError(f"Type {type(vec)} not supported")


def vectorize_sparse_matrix(matrix: SparseMatrixTypes) -> SparseMatrixTypes:
    """Vectorizes the given matrix from m x n to (m*n) x 1.

    Args:
        matrix (SparseMatrixTypes): the matrix to vectorize

    Returns:
        SparseMatrixTypes: the vectorized matrix
    """
    assert isinstance(
        matrix, (CooMatrix, CsrMatrix, LilMatrix)
    ), f"Expected csr_matrix, got {type(matrix)}"
    raise NotImplementedError("Not implemented yet")
    vec = matrix.reshape((-1, 1))
    return vec


def evaluate_cost_at_solution(Q_full: np.ndarray, var_vals: VariableValues) -> float:
    """Evaluates the cost of the given variable values.

    Args:
        Q_full (np.ndarray): the full data matrix
        var_vals (VariableValues): the variable values

    Returns:
        float: the cost of the variable values
    """
    X = get_stacked_mat_from_var_vals(var_vals)
    cost = np.trace(X.T @ Q_full @ X)
    return cost


def get_prior_index_offset(fg: FactorGraphData) -> int:
    """if there are prior factors then we introduce an auxiliary pose at the
    beginning to represent the frame of the prior

    Args:
        fg (FactorGraphData): the factor graph

    Returns:
        int: the offset of the prior factors
    """
    if fg.has_priors:
        num_aux_poses = 1
    else:
        num_aux_poses = 0
    prior_offset = num_aux_poses * (fg.dimension + 1)
    return prior_offset


def _get_pose_variable_idxs(fg: FactorGraphData) -> Dict[str, PoseVariableIndices]:
    """Returns the starting indices of the robot poses in the
    data matrix.

    Args:
        fg (FactorGraphData): the factor graph data describing the problem

    Returns:
        Dict[str, PoseVariableIndices]: the starting indices of the robot poses
    """
    pose_var_idxs = {}
    pose_num = 0

    # if priors are present then we introduce an auxiliary pose at the origin
    # (kept at beginning of our indexing)
    if fg.has_priors:
        prior_frame_name = "origin"
        prior_frame_idxs = PoseVariableIndices(
            var_name=prior_frame_name,
            rot_start=0,
            rot_end=fg.dimension,
            translation=fg.dimension,
        )
        pose_var_idxs[prior_frame_name] = prior_frame_idxs
        pose_num += 1

    for pose_chain in fg.pose_variables:
        for pose_var in pose_chain:
            rot_start = pose_num * (fg.dimension + 1)
            rot_end = rot_start + fg.dimension
            translation_idx = rot_end
            pose_var_idxs[pose_var.name] = PoseVariableIndices(
                pose_var.name, rot_start, rot_end, translation_idx
            )
            pose_num += 1

    assert len(pose_var_idxs) == fg.num_poses + int(fg.has_priors)
    return pose_var_idxs


def _get_beacon_variable_idxs(fg: FactorGraphData) -> Dict[str, int]:
    """Returns a list of the starting indices of the beacons in the
    data matrix.

    Args:
        fg (FactorGraphData): the factor graph data describing the problem

    Returns:
        List[int]: list of starting indices
    """
    beacon_var_idxs = {}
    pose_num = fg.num_poses
    prior_offset = get_prior_index_offset(fg)
    first_beacon_idx = pose_num * (fg.dimension + 1) + prior_offset
    cur_beacon_idx = first_beacon_idx
    for beacon_var in fg.landmark_variables:
        beacon_var_idxs[beacon_var.name] = cur_beacon_idx
        cur_beacon_idx += 1

    assert len(beacon_var_idxs) == fg.num_landmarks
    return beacon_var_idxs


def _get_dist_variable_idxs(
    fg: FactorGraphData,
) -> Dict[Tuple[str, str], DistanceVariableIndices]:
    """Returns a list of the starting indices of the range measurements in the
    data matrix.

    Args:
        fg (FactorGraphData): the factor graph data describing the problem

    Returns:
        List[int]: list of starting indices
    """
    pose_var_idxs = _get_pose_variable_idxs(fg)
    beacon_var_idxs = _get_beacon_variable_idxs(fg)
    dist_var_idxs = {}
    pose_num = fg.num_poses + int(fg.has_priors)
    first_beacon_idx = pose_num * (fg.dimension + 1)
    first_dist_idx = first_beacon_idx + fg.num_landmarks
    cur_dist_idx = first_dist_idx

    for dist_measure in fg.range_measurements:
        association = dist_measure.association
        assert association not in dist_var_idxs, f"Duplicate association {association}"

        first_trans, second_trans = association
        first_trans_idx = pose_var_idxs[first_trans].translation

        if second_trans.startswith("L"):
            second_trans_idx = beacon_var_idxs[second_trans]
        elif second_trans[0].isalpha():
            second_trans_idx = pose_var_idxs[second_trans].translation
        else:
            raise ValueError()

        dist_measure_idxs = DistanceVariableIndices(
            association, first_trans_idx, second_trans_idx, cur_dist_idx
        )
        dist_var_idxs[association] = dist_measure_idxs

        cur_dist_idx += 1

    assert len(dist_var_idxs) == fg.num_range_measurements
    return dist_var_idxs
