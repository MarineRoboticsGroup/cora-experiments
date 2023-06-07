from py_factor_graph.factor_graph import FactorGraphData
from typing import List, Tuple

from .data_matrix import _get_full_data_matrix_dim
from .matrix_utils import _get_dist_variable_idxs

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


def _get_num_constraints(fg: FactorGraphData) -> int:
    """Returns the number of quadratic constraints in the problem.

    Args:
        fg (FactorGraphData): the factor graph data describing the problem

    Returns:
        int: the number of constraints
    """
    num_poses = fg.num_poses
    d = fg.dimension
    num_rotation_constraints_per_pose: int = d * (d + 1) / 2

    num_rotation_constraints = num_poses * num_rotation_constraints_per_pose
    num_range_constraints = fg.num_range_measurements

    num_total_constraints = num_rotation_constraints + num_range_constraints
    return int(num_total_constraints)


def get_constraints(fg: FactorGraphData) -> torch.Tensor:
    """Returns the constraint matrices for the given problem as a  vertically
    stacked block matrix (the constraint matrices are stacked on top of each
    other)

    e.g. K \in \R^{k*n x d} = [K_1; K_2; ...; K_n] where K_i \in \R^{k x d} is
    the constraint matrix for the i-th variable

    Args:
        fg (FactorGraphData): the factor graph data describing the problem

    Returns:
        torch.sparse_coo_tensor: the constraint matrices
    """
    num_constraints = _get_num_constraints(fg)
    if num_constraints == 0:
        return torch.zeros((0, 0))

    constraint_matrix_width = _get_full_data_matrix_dim(
        fg
    )  # constraint matrices are square

    def get_sparse_matrix_idxs_vals_and_shape(
        num_poses: int, d: int, num_distance_measurements: int
    ) -> Tuple[List, List, List]:
        """Construct a (n x n x m) tensor of sparse matrices, where m is the
        number of constraints, and n is the dimension of the state space
        """
        m = num_constraints
        n = constraint_matrix_width

        # Construct a (n x n x m) tensor of sparse symmetric matrices
        # for each pose there are (d * (d + 1)) / 2 rotation constraints
        vals = []
        idxs = []
        constraint_cnt = 0

        def add_pose_diagonal_constraint(row: int, col: int, val: float):
            nonlocal constraint_cnt
            row_offset = constraint_cnt * n
            vals.append(val)
            idxs.append([row_offset + row, col])
            constraint_cnt += 1

        def add_pose_offdiagonal_constraint(row: int, col: int, val: float):
            nonlocal constraint_cnt
            row_offset = constraint_cnt * n
            vals.append(val)
            idxs.append([row_offset + row, col])
            vals.append(val)
            idxs.append([row_offset + col, row])
            constraint_cnt += 1

        for pose_idx in range(num_poses):
            pose_offset = pose_idx * (d + 1)
            for rot_constraint_row in range(d):
                for rot_constraint_col in range(rot_constraint_row, d):
                    if rot_constraint_row == rot_constraint_col:
                        diag_idx = pose_offset + rot_constraint_row
                        add_pose_diagonal_constraint(diag_idx, diag_idx, 1)
                    else:
                        offdiag_idx_1 = pose_offset + rot_constraint_row
                        offdiag_idx_2 = pose_offset + rot_constraint_col
                        add_pose_offdiagonal_constraint(
                            offdiag_idx_1, offdiag_idx_2, 1 / 2
                        )

        # for each distance measurement there is 1 distance constraint
        def add_distance_constraint(dist_idx):
            nonlocal constraint_cnt

            row_offset = constraint_cnt * n
            vals.append(1)
            idxs.append([row_offset + dist_idx, dist_idx])

            # update the constraint count
            constraint_cnt += 1

        dist_var_idxs_map = _get_dist_variable_idxs(fg)
        for dist_measure_cnt in range(num_distance_measurements):
            dist_measure = fg.range_measurements[dist_measure_cnt]
            association = dist_measure.association
            dist_var_idxs = dist_var_idxs_map[association]

            dist_measure_idx = dist_var_idxs.dist_var_idx

            add_distance_constraint(dist_measure_idx)

        # check that we have the correct number of constraints
        assert (
            constraint_cnt == m
        ), f"Expected {m} constraints, but got {constraint_cnt}"

        stacked_height = n * m
        dense_tensor_shape = [stacked_height, n]
        return idxs, vals, dense_tensor_shape

    idxs, vals, dense_tensor_shape = get_sparse_matrix_idxs_vals_and_shape(
        fg.num_poses, fg.dimension, fg.num_range_measurements
    )
    mat: torch.Tensor = torch.sparse_coo_tensor(
        torch.LongTensor(idxs).T, torch.DoubleTensor(vals), tuple(dense_tensor_shape)
    )
    return mat
