from attrs import define, field
import torch
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import pymanopt
from pymanopt.manifolds import (
    Stiefel,
    Product,
    Oblique,
    Euclidean,
)
from pymanopt.manifolds.manifold import Manifold
from typing import Sequence, Tuple, Callable, Optional, Union

from data_matrix import get_full_data_matrix
from matrix_type_conversions import convert_torch_tensor_to_scipy_sparse
from py_factor_graph.factor_graph import FactorGraphData
from numba import njit


def get_stiefel_manifolds(fg: FactorGraphData, lifted_dim: int):
    """Create the stiefel manifolds for the factor graph.

    Args:
        fg (FactorGraph): the factor graph

    Returns:
        List[Stiefel]: the stiefel manifolds
    """
    num_poses = fg.num_poses
    dim = fg.dimension
    return Stiefel(lifted_dim, dim, k=num_poses)


def get_pose_translation_manifold(fg: FactorGraphData, lifted_dim: int):
    """Create the euclidean manifold for the factor graph.

    Args:
        fg (FactorGraph): the factor graph

    Returns:
        Euclidean: the euclidean manifold
    """
    num_poses = fg.num_poses
    # Stiefel shape = num_poses, latent_dim, lifted_dim
    shape = (num_poses, lifted_dim, 1)
    return Euclidean(*shape)


def get_beacon_translation_manifold(fg: FactorGraphData, lifted_dim: int):
    """Create the euclidean manifold for the factor graph.

    Args:
        fg (FactorGraph): the factor graph

    Returns:
        Euclidean: the euclidean manifold
    """
    num_landmarks = fg.num_landmarks
    # Stiefel shape = num_poses, latent_dim, lifted_dim
    shape = (lifted_dim, num_landmarks)
    return Euclidean(*shape)


def get_oblique_manifold(fg: FactorGraphData, lifted_dim: int):
    """Create the oblique manifold for the factor graph.

    Args:
        fg (FactorGraph): the factor graph

    Returns:
        Oblique: the oblique manifold
    """
    num_oblique_cols = fg.num_range_measurements
    assert (
        lifted_dim >= fg.dimension
    ), f"lifted_dim must be >= {fg.dimension}, but got {lifted_dim}"
    return Oblique(lifted_dim, num_oblique_cols)


def unpack_point(
    point: np.ndarray,
    num_poses: int,
    num_beacons: int,
    latent_dim: int,
    lifted_dim: int,
):
    """Unpack a point (X) on the manifold into the stiefel, euclidean, and oblique
    parts.

    Args:
        point (torch.Tensor): the point on the manifold
        num_poses (int): the number of poses
        latent_dim (int): the latent dimension
        lifted_dim (int): the lifted dimension

    Returns:
        tuple: the stiefel, euclidean, and oblique parts of the point
    """
    # get the range measurements component
    beacon_var_start_idx = (latent_dim + 1) * num_poses
    range_var_start_idx = beacon_var_start_idx + num_beacons

    beacon_component = point[beacon_var_start_idx:range_var_start_idx, :].T
    range_component = point[range_var_start_idx:, :].T

    # get the translation gradient
    trans_col_idxs = []
    rot_col_idxs = []
    for i in range(beacon_var_start_idx):
        col_num = i + 1
        is_tran_col = col_num % (latent_dim + 1) == 0
        if is_tran_col:
            trans_col_idxs.append(i)
        else:
            rot_col_idxs.append(i)

    trans_component = point[trans_col_idxs, :, np.newaxis]
    rot_component = point[rot_col_idxs, :]

    # right now rot_grad is shape (num_poses * latent_dim, lifted_dim)
    # we want to reshape it to (num_poses, lifted_dim, latent_dim)
    rot_component = rot_component.reshape(num_poses, latent_dim, lifted_dim)

    # flip the axes to get (num_poses, lifted_dim, latent_dim)
    rot_component = np.swapaxes(rot_component, 1, 2)

    components = (rot_component, trans_component, beacon_component, range_component)
    return components


def lift_rot_matrix_to_stiefel(rot_mat: np.ndarray, lifted_dim: int):
    rot_dim = rot_mat.shape[0]
    lifted_mat = np.zeros((lifted_dim, rot_dim))
    lifted_mat[:rot_dim, :rot_dim] = rot_mat
    return lifted_mat


def lift_translation_vector(tran: Union[np.ndarray, tuple], lifted_dim: int):
    if isinstance(tran, tuple):
        tran = np.array(tran).reshape(-1, 1)
    base_dim = tran.shape[0]
    lifted_vec = np.zeros((lifted_dim, 1))
    lifted_vec[:base_dim] = tran.reshape(-1, 1)
    return lifted_vec


def get_ground_truth_initialization(
    fg: FactorGraphData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the ground truth initialization for the factor graph.

    Args:
        fg (PyFactorGraph): the factor graph
        lifted_dim (int): the lifted dimension

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: the ground truth
        initialization (rotations, translations, range directions)
    """

    rot_vals = []
    tran_vals = []
    name_to_tran_mapping = {}
    dist_vals = []

    # if has priors add identity pose
    if fg.has_priors:
        rot_vals.append(np.eye(fg.dimension))
        tran_vals.append(np.zeros((fg.dimension, 1)))

    for pose_chain in fg.pose_variables:
        for pose in pose_chain:
            pose_name = pose.name
            rot = pose.rotation_matrix
            rot_vals.append(rot)

            tran = pose.position_vector.reshape(-1, 1)
            tran_vals.append(tran)
            name_to_tran_mapping[pose_name] = tran

    beacon_vals = []
    for beacon in fg.landmark_variables:
        beacon_vals.append(beacon.true_position)
        name_to_tran_mapping[beacon.name] = np.array(beacon.true_position).reshape(
            -1, 1
        )

    for measure in fg.range_measurements:
        pose_i, pose_j = measure.association
        ti = name_to_tran_mapping[pose_i]
        tj = name_to_tran_mapping[pose_j]
        t_diff = ti - tj
        dij = t_diff / la.norm(t_diff)
        dist_vals.append(dij)

    dim = fg.dimension
    rot_vals = np.array(rot_vals)  # shape (num_poses, dim, dim)
    tran_vals = np.array(tran_vals)  # shape (num_poses, dim, 1)
    beacon_vals = np.array(beacon_vals).reshape(-1, dim).T  # shape (dim, num_beacons)
    dist_vals = (
        np.array(dist_vals).reshape(-1, dim).T
    )  # shape (dim, num_range_measurements)

    init_vals = (rot_vals, tran_vals, beacon_vals, dist_vals)
    return init_vals


def get_odometric_initialization(
    fg: FactorGraphData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    odom_chains = fg.odometry_trajectories
    dim = fg.dimension

    rot_vals = []
    tran_vals = []
    name_to_tran_mapping = {}
    dist_vals = []

    # if there are priors add an identity pose at the beginning
    if fg.has_priors:
        rot_vals.append(np.eye(dim))
        tran_vals.append(np.zeros((dim, 1)))

    for robot_idx in range(len(odom_chains)):
        for pose_idx in range(len(odom_chains[robot_idx])):
            pose_name = fg.pose_variables[robot_idx][pose_idx].name
            odom_pose = odom_chains[robot_idx][pose_idx]
            rot = odom_pose[:dim, :dim]
            rot_vals.append(rot)

            tran = odom_pose[:dim, dim].reshape(-1, 1)
            tran_vals.append(tran)
            name_to_tran_mapping[pose_name] = tran

    beacon_vals = []
    for beacon in fg.landmark_variables:
        beacon_vals.append(beacon.true_position)
        name_to_tran_mapping[beacon.name] = np.array(beacon.true_position).reshape(
            -1, 1
        )

    for measure in fg.range_measurements:
        pose_i, pose_j = measure.association
        ti = name_to_tran_mapping[pose_i]
        tj = name_to_tran_mapping[pose_j]
        t_diff = ti - tj
        dij = t_diff / la.norm(t_diff)
        dist_vals.append(dij)

    rot_vals = np.array(rot_vals)
    tran_vals = np.array(tran_vals)
    beacon_vals = np.array(beacon_vals).reshape(-1, dim).T
    dist_vals = np.array(dist_vals).reshape(-1, dim).T

    init_vals = (rot_vals, tran_vals, beacon_vals, dist_vals)
    return init_vals


def get_cost_egrad_ehessian_and_precond(
    fg: FactorGraphData, prod: Product, backend: str
) -> Tuple[Callable, Optional[Callable], Optional[Callable], np.ndarray]:
    """Get the cost, euclidean gradient, and euclidean hessian functions for the
    factor graph.

    Args:
        fg (FactorGraph): the factor graph

    Returns:
        Tuple[Callable, Optional[Callable], Optional[Callable]]: the cost,
        egrad, and hessian functions
    """
    Q = get_full_data_matrix(fg)

    Q_scipy = convert_torch_tensor_to_scipy_sparse(Q)
    m, n = Q_scipy.shape
    approx_Q_inv = sp.spdiags((Q_scipy.diagonal() ** -1), 0, m, n)
    Qinv = approx_Q_inv * 100
    # Qinv = spla.inv(Q_scipy)

    backend_options = ["numpy"]
    assert (
        backend in backend_options
    ), f"backend must be in {backend_options}, but got {backend}"

    if backend == "numpy":

        @pymanopt.function.numpy(prod)
        def ra_slam_cost_numpy(
            rot_point: Stiefel,
            trans_point: Euclidean,
            beacon_point: Euclidean,
            range_point: Oblique,
        ):

            X = get_X_from_manifold_numpy(
                rot_point, trans_point, beacon_point, range_point
            )
            X_torch = torch.from_numpy(X)

            # cost = trace(X^T Q X)
            QX = torch.sparse.mm(Q, X_torch)
            cost = torch.trace(torch.mm(X_torch.T, QX))
            return cost

        @pymanopt.function.numpy(prod)
        def ra_slam_egrad_numpy(
            rot_point: Stiefel,
            trans_point: Euclidean,
            beacon_point: Euclidean,
            range_point: Oblique,
        ):

            num_poses, lifted_dim, latent_dim = rot_point.shape
            X = get_X_from_manifold_numpy(
                rot_point, trans_point, beacon_point, range_point
            )
            X_torch = torch.from_numpy(X)

            # cost = trace(X^T Q X)
            QX = torch.sparse.mm(Q, X_torch)
            grad = 2 * QX.numpy()

            grads = unpack_point(
                grad, fg.num_poses, fg.num_landmarks, fg.dimension, lifted_dim
            )

            return grads

        @pymanopt.function.numpy(prod)
        def ra_slam_ehess_numpy(
            rot_point: Stiefel,
            trans_point: Euclidean,
            beacon_point: Euclidean,
            range_point: Oblique,
            rot_vec: np.ndarray,
            trans_vec: np.ndarray,
            beacon_vec: np.ndarray,
            range_vec: np.ndarray,
        ):
            """Compute the hessian of the cost function on the manifold.

            cost = trace(X^T Q X)
            gradient = 2 * QX
            hessian = 2 * Q

            Args:
                rot_point (Stiefel): _description_
                trans_point (Euclidean): _description_
                range_point (Oblique): _description_

            Returns:
                _type_: _description_
            """
            num_poses, lifted_dim, latent_dim = rot_point.shape
            vec = get_X_from_manifold_numpy(rot_vec, trans_vec, beacon_vec, range_vec)
            hess_product = 2 * torch.sparse.mm(Q, torch.from_numpy(vec)).numpy()
            hess_product_components = unpack_point(
                hess_product, fg.num_poses, fg.num_landmarks, fg.dimension, lifted_dim
            )
            return hess_product_components

        @pymanopt.function.numpy(prod)
        def ra_slam_precond_numpy(
            point: Product,
            tangent_vec: Product,
        ):
            """Compute the preconditioner for the manifold.

            Args:
                rot_point (Stiefel): _description_
                trans_point (Euclidean): _description_
                beacon_point (Euclidean): _description_
                range_point (Oblique): _description_

            Returns:
                _type_: _description_
            """
            num_poses, lifted_dim, latent_dim = point[0].shape
            tangent = get_X_from_manifold_numpy(*tangent_vec)

            # print(f"Tangent max: {np.max(tangent)}")

            preconditioned_tangent = Qinv @ tangent

            # print(f"Preconditioned tangent max: {np.max(preconditioned_tangent)}")

            conditioned_tangent_components = unpack_point(
                preconditioned_tangent,
                fg.num_poses,
                fg.num_landmarks,
                fg.dimension,
                lifted_dim,
            )
            # construct a tangent vector on the manifold
            tangent_vec[0] = conditioned_tangent_components[0]
            tangent_vec[1] = conditioned_tangent_components[1]
            tangent_vec[2] = conditioned_tangent_components[2]
            tangent_vec[3] = conditioned_tangent_components[3]
            return tangent_vec

        cost = ra_slam_cost_numpy
        egrad = ra_slam_egrad_numpy
        ehess = ra_slam_ehess_numpy
        precond = ra_slam_precond_numpy
    else:
        raise NotImplementedError(f"Backend {backend} not implemented")

    return cost, egrad, ehess, precond


@define
class RtrSolverOptions:
    # soln = optimizer.run(problem, initial_point=None, mininner=1,
    # maxinner=None, Delta_bar=delta_bar, Delta0=None)
    manifold: Manifold = field()
    initial_point: Optional[Sequence[np.ndarray]] = None
    mininner: int = 1
    maxinner: Optional[int] = None
    Delta_bar: Optional[float] = None
    Delta0: Optional[float] = None

    # set Delta bar to the typical distance of the manifold
    @property
    def delta_bar(self) -> float:
        if self.Delta_bar is None:
            return self.manifold.typical_dist
        else:
            return self.Delta_bar


def get_X_from_manifold_torch(
    rot_point: Stiefel, trans_point: Euclidean, range_point: Oblique
) -> torch.Tensor:
    # rot shape = num_poses, lifted_dim, latent_dim
    # trans shape = num_poses, lifted_dim, 1
    # range shape = lifted_dim, num_range_measurements
    num_poses, lifted_dim, latent_dim = rot_point.shape
    assert lifted_dim == range_point.shape[0] == trans_point.shape[1]
    assert num_poses == trans_point.shape[0]
    assert trans_point.shape[2] == 1

    # stitch together the manifold points into a block-vector
    # want T = transpose([rot0, trans0, rot1, trans1, ...]), D = [range0, range1, ...]
    # from this form X = [T, D], where D is the oblique manifold
    T = torch.concatenate(
        [
            torch.concatenate([rot_point[i, :, :], trans_point[i, :, :]], axis=1)
            for i in range(num_poses)
        ],
        axis=1,
    )
    assert T.shape == (
        lifted_dim,
        (latent_dim + 1) * num_poses,
    ), f"T.shape = {T.shape}"

    X = torch.concatenate([T, range_point], axis=1).T
    return X


# small example took 61.6 seconds w/ njit vs 83.0 secs w/o
@njit
def get_T_from_manifold_numpy(rot_point: Stiefel, trans_point: Euclidean) -> np.ndarray:
    num_poses, lifted_dim, latent_dim = rot_point.shape
    # assert (
    #     lifted_dim == range_point.shape[0] == trans_point.shape[1]
    # ), f"Lifted dim: {lifted_dim}, range point: {range_point.shape}, trans point: {trans_point.shape}"
    # assert num_poses == trans_point.shape[0]
    # assert trans_point.shape[2] == 1
    T = np.zeros((lifted_dim, (latent_dim + 1) * num_poses))
    pose_dim = latent_dim + 1
    for i in range(num_poses):
        rot_start_idx = i * (pose_dim)
        rot_end_idx = rot_start_idx + latent_dim
        trans_idx = rot_end_idx
        T[:, rot_start_idx:rot_end_idx] = rot_point[i, :, :]
        T[:, trans_idx] = trans_point[i, :, 0]
    # assert T.shape == (
    #     lifted_dim,
    #     (latent_dim + 1) * num_poses,
    # ), f"T.shape = {T.shape}"
    return T


def get_X_from_manifold_numpy(
    rot_point: Stiefel,
    trans_point: Euclidean,
    beacon_point: Euclidean,
    range_point: Oblique,
) -> np.ndarray:
    # rot shape = num_poses, lifted_dim, latent_dim
    # trans shape = num_poses, lifted_dim, 1
    # beacon shape = lifted_dim, num_beacons
    # range shape = lifted_dim, num_range_measurements
    T = get_T_from_manifold_numpy(rot_point, trans_point)
    X = np.concatenate([T, beacon_point, range_point], axis=1).T
    return X
