import numpy as np
import scipy.io as sio
import scipy.sparse.csr as csr
import scipy.sparse.csc as csc
import scipy.sparse.coo as coo
import scipy.sparse.lil as lil
from typing import Union
import torch

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


def convert_scipy_sparse_to_torch_sparse(mat: coo.coo_matrix) -> torch.Tensor:
    """Converts a scipy sparse matrix to a torch sparse tensor.

    Args:
        mat (SparseMatrixTypes): the scipy sparse matrix

    Returns:
        torch.sparse_coo_tensor: the torch sparse tensor
    """
    indices = np.vstack((mat.row, mat.col))
    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(mat.data)
    shape = mat.shape
    return torch.sparse_coo_tensor(i, v, torch.Size(shape))


def convert_torch_tensor_to_scipy_sparse(mat: torch.Tensor) -> coo.coo_matrix:
    """Converts a torch tensor to a scipy sparse matrix.

    Args:
        mat (torch.Tensor): the torch tensor

    Returns:
        CooMatrix: the scipy sparse matrix
    """
    assert isinstance(mat, torch.Tensor), f"Expected torch.Tensor, got {type(mat)}"
    if not mat.layout == torch.sparse_coo:
        mat = mat.to_sparse_coo()  # type: ignore
    values = mat._values()
    indices = mat._indices()
    return CooMatrix(
        (values, (indices[0], indices[1])),
        shape=tuple(mat.shape),
    )


def save_numpy_array_to_matlab_file(
    array: np.ndarray, filename: str, var_name: str
) -> None:
    """Saves the given numpy array to a matlab file.

    Args:
        array (np.ndarray): the array to save
        filename (str): the filename to save to
        var_name (str): the variable name to use in the matlab file
    """
    sio.savemat(filename, {var_name: array})


if __name__ == "__main__":
    rand_torch = torch.rand(4, 3)
    sp_rand = convert_torch_tensor_to_scipy_sparse(rand_torch)
    conv_torch = convert_scipy_sparse_to_torch_sparse(sp_rand)

    print(f"rand_torch:\n{rand_torch.numpy()}")
    print(f"sp_rand:\n{sp_rand.todense()}")
    print(f"conv_torch:\n{conv_torch.to_dense().numpy()}")

    diff = rand_torch - conv_torch.to_dense()
    print(f"diff:\n{diff.numpy()}")
