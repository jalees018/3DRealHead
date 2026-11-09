"""Rotation helpers (adapted from PyTorch3D)."""
import torch


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle rotations (..., 3) to rotation matrices (..., 3, 3) via Rodrigues' formula."""
    shape = axis_angle.shape
    device, dtype = axis_angle.device, axis_angle.dtype

    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True).unsqueeze(-1)

    rx, ry, rz = axis_angle[..., 0], axis_angle[..., 1], axis_angle[..., 2]
    zeros = torch.zeros(shape[:-1], dtype=dtype, device=device)
    cross_product_matrix = torch.stack(
        [zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=-1
    ).view(shape + (3,))
    cross_product_matrix_sqrd = cross_product_matrix @ cross_product_matrix

    identity = torch.eye(3, dtype=dtype, device=device)
    angles_sqrd = angles * angles
    angles_sqrd = torch.where(angles_sqrd == 0, 1, angles_sqrd)
    return (
        identity.expand(cross_product_matrix.shape)
        + torch.sinc(angles / torch.pi) * cross_product_matrix
        + ((1 - torch.cos(angles)) / angles_sqrd) * cross_product_matrix_sqrd
    )


def so3_exp_map(log_rot: torch.Tensor) -> torch.Tensor:
    """Map a batch of axis-angle vectors (N, 3) to rotation matrices (N, 3, 3)."""
    if log_rot.dim() != 2 or log_rot.shape[1] != 3:
        raise ValueError("Input tensor shape has to be Nx3.")
    return axis_angle_to_matrix(log_rot)
