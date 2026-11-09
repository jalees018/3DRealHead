"""Conversion of predicted UV Gaussian maps into posed 3D Gaussians, plus quaternion helpers.

Quaternions are stored as ``(w, x, y, z)``.
"""
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from simple_knn._C import distCUDA2

from gaussian_splatting import inverse_sigmoid

# Channel layout of the generator output (59 channels per texel).
XYZ, OPACITY, SCALING, ROTATION, SH = slice(0, 3), slice(3, 4), slice(4, 7), slice(7, 11), slice(11, 59)
NUM_SH_COEFFS = 16
DISPLACEMENT_SCALE = 0.07  # max. offset of a Gaussian from its mesh anchor (tanh-bounded)


class UVLayout(NamedTuple):
    """Barycentric UV parameterisation of the FLAME mesh used to place Gaussians on the surface."""
    uv: torch.Tensor  # (1, H, W, 2) first two barycentric coordinates per texel
    w: torch.Tensor  # (H, W, 1) third barycentric coordinate
    verts_ids: torch.Tensor  # (H, W, 3) vertex indices of the triangle each texel lies in
    verts_to_uv_index: torch.Tensor  # FLAME vertex -> UV-mesh vertex mapping


class FlameParams(NamedTuple):
    shape: torch.Tensor  # (1, 100)
    expression: torch.Tensor  # (1, 50 + 3 jaw + 6 eyes)
    scale: torch.Tensor  # (1, 1)
    pose: torch.Tensor  # (1, 6) global rotation (axis-angle) + translation
    neck_pose: torch.Tensor  # (1, 3)


class GaussianInit(NamedTuple):
    """Per-texel initial values the predicted residuals are added to."""
    scaling: torch.Tensor  # (N_valid, 3) log-scales from the nearest-neighbour distance
    opacity: torch.Tensor  # (H*W, 1) opacity logits
    rotation: torch.Tensor  # (H*W, 4) identity quaternions


def load_uv_layout(uv_dir, device):
    uvs = np.load(f"{uv_dir}/uvs.npy")
    w = 1 - uvs[0, ..., 0] - uvs[0, ..., 1]
    return UVLayout(
        uv=torch.tensor(uvs).float().to(device),
        w=torch.tensor(w).unsqueeze(-1).float().to(device),
        verts_ids=torch.tensor(np.load(f"{uv_dir}/vertex_ids.npy")).long().to(device),
        verts_to_uv_index=torch.tensor(np.load(f"{uv_dir}/verts_to_uv_index.npy")).long().to(device),
    )


def load_canonical_position_map(position_map_dir, device):
    """Canonical texel positions (H, W, 3) and the boolean mask (H, W) of texels covered by the mesh."""
    can_loc = torch.tensor(np.load(f"{position_map_dir}/map_0000_back.npy")).float().to(device)
    mask = (torch.tensor(np.load(f"{position_map_dir}/masks_can0000_back.npy")).to(device) / 255.0).bool()
    return can_loc, mask[:, :, 0]


def init_gaussians(can_loc, texel_mask, device, opacity=0.7):
    dist2 = torch.clamp_min(distCUDA2(can_loc.reshape(-1, 3)[texel_mask.reshape(-1)]), 0.0000001)
    num_texels = can_loc.shape[0] * can_loc.shape[1]
    rotation = torch.zeros((num_texels, 4)).to(device)
    rotation[:, 0] = 1
    return GaussianInit(
        scaling=torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3),
        opacity=inverse_sigmoid(opacity * torch.ones((num_texels, 1), dtype=torch.float, device=device)),
        rotation=rotation,
    )


def gaussian_maps_to_3d(g_maps, texel_mask, init, flame_model, flame, uv_layout, offsets=None,
                        freeze_geometry=False):
    """Turn one predicted Gaussian map into posed 3D Gaussians.

    Each valid texel carries one Gaussian anchored on the FLAME surface; the network predicts a
    bounded displacement, opacity/scale/rotation residuals and SH colours. The displaced points are
    posed with FLAME LBS and the Gaussian rotations are skinned with the same blend weights.

    Args:
        g_maps: (1, 59, H, W) generator output.
        texel_mask: (H, W) bool mask of texels covered by the UV layout.
        offsets: optional extra per-texel displacement (1, H, W, 3), e.g. optimised at test time.
        freeze_geometry: keep Gaussians exactly on the mesh with identity rotation (training warm-up).

    Returns:
        A dict with pre-activation ``xyz``, ``opacity``, ``scaling``, ``rotation`` and ``colors``
        (N, 48), and the raw displacement map (1, H, W, 3) used for regularisation.
    """
    mask = texel_mask.reshape(-1)
    maps = g_maps.permute(0, 2, 3, 1)

    rot_init = init.rotation[mask]
    if freeze_geometry:
        rotation = rot_init
        displacement = 0.0 * maps[..., XYZ]
    else:
        rotation = F.normalize(maps[..., ROTATION].reshape(-1, 4)[mask] + rot_init)
        displacement = maps[..., XYZ]

    opacity = (maps[..., OPACITY].reshape(-1, 1) + init.opacity)[mask]
    scaling = maps[..., SCALING].reshape(-1, 3)[mask] + init.scaling
    colors = maps[..., SH].reshape(-1, 3 * NUM_SH_COEFFS)[mask]

    surface_offsets = DISPLACEMENT_SCALE * torch.tanh(displacement)
    if offsets is not None:
        surface_offsets = surface_offsets + offsets
    vertices, head_rot, joint_rots, lbs_weights = flame_model(
        surface_offsets, flame.shape, flame.expression, flame.scale, flame.pose, uv_layout.verts_to_uv_index,
        uv_layout.uv, uv_layout.w, uv_layout.verts_ids, flame.neck_pose)

    skin_weights = lbs_weights.reshape((1, -1, 5))[:, mask, ...]
    rotation = skinning_gauss_rotations(skin_weights, joint_rots, rotation.unsqueeze(0))
    head_quat = F.normalize(rotation_matrix_to_quaternion(head_rot))
    rotation = quaternion_multiply(head_quat.repeat((rotation.shape[1], 1)), rotation[0])

    gaussians = {
        "xyz": vertices.reshape(-1, 3)[mask],
        "opacity": opacity,
        "scaling": scaling,
        "rotation": rotation,
        "colors": colors,
    }
    return gaussians, displacement


def split_sh(colors):
    """Split (N, 48) SH colours into the DC (N, 1, 3) and higher-order (N, 15, 3) features."""
    features = colors.reshape(-1, NUM_SH_COEFFS, 3)
    return features[:, 0:1, :], features[:, 1:, :]


def rotate_gaussians(gaussians, points, features_dc, features_rest, yaw, device):
    """Write ``points`` into ``gaussians`` rotated about the vertical axis by ``yaw`` radians."""
    angles = np.array([0, yaw, 0], dtype=np.float32)
    rot = torch.tensor(Rotation.from_euler("zyx", angles, degrees=False).as_matrix()).float().to(device)
    quat = rotation_matrix_to_quaternion(rot.unsqueeze(0))
    gaussians.modify(torch.matmul(rot, points["xyz"].t()).t(), points["opacity"], points["scaling"],
                     quaternion_multiply(F.normalize(quat), points["rotation"]), features_dc, features_rest)
    return gaussians


def quat_normalize(q, eps=1e-8):
    return q / (torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(eps))


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), dim=-1)


def skinning_gauss_rotations(skin_weights, joint_rot, gauss_rot_local):
    """Blend per-joint rotations with LBS weights (real part of dual-quaternion skinning).

    Args:
        skin_weights: (B, N, J) blend weights.
        joint_rot: (B, J, 3, 3) joint rotation matrices.
        gauss_rot_local: (B, N, 4) local Gaussian orientations.

    Returns:
        (B, N, 4) skinned unit quaternions.
    """
    joint_quat = rotation_matrix_to_quaternion_batched(joint_rot[0]).unsqueeze(0)
    _, N, _ = skin_weights.shape

    dq_j = torch.cat([joint_quat, torch.zeros_like(joint_quat)], dim=-1)  # (B, J, 8)
    dq_j = dq_j.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, J, 8)
    dq_blend = (dq_j * skin_weights.unsqueeze(-1)).sum(dim=2)  # (B, N, 8)

    blended_quat = quat_normalize(dq_blend[..., :4])
    return quat_normalize(quaternion_multiply(blended_quat, gauss_rot_local))


def rotation_matrix_to_quaternion_batched(R):
    """(B, 3, 3) rotation matrices -> (B, 4) unit quaternions."""
    q = torch.zeros((R.shape[0], 4), device=R.device, dtype=R.dtype)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    cond1 = trace > 0
    s1 = torch.sqrt(trace[cond1] + 1.0) * 2
    q[cond1, 0] = 0.25 * s1
    q[cond1, 1] = (R[cond1, 2, 1] - R[cond1, 1, 2]) / s1
    q[cond1, 2] = (R[cond1, 0, 2] - R[cond1, 2, 0]) / s1
    q[cond1, 3] = (R[cond1, 1, 0] - R[cond1, 0, 1]) / s1

    cond2 = ~cond1 & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s2 = torch.sqrt(1.0 + R[cond2, 0, 0] - R[cond2, 1, 1] - R[cond2, 2, 2]) * 2
    q[cond2, 0] = (R[cond2, 2, 1] - R[cond2, 1, 2]) / s2
    q[cond2, 1] = 0.25 * s2
    q[cond2, 2] = (R[cond2, 0, 1] + R[cond2, 1, 0]) / s2
    q[cond2, 3] = (R[cond2, 0, 2] + R[cond2, 2, 0]) / s2

    cond3 = ~cond1 & ~cond2 & (R[:, 1, 1] > R[:, 2, 2])
    s3 = torch.sqrt(1.0 + R[cond3, 1, 1] - R[cond3, 0, 0] - R[cond3, 2, 2]) * 2
    q[cond3, 0] = (R[cond3, 0, 2] - R[cond3, 2, 0]) / s3
    q[cond3, 1] = (R[cond3, 0, 1] + R[cond3, 1, 0]) / s3
    q[cond3, 2] = 0.25 * s3
    q[cond3, 3] = (R[cond3, 1, 2] + R[cond3, 2, 1]) / s3

    cond4 = ~cond1 & ~cond2 & ~cond3
    s4 = torch.sqrt(1.0 + R[cond4, 2, 2] - R[cond4, 0, 0] - R[cond4, 1, 1]) * 2
    q[cond4, 0] = (R[cond4, 1, 0] - R[cond4, 0, 1]) / s4
    q[cond4, 1] = (R[cond4, 0, 2] + R[cond4, 2, 0]) / s4
    q[cond4, 2] = (R[cond4, 1, 2] + R[cond4, 2, 1]) / s4
    q[cond4, 3] = 0.25 * s4

    return q / q.norm(dim=1, keepdim=True)


def rotation_matrix_to_quaternion(R):
    """Quaternion of the first matrix in a (B, 3, 3) batch (used for the global head rotation)."""
    trace = R[0].trace().unsqueeze(0)
    q = torch.zeros((R.shape[0], 4), dtype=R.dtype, device=R.device)

    mask = trace > 0
    s = torch.sqrt(1.0 + trace[mask])
    q[mask, 0] = 0.5 * s
    q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / (2.0 * s)
    q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / (2.0 * s)
    q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / (2.0 * s)

    mask = ~mask
    i = torch.argmax(R.diagonal(dim1=-2, dim2=-1), dim=-1)[mask]
    j = (i + 1) % 3
    k = (j + 1) % 3
    s = torch.sqrt(1.0 + R[mask, i, i] - R[mask, j, j] - R[mask, k, k])
    q[mask, 1] = (R[mask, k, j] - R[mask, j, k]) / (2.0 * s)
    q[mask, 2] = 0.5 * s
    q[mask, 3] = (R[mask, j, i] + R[mask, i, j]) / (2.0 * s)

    q[:, 0] = torch.sqrt(1.0 + trace) / 2.0
    return q
