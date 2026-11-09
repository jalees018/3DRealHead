"""FLAME head model evaluated on a dense UV grid.

Instead of deforming the FLAME vertices, every texel of a 512x512 UV map is placed on the
mesh via barycentric interpolation of its triangle's vertices, displaced by a per-texel
offset, and then skinned with the interpolated FLAME LBS weights.
"""
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes, vertices2joints
from smplx.utils import Struct, to_np, to_tensor

from .transforms import so3_exp_map

UV_RES = 512


class _ChumpyArray:
    """Stand-in for ``chumpy.Ch`` so the FLAME pickle loads without chumpy; keeps only the array."""

    def __setstate__(self, state):
        self.x = state["x"]


class _FlameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _ChumpyArray
        return super().find_class(module, name)


def load_flame_pickle(path):
    with open(path, "rb") as f:
        model = _FlameUnpickler(f, encoding="latin1").load()
    return {k: v.x if isinstance(v, _ChumpyArray) else v for k, v in model.items()}


def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents, lbs_weights,
        offsets, verts_to_uv_index, uv, w, verts_ids):
    """Linear blend skinning of UV-sampled surface points.

    Args:
        betas: (B, n_shape + n_expr) shape and expression coefficients.
        pose: (B, (J + 1) * 3) axis-angle joint rotations.
        offsets: (B, 512, 512, 3) per-texel displacement added before skinning.
        verts_to_uv_index: maps FLAME vertices to the vertices of the UV mesh.
        uv: (1, 512, 512, 2) barycentric coordinates of each texel in its triangle.
        w: (512, 512, 1) third barycentric coordinate (1 - u - v).
        verts_ids: (512, 512, 3) UV-mesh vertex indices of the triangle under each texel.

    Returns:
        verts (B, 512, 512, 3), joints (B, J, 3), rot_mats (B, J, 3, 3),
        per-texel skinning weights (B, 512, 512, J).
    """
    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype

    v_shaped = v_template + blend_shapes(betas, shapedirs)
    J = vertices2joints(J_regressor, v_shaped)

    ident = torch.eye(3, dtype=dtype, device=device)
    rot_mats = batch_rodrigues(pose.view(-1, 3)).view([batch_size, -1, 3, 3])
    pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
    pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)

    # Interpolate the posed vertices at every texel and add the predicted offsets.
    v_posedi = pose_offsets + v_shaped
    tri = v_posedi[:, verts_to_uv_index][:, verts_ids]
    v_posed = (tri[:, :, :, 0, :] * uv[..., 0:1]
               + tri[:, :, :, 1, :] * uv[..., 1:]
               + tri[:, :, :, 2, :] * (1 - uv[..., 0:1] - uv[..., 1:])
               + offsets)

    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    tri_weights = lbs_weights[verts_to_uv_index][verts_ids]
    lbs_weights2 = (tri_weights[:, :, 0, :] * uv[0, ..., 0:1]
                    + tri_weights[:, :, 1, :] * uv[0, ..., 1:]
                    + tri_weights[:, :, 2, :] * w)
    W = lbs_weights2.unsqueeze(dim=0).expand([batch_size, -1, -1, -1])
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, UV_RES, UV_RES, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], v_posed.shape[2], 1], dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=3)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    verts = v_homo[:, :, :, :3, 0]
    return verts, J_transformed, rot_mats, W


class FlameHead(nn.Module):
    """FLAME (100 shape / 50 expression components) evaluated on the UV grid."""

    def __init__(self, batch_size, flame_model_path):
        super().__init__()
        self.n_shape_params = 100
        self.n_expr_params = 50
        self.batch_size = batch_size
        self.dtype = torch.float32

        flame_model = Struct(**load_flame_pickle(flame_model_path))

        self.register_buffer("v_template", to_tensor(to_np(flame_model.v_template), dtype=self.dtype))

        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat(
            [shapedirs[:, :, :self.n_shape_params], shapedirs[:, :, 300:300 + self.n_expr_params]], 2)
        self.register_buffer("shapedirs", shapedirs)

        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(to_np(posedirs), dtype=self.dtype))
        self.register_buffer("J_regressor", to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype))
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)
        self.register_buffer("lbs_weights", to_tensor(to_np(flame_model.weights), dtype=self.dtype))

    def forward(self, shape_params, expression_params, pose_params, neck_pose, eye_pose,
                offsets, verts_to_uv_index, uv, w, verts_ids):
        """pose_params = [global rotation (3), jaw (3)]; returns (verts, rot_mats, skinning weights)."""
        betas = torch.cat([shape_params, expression_params], dim=1)
        full_pose = torch.cat([pose_params[:, :3], neck_pose, pose_params[:, 3:], eye_pose], dim=1)
        template_vertices = self.v_template.unsqueeze(0).repeat(self.batch_size, 1, 1)

        if offsets is not None and offsets.dim() == 2:
            offsets = offsets.unsqueeze(0).expand(self.batch_size, -1, -1)

        vertices, _, rot_mats, lbs_weights = lbs(
            betas, full_pose, template_vertices, self.shapedirs, self.posedirs, self.J_regressor,
            self.parents, self.lbs_weights, offsets, verts_to_uv_index, uv, w, verts_ids)
        return vertices, rot_mats, lbs_weights


class FLAMEModule(nn.Module):
    """Wraps `FlameHead` with tracker-style parameters (VHAP/FLAME conventions).

    Args:
        batch_size: number of heads evaluated per call.
        device: device of the FLAME buffers.
        assets_dir: directory containing `flame2023.pkl`.
    """

    def __init__(self, batch_size, device, assets_dir):
        super().__init__()
        self.shape_dims = 100
        self.exp_dims = 50
        self.batch_size = batch_size
        self.flame = FlameHead(batch_size, os.path.join(assets_dir, "flame2023.pkl")).to(device)
        self.neck_pose = torch.zeros(self.batch_size, 3, dtype=torch.float32).to(device)

    def forward(self, offsets, id_coeff, exp_coeff, scale, pose, verts_to_uv_index, uv, w, verts_ids,
                neck_pose=None):
        """Place UV texels on the posed FLAME surface.

        Args:
            offsets: (B, 512, 512, 3) per-texel offsets in canonical space.
            id_coeff: (B, 100) shape coefficients.
            exp_coeff: (B, 59) = expression (50) | jaw pose (3) | eye poses (6).
            scale: unused (the tracker scale is not applied).
            pose: (B, 6) = global rotation (axis-angle) | translation.
            neck_pose: (B, 3), zeros if None.

        Returns:
            vertices (B, 512, 512, 3), global rotation R (B, 3, 3), joint rotations,
            per-texel skinning weights.
        """
        expression_params = exp_coeff[:, :self.exp_dims]
        jaw_rotation = exp_coeff[:, self.exp_dims:self.exp_dims + 3]
        eye_pose = exp_coeff[:, self.exp_dims + 3:self.exp_dims + 9]
        if neck_pose is None:
            neck_pose = self.neck_pose
        pose_params = torch.cat([pose[:, :3], jaw_rotation], 1)

        vertices, rot_mats, lbs_weights = self.flame(
            id_coeff, expression_params, pose_params, neck_pose, eye_pose,
            offsets, verts_to_uv_index, uv, w, verts_ids)
        R = so3_exp_map(pose[:, :3])
        T = pose[:, 3:]

        # The global rotation is already part of the skinning; only the translation is added.
        vertices = vertices.view(self.batch_size, UV_RES * UV_RES, 3) + T[:, None, :]
        vertices = vertices.view(self.batch_size, UV_RES, UV_RES, 3)
        return vertices, R, rot_mats, lbs_weights
