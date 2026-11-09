import random
import struct

import cv2
import numpy as np
import torch


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(torch.device("cuda:0"))


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def load_uv_mask(path, batch_size):
    """Load a UV-space mask image as a (B, 3, H, W) float tensor in [0, 1]."""
    mask = torch.tensor(cv2.imread(path, -1)[:, :, :3] / 255)
    return mask.unsqueeze(0).repeat((batch_size, 1, 1, 1)).permute((0, 3, 1, 2)).float()


def to_uint8_bgr(image):
    """(3, H, W) tensor in [0, 1] -> (H, W, 3) uint8 BGR array for OpenCV."""
    return cv2.cvtColor((image.detach().permute((1, 2, 0)).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def write_pointcloud(filename, xyz_points, rgb_points=None):
    """Write an (N, 3) point cloud with optional uint8 colours to a binary PLY file."""
    assert xyz_points.shape[1] == 3, "Input XYZ points should be Nx3 float array"
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8) * 255
    assert xyz_points.shape == rgb_points.shape, "Input RGB colors should be Nx3 and match the XYZ points"

    with open(filename, "wb") as fid:
        fid.write(bytes("ply\n", "utf-8"))
        fid.write(bytes("format binary_little_endian 1.0\n", "utf-8"))
        fid.write(bytes("element vertex %d\n" % xyz_points.shape[0], "utf-8"))
        for prop in ("float x", "float y", "float z", "uchar red", "uchar green", "uchar blue"):
            fid.write(bytes(f"property {prop}\n", "utf-8"))
        fid.write(bytes("end_header\n", "utf-8"))
        for i in range(xyz_points.shape[0]):
            fid.write(bytearray(struct.pack("fffccc", xyz_points[i, 0], xyz_points[i, 1], xyz_points[i, 2],
                                            rgb_points[i, 0].tobytes(), rgb_points[i, 1].tobytes(),
                                            rgb_points[i, 2].tobytes())))
