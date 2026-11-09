"""Camera loading for NeRSemble (multi-view) and in-the-wild (monocular) VHAP exports."""
import json
import os

import numpy as np

from avatar.subjects import NERSEMBLE_SEQUENCES, NERSEMBLE_SUBJECTS
from gaussian_splatting import Camera

NERSEMBLE_IMAGE_SIZE = (550, 802)  # (width, height)
ITW_IMAGE_SIZE = (512, 512)


def _camera_from_c2w(c2w, fov_x, fov_y, size):
    """Build a rasterizer camera from a VHAP (OpenGL convention) camera-to-world matrix."""
    w2c = np.linalg.inv(np.array(c2w))
    w2c[[2, 1]] *= -1  # OpenGL -> COLMAP axes
    R = np.transpose(w2c[:3, :3])  # stored transposed, as expected by the CUDA (glm) code
    T = w2c[:3, 3]
    return Camera(R, T, fov_x, fov_y, *size)


def nersemble_camera_index(subject_idx, sequence_idx, view_idx, num_views=16):
    """Index into the list returned by :func:`load_nersemble_cameras`."""
    return num_views * (len(NERSEMBLE_SEQUENCES) * subject_idx + sequence_idx) + view_idx


def load_nersemble_cameras(export_root, subjects=NERSEMBLE_SUBJECTS, sequences=NERSEMBLE_SEQUENCES, num_views=16):
    """Load the calibrated cameras of every (subject, sequence) pair.

    Cameras are ordered subject-major, then sequence, then view; use :func:`nersemble_camera_index`.
    Sequences missing for a subject fall back to the cameras of ``EXP-5-mouth``.
    """
    cameras = []
    for subject in subjects:
        for sequence in sequences:
            path = os.path.join(export_root, f"{subject:03d}", sequence, "transforms.json")
            if not os.path.exists(path):
                path = os.path.join(export_root, f"{subject:03d}", "EXP-5-mouth", "transforms.json")
            with open(path) as f:
                contents = json.load(f)
            for frame in contents["frames"][:num_views]:
                cameras.append(_camera_from_c2w(frame["transform_matrix"], contents["camera_angle_x"],
                                                contents["camera_angle_y"], NERSEMBLE_IMAGE_SIZE))
    return cameras


def load_itw_cameras(transforms_files, num_frames):
    """Load the per-frame cameras of monocular VHAP tracks.

    ``transforms_files`` is one ``transforms_train.json`` per subject; the first ``num_frames``
    frames of each are loaded, so frame ``f`` of subject ``s`` is at index ``s * num_frames + f``.
    """
    cameras = []
    for path in transforms_files:
        with open(path) as f:
            frames = json.load(f)["frames"]
        for frame in frames[:num_frames]:
            cameras.append(_camera_from_c2w(frame["transform_matrix"], frame["camera_angle_x"],
                                            frame["camera_angle_y"], ITW_IMAGE_SIZE))
    return cameras
