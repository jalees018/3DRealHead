import json
import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import trimesh
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from avatar.subjects import NEUTRAL_SEQUENCE_INDEX, NERSEMBLE_CAMERAS, NERSEMBLE_SEQUENCES, NERSEMBLE_SUBJECTS

# Segmentation labels in ``segs/``: 22 is removed from the foreground, >= 23 is the mouth interior.
SEG_REMOVE_LABEL = 22
SEG_MOUTH_MIN_LABEL = 23


class PairedColorJitter:
    """Colour jitter with identical random factors for two images."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, img1, img2):
        if random.random() >= self.p:
            return img1, img2
        b = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness) if self.brightness else 1
        c = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast) if self.contrast else 1
        s = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation) if self.saturation else 1
        h = random.uniform(-self.hue, self.hue) if self.hue else 0

        def apply(img):
            img = TF.adjust_brightness(img, b)
            img = TF.adjust_contrast(img, c)
            img = TF.adjust_saturation(img, s)
            return TF.adjust_hue(img, h)

        return apply(img1), apply(img2)


class PairedGaussianBlur:
    """Gaussian blur with an identical random sigma for two images."""

    def __init__(self, kernel_size, sigma=(0.1, 2.0), p=0.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p

    def __call__(self, img1, img2):
        if random.random() >= self.p:
            return img1, img2
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return TF.gaussian_blur(img1, self.kernel_size, [sigma]), TF.gaussian_blur(img2, self.kernel_size, [sigma])


class PairedCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1, img2):
        for t in self.transforms:
            img1, img2 = t(img1, img2)
        return img1, img2


def _to_tensor_pair(img1, img2):
    return transforms.ToTensor()(img1), transforms.ToTensor()(img2)


class NeRSembleDataset(Dataset):
    """Multi-view NeRSemble frames with FLAME tracking, SMIRK expression meshes and fitted UV textures.

    ``__getitem__(view)`` samples a random (subject, sequence, frame) and returns camera ``view`` of it.
    Frames are drawn with the per-frame sampling weights in ``probability_file`` (+0.1 smoothing).

    Returns, in order:
        image (3, H, W) colour-corrected foreground image in [0, 1],
        expression_map (512, 512, 3) per-texel offset of the SMIRK mesh from the subject's neutral mesh,
        texture (512, 512, 3) fitted UV texture in [-1, 1] with the mouth interior blanked,
        view, subject_idx, frame_number (running frame id), mouth_mask (H, W, 3), alpha (H, W, 3) uint8,
        texture_grad (512, 512, 3) per-frame texture-gradient map, FLAME shape (1, 100),
        expression (1, 59) = expr + jaw + eyes, scale (1, 1), pose (1, 6) = rotation + translation,
        neck_pose (1, 3), sequence_idx.
    """

    def __init__(self, paths, num_subjects, frame_counts_file, probability_file, uv_dir, mouth_mask_file):
        self.paths = paths
        self.num_subjects = num_subjects
        self.transform = PairedCompose([
            PairedColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
            PairedGaussianBlur(kernel_size=5, sigma=(0.1, 1.0), p=0.5),
            _to_tensor_pair,
        ])
        self.frame_counts = np.load(frame_counts_file).tolist()  # [subject][sequence] -> number of frames
        with open(probability_file) as f:
            self.frame_weights = json.load(f)  # [subject][sequence][frame] -> sampling weight

        uvs = np.load(os.path.join(uv_dir, "uvs.npy"))
        self.uv = uvs.astype(np.float32)
        self.w = (1 - uvs[0, ..., 0] - uvs[0, ..., 1])[..., None].astype(np.float32)
        self.verts_ids = np.load(os.path.join(uv_dir, "vertex_ids.npy"))
        self.verts_to_uv_index = np.load(os.path.join(uv_dir, "verts_to_uv_index.npy"))

        mouth_mask = cv2.imread(mouth_mask_file, -1)[:, :, :3] / 255
        self.mouth_mask = mouth_mask.astype(np.uint8)
        self.mouth_mask_float = torch.tensor(mouth_mask.astype(np.float32))

        np.random.seed(0)

    def __len__(self):
        return len(NERSEMBLE_CAMERAS)

    def _sample_frame(self):
        subject_idx = np.random.randint(self.num_subjects)
        sequence_idx = random.choice(range(len(NERSEMBLE_SEQUENCES)))
        while self.frame_counts[subject_idx][sequence_idx] <= 0:
            subject_idx = np.random.randint(self.num_subjects)
            sequence_idx = random.choice(range(len(NERSEMBLE_SEQUENCES)))
        weights = np.array(self.frame_weights[subject_idx][sequence_idx]) + 0.1
        frame = np.random.choice(self.frame_counts[subject_idx][sequence_idx], p=(weights / sum(weights)).tolist())
        return subject_idx, sequence_idx, frame

    def _export_file(self, subject, sequence, folder, frame, suffix):
        """Path of an exported frame; some exports are offset by one frame, so fall back to frame + 1."""
        base = os.path.join(self.paths.nersemble_export, f"{subject:03d}", sequence, folder)
        path = os.path.join(base, f"{frame:05d}{suffix}")
        return path if os.path.exists(path) else os.path.join(base, f"{frame + 1:05d}{suffix}")

    def __getitem__(self, view):
        subject_idx, sequence_idx, frame = self._sample_frame()
        frame_number = (sum(sum(x) for x in self.frame_counts[0:subject_idx])
                        + sum(self.frame_counts[subject_idx][0:sequence_idx]) + frame)
        subject = NERSEMBLE_SUBJECTS[subject_idx]
        sequence = NERSEMBLE_SEQUENCES[sequence_idx]
        sub = f"{subject:03d}"

        # Image, foreground alpha and mouth mask.
        image = cv2.imread(self._export_file(subject, sequence, "images", frame, f"_{view:02d}.png"), -1)[:, :, :3]
        alpha = cv2.imread(self._export_file(subject, sequence, "fg_masks", frame, f"_{view:02d}.png"))[:, :, :3]
        seg = cv2.imread(self._export_file(subject, sequence, "segs", frame, f"_{view:02d}.png"), -1)
        alpha[seg == SEG_REMOVE_LABEL, :] = 0
        mouth_mask = np.repeat((seg >= SEG_MOUTH_MIN_LABEL)[..., None].astype(np.float32), 3, axis=2)

        color_correction = torch.tensor(np.load(os.path.join(
            self.paths.nersemble_color_correction, sub, NERSEMBLE_CAMERAS[view] + ".npy"))).unsqueeze(0).float()

        image = image * (alpha / 255)
        image = Image.fromarray(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))

        flame_params = np.load(self._export_file(subject, sequence, "flame_param", frame, ".npz"))

        # Expression conditioning: SMIRK mesh offsets w.r.t. the neutral mesh, rasterised into UV space.
        mesh = trimesh.load(os.path.join(self.paths.nersemble_smirk_meshes, sub, sequence, f"{frame:05d}.obj"),
                            process=False)
        neutral = trimesh.load(os.path.join(self.paths.nersemble_smirk_meshes, sub,
                                            NERSEMBLE_SEQUENCES[NEUTRAL_SEQUENCE_INDEX], "00000.obj"), process=False)
        offsets = (mesh.vertices - neutral.vertices)[self.verts_to_uv_index][self.verts_ids]
        expression_map = (offsets[:, :, 0, :] * self.uv[0, ..., 0:1] + offsets[:, :, 1, :] * self.uv[0, ..., 1:]
                          + offsets[:, :, 2, :] * self.w)

        # Appearance conditioning: the subject's fitted texture, or a neutral-frame texture.
        texture_dir = os.path.join(self.paths.nersemble_textures, sub)
        if random.random() < 0.65:
            texture = cv2.imread(os.path.join(texture_dir, f"tex_can_new_{sub}.png"), -1)[:, :, :3]
        else:
            texture = cv2.imread(os.path.join(texture_dir, "EXP-5-mouth", "tex_can_new_00000.png"), -1)[:, :, :3]
        texture = Image.fromarray(cv2.cvtColor(texture.astype(np.uint8), cv2.COLOR_BGR2RGB))

        image, texture = self.transform(image, texture)
        image = (torch.matmul(image.permute(1, 2, 0).unsqueeze(-2), color_correction[:, 0:3, 0:3])
                 + color_correction[:, 0:3, 3].unsqueeze(0).unsqueeze(0))[:, :, 0, :].permute(2, 0, 1)
        texture = -1 + texture.permute(1, 2, 0) * (1 - self.mouth_mask_float) * 2

        grad_path = os.path.join(texture_dir, sequence, f"tex_can_new_{frame:05d}_grad.npy")
        if not os.path.exists(grad_path):
            grad_path = os.path.join(texture_dir, sequence, f"tex_can_new_{frame + 1:05d}_grad.npy")
        texture_grad = np.load(grad_path).astype(np.float32) * self.mouth_mask

        shape = flame_params["shape"][None][:, 0:100]
        expression = np.concatenate((flame_params["expr"][0:1, 0:50], flame_params["jaw_pose"][0:1],
                                     flame_params["eyes_pose"][0:1]), axis=-1)
        neck_pose = flame_params["neck_pose"][0:1]
        pose = np.concatenate((flame_params["rotation"][0:1], flame_params["translation"][0:1]), axis=-1)
        scale = np.ones((1, 1), dtype=np.float32)

        return (image, np.asarray(expression_map, dtype=np.float32), texture, view, subject_idx, frame_number,
                mouth_mask, alpha, texture_grad, shape, expression, scale, pose, neck_pose, sequence_idx)
