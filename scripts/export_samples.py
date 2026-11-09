"""Package sample data for inference from the full preprocessed datasets.

Copies exactly the files ``test_nersemble.py`` and ``test_itw.py`` read for the chosen subjects into
the repository's default ``data/`` layout (see ``configs/paths.yaml``), and strips a training
checkpoint down to the generator weights. Run it where the full data lives, then publish ``--out``.

    python scripts/export_samples.py --config configs/cluster.yaml --ckpt <train ckpt> \\
        --nersemble_subjects 368 --itw_subjects obama --out sample_release
"""
import argparse
import glob
import os
import shutil
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from avatar.config import REPO_ROOT, load_paths  # noqa: E402
from avatar.subjects import NERSEMBLE_CAMERAS, NERSEMBLE_SEQUENCES  # noqa: E402

NEUTRAL_SEQUENCE = "EXP-5-mouth"
FRONT_VIEW = 8
VHAP_EXPORT_SUFFIX = "_whiteBg_staticOffset_maskBelowLine"


def copy(src, dst_root, rel):
    dst = os.path.join(dst_root, rel)
    if not os.path.exists(src):
        print(f"[missing] {src}")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def copy_glob(pattern, src_root, dst_root, rel_root):
    for src in sorted(glob.glob(os.path.join(src_root, pattern))):
        copy(src, dst_root, os.path.join(rel_root, os.path.relpath(src, src_root)))


def export_nersemble(paths, data_dir, subject, inversion_seq, inversion_frame, driving_seq):
    sub = f"{subject:03d}"
    export, smirk, tex = paths.nersemble_export, paths.nersemble_smirk_meshes, paths.nersemble_test_textures
    out_export, out_smirk, out_tex = "nersemble/export", "nersemble/smirk", "nersemble/textures_test"

    for seq in set(NERSEMBLE_SEQUENCES) | {inversion_seq, driving_seq}:
        copy(os.path.join(export, sub, seq, "transforms.json"), data_dir, f"{out_export}/{sub}/{seq}/transforms.json")

    # Inversion frame: all views, FLAME, SMIRK mesh and texture gradients.
    for view in range(len(NERSEMBLE_CAMERAS)):
        for folder in ("images", "fg_masks", "segs"):
            name = f"{folder}/{inversion_frame:05d}_{view:02d}.png"
            copy(os.path.join(export, sub, inversion_seq, name), data_dir, f"{out_export}/{sub}/{inversion_seq}/{name}")
    for seq, frame in ((inversion_seq, inversion_frame), (NEUTRAL_SEQUENCE, 0)):
        copy(os.path.join(export, sub, seq, "flame_param", f"{frame:05d}.npz"), data_dir,
             f"{out_export}/{sub}/{seq}/flame_param/{frame:05d}.npz")
        copy(os.path.join(smirk, sub, seq, f"{frame:05d}.obj"), data_dir, f"{out_smirk}/{sub}/{seq}/{frame:05d}.obj")
        copy(os.path.join(tex, sub, seq, f"tex_can_new_{frame:05d}_grad.npy"), data_dir,
             f"{out_tex}/{sub}/{seq}/tex_can_new_{frame:05d}_grad.npy")
    copy(os.path.join(tex, sub, f"tex_can_new_{sub}.png"), data_dir, f"{out_tex}/{sub}/tex_can_new_{sub}.png")

    # Driving sequence: frontal view of every frame plus per-frame FLAME, SMIRK and gradients.
    for folder in ("images", "fg_masks", "segs"):
        copy_glob(f"{folder}/*_{FRONT_VIEW:02d}.png", os.path.join(export, sub, driving_seq), data_dir,
                  f"{out_export}/{sub}/{driving_seq}")
    copy_glob("flame_param/*.npz", os.path.join(export, sub, driving_seq), data_dir, f"{out_export}/{sub}/{driving_seq}")
    copy(os.path.join(smirk, sub, driving_seq), data_dir, f"{out_smirk}/{sub}/{driving_seq}")
    copy_glob("*_grad.npy", os.path.join(tex, sub, driving_seq), data_dir, f"{out_tex}/{sub}/{driving_seq}")

    copy(os.path.join(paths.nersemble_color_correction, sub), data_dir, f"nersemble/color_correction/{sub}")


def export_itw(paths, data_dir, name):
    copy(os.path.join(paths.itw_vhap_export, name + VHAP_EXPORT_SUFFIX), data_dir,
         f"itw/vhap_export/{name}{VHAP_EXPORT_SUFFIX}")
    copy(os.path.join(paths.itw_frames, name), data_dir, f"itw/frames/{name}")
    copy(os.path.join(paths.itw_smirk_meshes, name), data_dir, f"itw/smirk/{name}")
    copy(os.path.join(paths.itw_textures, name), data_dir, f"itw/textures/{name}")


def export_checkpoint(src, dst):
    import torch
    ckpt = torch.load(src, map_location="cpu")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    torch.save({"g": ckpt["g"]}, dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="path config pointing at the full datasets")
    parser.add_argument("--out", required=True, help="output folder (mirrors the repository root)")
    parser.add_argument("--ckpt", default=None, help="training checkpoint to strip to generator weights")
    parser.add_argument("--nersemble_subjects", type=int, nargs="*", default=[])
    parser.add_argument("--itw_subjects", nargs="*", default=[])
    parser.add_argument("--inversion_seq", default="EXP-9-jaw-2")
    parser.add_argument("--inversion_frame", type=int, default=0)
    parser.add_argument("--driving_seq", default="EXP-5-mouth")
    args = parser.parse_args()

    paths = load_paths(args.config)
    with open(os.path.join(REPO_ROOT, "configs", "paths.yaml")) as f:
        defaults = yaml.safe_load(f)
    # The sample data goes to the parent of the default NeRSemble/ITW folders (``data/``).
    data_dir = os.path.join(args.out, os.path.dirname(os.path.dirname(defaults["nersemble_export"])))

    for subject in args.nersemble_subjects:
        export_nersemble(paths, data_dir, subject, args.inversion_seq, args.inversion_frame, args.driving_seq)
    for name in args.itw_subjects:
        export_itw(paths, data_dir, name)
    if args.ckpt:
        export_checkpoint(args.ckpt, os.path.join(args.out, "checkpoints", "gaussian_maps.pt"))
    print(f"Done. Unpack {args.out} into the repository root.")


if __name__ == "__main__":
    main()
