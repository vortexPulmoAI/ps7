"""
Batch patch extraction script for LUNA16 subset0.
Extracts 32^3 and 64^3 3D patches for:
- True consensus nodules (Class 1, positive candidates)
- Hard negative false alarm proposals (Class 0, negative candidates)
Saves extracted volumetric arrays to dataset/processed_patches/subset0/
and compiles index metadata to dataset/processed_patches/subset0_index.csv.
"""

import os
import sys
import csv
import glob
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.luna_dataset import LUNA16SubsetIngestor


def prepare_subset0_patches(max_scans=None, neg_per_scan=10):
    output_dir = os.path.join("dataset", "processed_patches", "subset0")
    os.makedirs(output_dir, exist_ok=True)
    index_csv = os.path.join("dataset", "processed_patches", "subset0_index.csv")

    ingestor = LUNA16SubsetIngestor(dataset_root="dataset")
    scans = ingestor.discover_subset_scans("subset0")
    if max_scans:
        scans = scans[:max_scans]

    print(f"[*] Processing {len(scans)} scans from subset0...")
    index_records = []
    total_pos = 0
    total_neg = 0

    for i, scan_path in enumerate(scans):
        series_uid = os.path.splitext(os.path.basename(scan_path))[0]
        try:
            # 1. Extract positive nodule patches
            pos_patches = ingestor.extract_nodule_patches_for_series(
                scan_path,
                target_spacing=(1.0, 1.0, 1.0),
                inner_size=(32, 32, 32),
                context_size=(64, 64, 64)
            )

            for p_idx, p in enumerate(pos_patches):
                sample_id = f"{series_uid[:16]}_pos_{p_idx}"
                npz_path = os.path.join(output_dir, f"{sample_id}.npz")
                np.savez_compressed(
                    npz_path,
                    inner_patch=p["inner_patch"].astype(np.float32),
                    context_patch=p["context_patch"].astype(np.float32),
                    label=1,
                    diameter_mm=p["diameter_mm"],
                    world_coord=p["world_coord"],
                )
                index_records.append({
                    "sample_id": sample_id,
                    "series_uid": series_uid,
                    "label": 1,
                    "diameter_mm": p["diameter_mm"],
                    "file_path": npz_path,
                })
                total_pos += 1

            # 2. Extract negative candidate proposals from candidates.csv
            if ingestor.cand_parser and neg_per_scan > 0:
                cands = ingestor.cand_parser.get_by_series(series_uid)
                neg_cands = [c for c in cands if c["label"] == 0][:neg_per_scan]

                if neg_cands:
                    scan = ingestor.load_scan(scan_path)
                    vol_hu = scan["volume_hu"]
                    spacing_zyx = scan["spacing_zyx"]
                    from src.preprocessing import resample_isotropic, apply_hu_window
                    resampled, _ = resample_isotropic(vol_hu, spacing_zyx, target_spacing=(1.0, 1.0, 1.0))
                    normalized = apply_hu_window(resampled, window="lung_screening", normalize=True)

                    ox, oy, oz = scan["origin_xyz"]
                    sx, sy, sz = spacing_zyx[2], spacing_zyx[1], spacing_zyx[0]

                    for n_idx, nc in enumerate(neg_cands):
                        world_coord = (nc["coord_x"], nc["coord_y"], nc["coord_z"])
                        vox_x = (nc["coord_x"] - ox) / sx
                        vox_y = (nc["coord_y"] - oy) / sy
                        vox_z = (nc["coord_z"] - oz) / sz

                        iso_center = (vox_z * sz, vox_y * sy, vox_x * sx)
                        dual_p = ingestor.patch_extractor.extract_dual_scale_patches(
                            normalized,
                            iso_center,
                            inner_size=(32, 32, 32),
                            context_size=(64, 64, 64),
                            pad_value=0.0
                        )

                        sample_id = f"{series_uid[:16]}_neg_{n_idx}"
                        npz_path = os.path.join(output_dir, f"{sample_id}.npz")
                        np.savez_compressed(
                            npz_path,
                            inner_patch=dual_p["inner_patch"].astype(np.float32),
                            context_patch=dual_p["context_patch"].astype(np.float32),
                            label=0,
                            diameter_mm=0.0,
                            world_coord=world_coord,
                        )
                        index_records.append({
                            "sample_id": sample_id,
                            "series_uid": series_uid,
                            "label": 0,
                            "diameter_mm": 0.0,
                            "file_path": npz_path,
                        })
                        total_neg += 1

        except Exception as e:
            print(f"    [!] Error on scan {series_uid[:16]}: {e}")
            continue

        if (i + 1) % 10 == 0 or (i + 1) == len(scans):
            print(f"    Processed {i + 1}/{len(scans)} scans... (Positives: {total_pos}, Negatives: {total_neg})")

    # Save index CSV
    if index_records:
        with open(index_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "series_uid", "label", "diameter_mm", "file_path"])
            writer.writeheader()
            writer.writerows(index_records)

    print(f"\n[OK] Patch Extraction Complete:")
    print(f"    - True Positive Patches (Class 1): {total_pos}")
    print(f"    - False Alarm Patches   (Class 0): {total_neg}")
    print(f"    - Saved Patches to: {output_dir}")
    print(f"    - Index CSV saved:  {index_csv}")


if __name__ == "__main__":
    # Run on first 10 scans for rapid verification
    prepare_subset0_patches(max_scans=10, neg_per_scan=5)
