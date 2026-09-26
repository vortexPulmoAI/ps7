"""
Test script to verify real 3D CT scan loading, lung mask pairing,
and candidate patch extraction from LUNA16 subset0.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.luna_dataset import LUNA16SubsetIngestor


def test_subset0():
    print("[*] Testing LUNA16 subset0 Ingestion...")
    ingestor = LUNA16SubsetIngestor(dataset_root="dataset")
    scans = ingestor.discover_subset_scans("subset0")
    print(f"    Discovered {len(scans)} scans in subset0.")

    # Find scans in subset0 that have annotated nodules in annotations.csv
    matching_scans = []
    for s in scans:
        uid = os.path.splitext(os.path.basename(s))[0]
        nodules = ingestor.anno_parser.get_by_series(uid)
        if nodules:
            matching_scans.append((s, nodules))

    print(f"    Found {len(matching_scans)} scans in subset0 with consensus nodules.")
    if not matching_scans:
        print("    No matching annotated scans found in subset0.")
        return

    # Process first matching scan
    sample_scan, nodules = matching_scans[0]
    uid = os.path.splitext(os.path.basename(sample_scan))[0]
    print(f"\n[*] Processing sample scan: {uid}")
    print(f"    Consensus nodules in this scan: {len(nodules)}")
    for n in nodules:
        print(f"    - Nodule at ({n['coord_x']:.1f}, {n['coord_y']:.1f}, {n['coord_z']:.1f}) mm, diameter: {n['diameter_mm']:.1f} mm")

    scan_data = ingestor.load_scan(sample_scan)
    vol_hu = scan_data["volume_hu"]
    spacing = scan_data["spacing_zyx"]
    lung_mask = scan_data["lung_mask"]

    print(f"\n    Raw CT Volume Loaded:")
    print(f"    - Shape:       {vol_hu.shape} (Z, Y, X)")
    print(f"    - Spacing:     {spacing} (sz, sy, sx) mm")
    print(f"    - HU Range:    [{vol_hu.min()}, {vol_hu.max()}] HU")
    print(f"    - Paired Lung Mask: {'Loaded (' + str(lung_mask.shape) + ')' if lung_mask is not None else 'Not Found'}")

    # Extract 3D isotropic patches
    print(f"\n[*] Extracting 3D isotropic patches for annotated nodules...")
    patches = ingestor.extract_nodule_patches_for_series(
        sample_scan,
        target_spacing=(1.0, 1.0, 1.0),
        inner_size=(32, 32, 32),
        context_size=(64, 64, 64)
    )

    print(f"[OK] Extracted {len(patches)} nodule patch sample(s):")
    for idx, p in enumerate(patches):
        print(f"    Sample {idx + 1}:")
        print(f"    - World Coordinates: {p['world_coord']} mm")
        print(f"    - Resampled Center:  {p['center_zyx']}")
        print(f"    - Inner Patch Shape: {p['inner_patch'].shape} (Range: [{p['inner_patch'].min():.2f}, {p['inner_patch'].max():.2f}])")
        print(f"    - Context Patch:     {p['context_patch'].shape} (Range: [{p['context_patch'].min():.2f}, {p['context_patch'].max():.2f}])")


if __name__ == "__main__":
    test_subset0()
