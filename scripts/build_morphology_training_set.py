"""
NEW SCRIPT — builds a malignancy-training CSV from REAL computed morphology
and density features, instead of LIDC's hand-typed semantic ratings.

For each LIDC-XML nodule whose series_uid matches a scan present in
dataset/subset0, this:
  1. Loads the real CT volume + spacing (via LUNA16SubsetIngestor).
  2. Rasterizes that nodule's radiologist polygon ROIs into a 3D binary mask
     at the scan's native resolution.
  3. Resamples the mask to 1.0mm isotropic (matching the volume).
  4. Runs your real morphology.py / density.py on that mask.
  5. Writes one row per nodule: computed features + Option B label.

*** ADAPT BEFORE RUNNING ***
This script assumes LIDCXMLParser's roi dicts expose polygon vertices under
a key such as `edge_map` or `points` (a list of (x_pixel, y_pixel) tuples)
alongside `z_position`. Check the actual return shape of
`LIDCXMLParser().parse_file(...)['nodules'][i]['rois'][j]` in your
src/dataset_parser.py and adjust `_get_polygon_points()` below to match the
real key name before running — this is the one place field names are
guessed rather than confirmed from your code.
"""

import os
import sys
import csv
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.dataset_parser import LIDCXMLParser
from src.luna_dataset import LUNA16SubsetIngestor
from src.preprocessing import resample_isotropic
from src.morphology import (
    calculate_volume, calculate_diameters_and_axes,
    calculate_sphericity, calculate_spiculation_index
)
from src.density import decompose_density

try:
    from skimage.draw import polygon as sk_polygon
except ImportError:
    raise SystemExit("Missing dependency: pip install scikit-image")


def _get_polygon_points(roi):
    """ADAPT: return list of (x_pixel, y_pixel) for this ROI slice.
    Guessed key names below — check your parser's actual output shape."""
    for key in ("edge_map", "points", "edgeMap", "polygon"):
        if key in roi and roi[key]:
            return roi[key]
    raise KeyError(
        f"Could not find polygon vertices in roi dict. Keys present: {list(roi.keys())}. "
        f"Update _get_polygon_points() with the correct key."
    )


def rasterize_nodule_mask(rois, vol_shape_zyx, spacing_zyx, origin_xyz):
    """Turn a list of per-slice polygon ROIs into a 3D binary mask in
    the scan's native voxel grid."""
    mask = np.zeros(vol_shape_zyx, dtype=np.uint8)
    sz = spacing_zyx[0]
    oz = origin_xyz[2]

    for roi in rois:
        z_mm = roi.get("z_position")
        if z_mm is None:
            continue
        z_idx = int(round((z_mm - oz) / sz))
        if not (0 <= z_idx < vol_shape_zyx[0]):
            continue

        pts = _get_polygon_points(roi)
        if len(pts) < 3:
            continue
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        rr, cc = sk_polygon(ys, xs, shape=vol_shape_zyx[1:])
        mask[z_idx, rr, cc] = 1

    return mask


def build_training_set(xml_root="dataset/LIDC-XML-only",
                        dataset_root="dataset",
                        subset_name="subset0",
                        output_csv="dataset/lidc_morphology_features.csv"):
    print("[*] Building REAL morphology/density feature training set...")
    parser = LIDCXMLParser()
    ingestor = LUNA16SubsetIngestor(dataset_root=dataset_root)
    scans = ingestor.discover_subset_scans(subset_name)
    scan_uids = {os.path.splitext(os.path.basename(s))[0]: s for s in scans}
    print(f"    {len(scan_uids)} scans available in {subset_name}")

    import glob
    xml_files = glob.glob(os.path.join(xml_root, "**", "*.xml"), recursive=True)

    rows = []
    matched, skipped = 0, 0

    for xml_path in xml_files:
        try:
            res = parser.parse_file(xml_path)
        except Exception:
            continue
        series_uid = res["series_uid"]
        if series_uid not in scan_uids:
            continue  # only process nodules we have real pixel data for

        scan_data = ingestor.load_scan(scan_uids[series_uid])
        vol_hu = scan_data["volume_hu"]
        spacing = scan_data["spacing_zyx"]
        origin = scan_data["origin_xyz"]

        for nod in res["nodules"]:
            opt_b = nod["option_b_label"]
            if opt_b is None:
                skipped += 1
                continue  # ambiguous score=3, dropped per Option B

            try:
                mask_native = rasterize_nodule_mask(nod["rois"], vol_hu.shape, spacing, origin)
                if mask_native.sum() < 4:
                    skipped += 1
                    continue

                mask_iso, iso_spacing = resample_isotropic(
                    mask_native.astype(np.float32), spacing,
                    target_spacing=(1.0, 1.0, 1.0), is_mask=True
                )
                mask_iso = (mask_iso > 0.5).astype(np.uint8)

                zc, yc, xc = np.array(np.where(mask_iso)).mean(axis=1).astype(int)
                r = 24
                z0, z1 = max(0, zc - r), min(mask_iso.shape[0], zc + r)
                y0, y1 = max(0, yc - r), min(mask_iso.shape[1], yc + r)
                x0, x1 = max(0, xc - r), min(mask_iso.shape[2], xc + r)
                mask_patch = mask_iso[z0:z1, y0:y1, x0:x1]

                vol_hu_iso, _ = resample_isotropic(vol_hu, spacing, target_spacing=(1.0, 1.0, 1.0))
                hu_patch = vol_hu_iso[z0:z1, y0:y1, x0:x1]

                target_iso = (1.0, 1.0, 1.0)
                volume_mm3 = calculate_volume(mask_patch, target_iso)
                axes = calculate_diameters_and_axes(mask_patch, target_iso)
                sphericity = calculate_sphericity(mask_patch, target_iso)
                spic_idx = calculate_spiculation_index(mask_patch, target_iso)
                density = decompose_density(hu_patch, mask_patch, target_iso)

                rows.append({
                    "series_uid": series_uid,
                    "nodule_id": nod["nodule_id"],
                    "volume_mm3": volume_mm3,
                    "d_mean_mm": axes["d_mean_mm"],
                    "d_long_mm": axes["d_long_mm"],
                    "sphericity": sphericity,
                    "spiculation_index": spic_idx,
                    "solid_ratio": density["solid_core_ratio"],
                    "ggo_ratio": density["ggo_ratio"],
                    "nodule_type": density["nodule_type"],
                    "option_b_label": opt_b,
                })
                matched += 1
            except Exception as e:
                print(f"    [!] Skipped nodule {nod.get('nodule_id')} in {series_uid[:16]}: {e}")
                skipped += 1
                continue

    fieldnames = ["series_uid", "nodule_id", "volume_mm3", "d_mean_mm", "d_long_mm",
                  "sphericity", "spiculation_index", "solid_ratio", "ggo_ratio",
                  "nodule_type", "option_b_label"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Built {matched} real feature rows ({skipped} skipped) -> {output_csv}")
    print("    NOTE: this only covers nodules whose series_uid is present in subset0.")
    print("    Download more LUNA16 subsets to grow this training set.")
    return rows


if __name__ == "__main__":
    build_training_set()
