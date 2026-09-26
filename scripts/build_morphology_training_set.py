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


def rasterize_nodule_mask_local(rois, crop_shape_zyx, spacing_zyx, origin_zyx_mm, crop_offset_zyx_idx):
    """Same as rasterize_nodule_mask, but rasterizes directly into a small
    pre-cropped local volume instead of the full scan — this is what makes
    per-nodule processing fast instead of resampling the whole 512x512xN scan.
    Each ROI's own true z-slice is preserved (no flattening to one slice)."""
    mask = np.zeros(crop_shape_zyx, dtype=np.uint8)
    sz = spacing_zyx[0]
    oz_mm = origin_zyx_mm[0]
    z0_off, y0_off, x0_off = crop_offset_zyx_idx

    for roi in rois:
        z_mm = roi.get("z_position")
        if z_mm is None:
            continue
        z_idx_global = int(round((z_mm - oz_mm) / sz))
        z_idx_local = z_idx_global - z0_off
        if not (0 <= z_idx_local < crop_shape_zyx[0]):
            continue

        pts = _get_polygon_points(roi)
        if len(pts) < 3:
            continue
        xs = np.array([p[0] - x0_off for p in pts])
        ys = np.array([p[1] - y0_off for p in pts])
        try:
            rr, cc = sk_polygon(ys, xs, shape=crop_shape_zyx[1:])
            mask[z_idx_local, rr, cc] = 1
        except Exception:
            continue
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
                # --- FIX: crop a small NATIVE-resolution subvolume around the
                # nodule FIRST, then resample only that crop. The old version
                # resampled the entire ~512x512x300 scan per nodule (30-90s
                # each via scipy spline_filter1d) which is why it looked stuck.
                z_positions = [r["z_position"] for r in nod["rois"] if r.get("z_position") is not None]
                if not z_positions:
                    skipped += 1
                    continue
                z_mm = float(np.mean(z_positions))
                # rough centroid from first ROI's polygon, native pixel coords
                pts = _get_polygon_points(nod["rois"][0])
                xc_px = float(np.mean([p[0] for p in pts]))
                yc_px = float(np.mean([p[1] for p in pts]))

                sz, sy, sx = spacing
                zc = int(round((z_mm - origin[2]) / sz))
                yc = int(round(yc_px))  # ROI coords are already in this slice's pixel grid
                xc = int(round(xc_px))

                # native-resolution margin: ~40mm each side, converted to voxels per axis
                mz = max(3, int(np.ceil(40.0 / sz)))
                my = max(3, int(np.ceil(40.0 / sy)))
                mx = max(3, int(np.ceil(40.0 / sx)))
                z0n, z1n = max(0, zc - mz), min(vol_hu.shape[0], zc + mz)
                y0n, y1n = max(0, yc - my), min(vol_hu.shape[1], yc + my)
                x0n, x1n = max(0, xc - mx), min(vol_hu.shape[2], xc + mx)
                if z1n <= z0n or y1n <= y0n or x1n <= x0n:
                    skipped += 1
                    continue

                hu_crop_native = vol_hu[z0n:z1n, y0n:y1n, x0n:x1n]

                # rasterize the mask directly into this small native crop's frame
                mask_crop_native = rasterize_nodule_mask_local(
                    nod["rois"], hu_crop_native.shape, spacing,
                    origin_zyx_mm=(origin[2], origin[1], origin[0]),
                    crop_offset_zyx_idx=(z0n, y0n, x0n)
                )
                if mask_crop_native.sum() < 4:
                    skipped += 1
                    continue

                # resample ONLY the small crop (fast — a few hundred voxels, not millions)
                hu_patch, _ = resample_isotropic(hu_crop_native, spacing,
                                                  target_spacing=(1.0, 1.0, 1.0), is_mask=False)
                mask_patch, _ = resample_isotropic(mask_crop_native.astype(np.float32), spacing,
                                                     target_spacing=(1.0, 1.0, 1.0), is_mask=True)
                mask_patch = (mask_patch > 0.5).astype(np.uint8)

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
