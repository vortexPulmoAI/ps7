"""
Parsers for LUNA16 and LIDC-IDRI local dataset repositories.
Parses annotations.csv, candidates.csv, LIDC XML annotations (contour edge maps and semantic ratings),
and LUNA16 lung mask volumes (.mhd / .zraw).
"""

import os
import csv
import xml.etree.ElementTree as ET
import numpy as np
from .preprocessing import load_mhd_mask


class LUNA16AnnotationParser:
    """
    Parses LUNA16 annotations.csv containing consensus true nodules >= 3 mm.
    Schema: seriesuid, coordX, coordY, coordZ, diameter_mm
    """

    def __init__(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Annotations file not found: {csv_path}")
        self.csv_path = csv_path
        self._nodules = []
        self._load()

    def _load(self):
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._nodules.append({
                    "seriesuid": row["seriesuid"],
                    "coord_x": float(row["coordX"]),
                    "coord_y": float(row["coordY"]),
                    "coord_z": float(row["coordZ"]),
                    "diameter_mm": float(row["diameter_mm"]),
                })

    def __len__(self):
        return len(self._nodules)

    def get_all(self):
        return self._nodules

    def get_by_series(self, seriesuid):
        return [n for n in self._nodules if n["seriesuid"] == seriesuid]

    def get_unique_series_uids(self):
        return sorted(list(set(n["seriesuid"] for n in self._nodules)))


class LUNA16CandidateParser:
    """
    Parses LUNA16 candidates.csv containing candidate proposals with binary labels:
        0 = non-nodule (false alarm)
        1 = nodule (true positive candidate)
    Schema: seriesuid, coordX, coordY, coordZ, class
    """

    def __init__(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Candidates file not found: {csv_path}")
        self.csv_path = csv_path

    def get_by_series(self, seriesuid):
        """
        Extracts all candidates for a specific seriesuid.
        """
        results = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["seriesuid"] == seriesuid:
                    results.append({
                        "seriesuid": row["seriesuid"],
                        "coord_x": float(row["coordX"]),
                        "coord_y": float(row["coordY"]),
                        "coord_z": float(row["coordZ"]),
                        "label": int(row["class"]),
                    })
        return results

    def get_sample_candidates(self, max_rows=1000):
        """
        Reads a limited sample of candidate records for fast verification.
        """
        sample = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                sample.append({
                    "seriesuid": row["seriesuid"],
                    "coord_x": float(row["coordX"]),
                    "coord_y": float(row["coordY"]),
                    "coord_z": float(row["coordZ"]),
                    "label": int(row["class"]),
                })
        return sample


class LIDCXMLParser:
    """
    Parses LIDC-IDRI radiologist XML annotation files.
    Extracts 3D nodule contours (polygon edge maps per slice) and 8 semantic ratings:
        - subtlety, internalStructure, calcification, sphericity, margin,
          lobulation, spiculation, texture, malignancy (1-5 scale)
    Applies Option B label scheme (drops indeterminate score 3; 1-2 = 0, 4-5 = 1).
    """

    def __init__(self):
        self.ns = "{http://www.nih.gov}"

    def _strip_ns(self, tag):
        if "}" in tag:
            return tag.split("}", 1)[1]
        return tag

    def parse_file(self, xml_path):
        """
        Parses an individual LIDC XML annotation file.

        Returns:
            dict containing:
                series_uid: str
                study_uid: str
                nodules: list of nodule dicts with ROIs, semantic ratings, and Option B label
        """
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found: {xml_path}")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Find SeriesInstanceUid
        series_uid = ""
        study_uid = ""
        for elem in root.iter():
            tag = self._strip_ns(elem.tag)
            if tag == "SeriesInstanceUid" and elem.text:
                series_uid = elem.text.strip()
            elif tag == "StudyInstanceUID" and elem.text:
                study_uid = elem.text.strip()

        nodules = []

        # Find reading sessions and unblindedReadNodules
        for session in root.iter():
            if self._strip_ns(session.tag) != "readingSession":
                continue

            for nodule_elem in session.iter():
                if self._strip_ns(nodule_elem.tag) != "unblindedReadNodule":
                    continue

                nodule_id = None
                characteristics = None
                rois = []

                for child in nodule_elem:
                    child_tag = self._strip_ns(child.tag)
                    if child_tag == "noduleID" and child.text:
                        nodule_id = child.text.strip()
                    elif child_tag == "characteristics":
                        characteristics = {}
                        for char_elem in child:
                            c_tag = self._strip_ns(char_elem.tag)
                            try:
                                characteristics[c_tag] = float(char_elem.text.strip())
                            except (ValueError, TypeError):
                                pass
                    elif child_tag == "roi":
                        z_pos = None
                        sop_uid = ""
                        edge_map = []
                        for roi_child in child:
                            r_tag = self._strip_ns(roi_child.tag)
                            if r_tag == "imageZposition" and roi_child.text:
                                try:
                                    z_pos = float(roi_child.text.strip())
                                except ValueError:
                                    pass
                            elif r_tag == "imageSOP_UID" and roi_child.text:
                                sop_uid = roi_child.text.strip()
                            elif r_tag == "edgeMap":
                                x_coord, y_coord = None, None
                                for coord in roi_child:
                                    c_tag = self._strip_ns(coord.tag)
                                    if c_tag == "xCoord" and coord.text:
                                        x_coord = int(coord.text.strip())
                                    elif c_tag == "yCoord" and coord.text:
                                        y_coord = int(coord.text.strip())
                                if x_coord is not None and y_coord is not None:
                                    edge_map.append((x_coord, y_coord))

                        if edge_map or z_pos is not None:
                            rois.append({
                                "z_position": z_pos,
                                "sop_uid": sop_uid,
                                "polygon": edge_map,
                            })

                if rois or characteristics:
                    # Calculate Option B label
                    option_b_label = None
                    if characteristics and "malignancy" in characteristics:
                        mal_score = characteristics["malignancy"]
                        if mal_score in (1.0, 2.0):
                            option_b_label = 0  # Benign
                        elif mal_score in (4.0, 5.0):
                            option_b_label = 1  # Malignant
                        else:
                            option_b_label = None  # Drop score 3 (indeterminate)

                    nodules.append({
                        "nodule_id": nodule_id,
                        "characteristics": characteristics,
                        "rois": rois,
                        "option_b_label": option_b_label,
                    })

        return {
            "series_uid": series_uid,
            "study_uid": study_uid,
            "nodules": nodules,
            "xml_path": xml_path,
        }


class LUNALungMaskLoader:
    """
    Loads lung segmentation masks from dataset/seg-lungs-LUNA16.
    """

    def __init__(self, masks_dir):
        if not os.path.exists(masks_dir):
            raise FileNotFoundError(f"Lungs directory not found: {masks_dir}")
        self.masks_dir = masks_dir

    def find_mhd_for_series(self, seriesuid):
        filename = f"{seriesuid}.mhd"
        direct_path = os.path.join(self.masks_dir, filename)
        if os.path.exists(direct_path):
            return direct_path
        # Search recursively
        for root, _, files in os.walk(self.masks_dir):
            if filename in files:
                return os.path.join(root, filename)
        return None

    def load_mask_for_series(self, seriesuid):
        mhd_path = self.find_mhd_for_series(seriesuid)
        if not mhd_path:
            raise FileNotFoundError(f"Lung mask MHD not found for series: {seriesuid}")
        return load_mhd_mask(mhd_path)
