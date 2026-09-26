"""
Integration tests for real LUNA16 and LIDC dataset parsers
testing against real files in C:/PARTH/ps7/dataset.
"""

import os
import unittest
from src.dataset_parser import (
    LUNA16AnnotationParser,
    LUNA16CandidateParser,
    LIDCXMLParser,
    LUNALungMaskLoader
)

DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")


class TestDatasetParsers(unittest.TestCase):

    def test_luna16_annotations_csv(self):
        csv_path = os.path.join(DATASET_ROOT, "annotations.csv")
        self.assertTrue(os.path.exists(csv_path), "annotations.csv must exist")

        parser = LUNA16AnnotationParser(csv_path)
        self.assertEqual(len(parser), 1186, "LUNA16 contains exactly 1,186 consensus nodules")

        nodules = parser.get_all()
        first = nodules[0]
        self.assertIn("seriesuid", first)
        self.assertIn("coord_x", first)
        self.assertIn("coord_y", first)
        self.assertIn("coord_z", first)
        self.assertIn("diameter_mm", first)
        self.assertGreater(first["diameter_mm"], 0.0)

        # Test querying by series
        series_nodules = parser.get_by_series(first["seriesuid"])
        self.assertGreaterEqual(len(series_nodules), 1)

    def test_luna16_candidates_csv(self):
        csv_path = os.path.join(DATASET_ROOT, "candidates.csv")
        self.assertTrue(os.path.exists(csv_path), "candidates.csv must exist")

        parser = LUNA16CandidateParser(csv_path)
        # Test streaming sample
        sample = parser.get_sample_candidates(max_rows=100)
        self.assertEqual(len(sample), 100)
        for cand in sample:
            self.assertIn(cand["label"], (0, 1))

    def test_lidc_xml_parsing(self):
        xml_dir = os.path.join(DATASET_ROOT, "LIDC-XML-only", "tcia-lidc-xml")
        self.assertTrue(os.path.exists(xml_dir), "LIDC XML directory must exist")

        # Find first available XML
        xml_file = None
        for root, _, files in os.walk(xml_dir):
            for f in files:
                if f.endswith(".xml"):
                    xml_file = os.path.join(root, f)
                    break
            if xml_file:
                break

        self.assertIsNotNone(xml_file, "At least one XML file should be found")

        parser = LIDCXMLParser()
        result = parser.parse_file(xml_file)

        self.assertIn("series_uid", result)
        self.assertIn("nodules", result)
        self.assertGreater(len(result["series_uid"]), 0)

        if len(result["nodules"]) > 0:
            first_nod = result["nodules"][0]
            self.assertIn("rois", first_nod)
            self.assertIn("characteristics", first_nod)
            self.assertIn("option_b_label", first_nod)
            # If characteristics has malignancy rating, option_b_label must be 0, 1, or None
            if first_nod["characteristics"] and "malignancy" in first_nod["characteristics"]:
                self.assertIn(first_nod["option_b_label"], (0, 1, None))

    def test_luna_lung_mask_loader(self):
        lungs_dir = os.path.join(DATASET_ROOT, "seg-lungs-LUNA16")
        self.assertTrue(os.path.exists(lungs_dir), "seg-lungs-LUNA16 directory must exist")

        loader = LUNALungMaskLoader(lungs_dir)
        # Find first .mhd file in directory
        mhd_file = None
        for root, _, files in os.walk(lungs_dir):
            for f in files:
                if f.endswith(".mhd"):
                    mhd_file = os.path.join(root, f)
                    break
            if mhd_file:
                break

        self.assertIsNotNone(mhd_file, "At least one lung mask MHD should be found")
        series_uid = os.path.splitext(os.path.basename(mhd_file))[0]

        mask, meta = loader.load_mask_for_series(series_uid)
        self.assertEqual(mask.ndim, 3, "Mask must be 3D [dimZ, dimY, dimX]")
        self.assertIn("spacing_xyz", meta)
        self.assertEqual(len(meta["dim_size_xyz"]), 3)


if __name__ == "__main__":
    unittest.main()
