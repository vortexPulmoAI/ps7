"""
Unit tests for deterministic clinical calculators:
Brock / PanCan multivariable logistic regression and ACR Lung-RADS Version 2022 rules.
"""

import unittest
from src.risk_engine import BrockCalculator, LungRADSClassifier


class TestBrockCalculator(unittest.TestCase):

    def test_canonical_screening_case(self):
        # 10 mm solid nodule in standard 62yo male baseline, 1 nodule, no emphysema, lower lobe
        res = BrockCalculator.calculate(
            diameter_max_mm=10.0,
            attenuation_type="solid",
            is_spiculated=False,
            is_upper_lobe=False,
            nodule_count=1,
            age=62.0,
            is_female=False,
            family_history=False,
            emphysema=False,
        )
        # Expected logit X ~ -3.4123 -> prob ~ 0.0319 (3.19%)
        self.assertAlmostEqual(res["probability"], 0.0319, places=3)
        self.assertAlmostEqual(res["brock_percent"], 3.19, places=1)
        self.assertIn("Low risk", res["bts_recommendation"])

    def test_high_risk_spiculated_case(self):
        # 20 mm part-solid spiculated upper-lobe nodule in 68yo female with emphysema
        res = BrockCalculator.calculate(
            diameter_max_mm=20.0,
            attenuation_type="part_solid",
            is_spiculated=True,
            is_upper_lobe=True,
            nodule_count=1,
            age=68.0,
            is_female=True,
            family_history=True,
            emphysema=True,
        )
        # Should be high risk (prob > 50%)
        self.assertGreater(res["probability"], 0.50)
        self.assertIn("Elevated risk", res["bts_recommendation"])

    def test_attenuation_weights(self):
        # Pure GGN should have lower logit than part-solid for identical geometry
        p_ggn = BrockCalculator.calculate(diameter_max_mm=12.0, attenuation_type="pure_ggn")
        p_subsolid = BrockCalculator.calculate(diameter_max_mm=12.0, attenuation_type="part_solid")
        self.assertLess(p_ggn["probability"], p_subsolid["probability"])


class TestLungRADSClassifier(unittest.TestCase):

    def test_solid_nodule_size_tiers(self):
        # Category 2: < 6 mm
        c2 = LungRADSClassifier.classify(mean_diameter_mm=5.8, nodule_type="solid")
        self.assertEqual(c2["category"], "2")
        self.assertEqual(c2["risk_range"], "<1%")

        # Category 3: 6 to < 8 mm
        c3 = LungRADSClassifier.classify(mean_diameter_mm=6.5, nodule_type="solid")
        self.assertEqual(c3["category"], "3")
        self.assertEqual(c3["risk_range"], "1-2%")

        # Category 4A: 8 to < 15 mm
        c4a = LungRADSClassifier.classify(mean_diameter_mm=10.0, nodule_type="solid")
        self.assertEqual(c4a["category"], "4A")
        self.assertEqual(c4a["risk_range"], "5-15%")

        # Category 4B: >= 15 mm
        c4b = LungRADSClassifier.classify(mean_diameter_mm=16.0, nodule_type="solid")
        self.assertEqual(c4b["category"], "4B")
        self.assertEqual(c4b["risk_range"], ">15%")

    def test_part_solid_nodule_tiers(self):
        # Category 2: total < 6 mm
        c2 = LungRADSClassifier.classify(mean_diameter_mm=5.0, nodule_type="part_solid", solid_core_diameter_mm=2.0)
        self.assertEqual(c2["category"], "2")

        # Category 3: total >= 6 mm with solid core < 6 mm
        c3 = LungRADSClassifier.classify(mean_diameter_mm=10.0, nodule_type="part_solid", solid_core_diameter_mm=4.0)
        self.assertEqual(c3["category"], "3")

        # Category 4A: solid core 6 to < 8 mm
        c4a = LungRADSClassifier.classify(mean_diameter_mm=12.0, nodule_type="part_solid", solid_core_diameter_mm=7.0)
        self.assertEqual(c4a["category"], "4A")

        # Category 4B: solid core >= 8 mm
        c4b = LungRADSClassifier.classify(mean_diameter_mm=15.0, nodule_type="part_solid", solid_core_diameter_mm=9.5)
        self.assertEqual(c4b["category"], "4B")

    def test_pure_ggn_tiers(self):
        # Category 2: < 30 mm
        c2 = LungRADSClassifier.classify(mean_diameter_mm=20.0, nodule_type="pure_ggn")
        self.assertEqual(c2["category"], "2")

        # Category 3: >= 30 mm
        c3 = LungRADSClassifier.classify(mean_diameter_mm=32.0, nodule_type="pure_ggn")
        self.assertEqual(c3["category"], "3")

    def test_4x_spiculation_upgrade(self):
        # Category 3 nodule with marked spiculation upgraded to 4X
        c4x = LungRADSClassifier.classify(mean_diameter_mm=7.0, nodule_type="solid", is_spiculated=True)
        self.assertEqual(c4x["category"], "4X")
        self.assertIn(">15%", c4x["risk_range"])

    def test_calibrated_malignancy_model_and_triad_output(self):
        from src.risk_engine import CalibratedMalignancyModel, assess_risk

        # 1. Benign-leaning nodule features
        benign_features = {
            "mean_diameter_mm": 5.0,
            "d_long_mm": 5.0,
            "nodule_type": "solid",
            "is_spiculated": False,
            "spiculation": 1.0,
            "subtlety": 4.0,
            "sphericity": 5.0,
            "margin": 5.0,
            "lobulation": 1.0,
            "texture": 5.0,
            "calcification": 6.0,
            "roi_count": 3.0,
            "age": 55.0,
        }
        res_benign = assess_risk(benign_features)
        self.assertIn("risk_percent", res_benign)
        self.assertIn("brock_percent", res_benign)
        self.assertIn("lung_rads_category", res_benign)
        self.assertIn("confidence_flag", res_benign)
        self.assertLess(res_benign["risk_percent"], 35.0)
        self.assertEqual(res_benign["lung_rads_category"], "2")

        # 2. Malignant-leaning nodule features (large, spiculated, lobulated)
        malignant_features = {
            "mean_diameter_mm": 18.0,
            "d_long_mm": 20.0,
            "nodule_type": "part_solid",
            "is_spiculated": True,
            "spiculation": 5.0,
            "subtlety": 5.0,
            "sphericity": 1.0,
            "margin": 2.0,
            "lobulation": 4.0,
            "texture": 5.0,
            "calcification": 6.0,
            "roi_count": 12.0,
            "is_upper_lobe": True,
            "age": 70.0,
            "emphysema": True,
        }
        res_mal = assess_risk(malignant_features)
        self.assertGreater(res_mal["risk_percent"], 60.0)
        self.assertGreater(res_mal["brock_percent"], 50.0)
        self.assertEqual(res_mal["lung_rads_category"], "4X")

    def test_generate_diagnostic_json(self):
        from src.risk_engine import generate_diagnostic_json, assess_risk

        features = {
            "mean_diameter_mm": 8.6,
            "d_long_mm": 9.4,
            "d_short_mm": 7.8,
            "volume_mm3": 412.6,
            "nodule_type": "part_solid",
            "solid_core_diameter_mm": 6.1,
            "is_spiculated": True,
        }
        risk_res = assess_risk(features)
        payload = generate_diagnostic_json(
            candidate_id="NOD_0042",
            centroid_world_mm=(-64.2, 42.8, -180.5),
            dimensions={
                "volume_mm3": 412.6,
                "d_long_mm": 9.4,
                "d_short_mm": 7.8,
                "d_mean_mm": 8.6,
            },
            morphology={
                "type": "Part-Solid",
                "solid_core_diameter_mm": 6.1,
                "is_spiculated": True,
            },
            risk_assessment=risk_res,
        )

        self.assertEqual(payload["candidate_id"], "NOD_0042")
        self.assertEqual(payload["centroid_world_mm"], {"x": -64.2, "y": 42.8, "z": -180.5})
        self.assertEqual(payload["dimensions"]["volume_mm3"], 412.6)
        self.assertEqual(payload["dimensions"]["mean_diameter_mm"], 8.6)
        self.assertEqual(payload["morphology"]["type"], "Part-Solid")
        self.assertTrue(payload["morphology"]["spiculation_detected"])
        self.assertIn("malignancy_probability", payload["risk_assessment"])
        self.assertIn("lung_rads_category", payload["risk_assessment"])


if __name__ == "__main__":
    unittest.main()
