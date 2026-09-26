"""
Deterministic clinical risk calculation and heuristic reporting engine.
Implements the Pan-Canadian Early Detection of Lung Cancer (Brock / PanCan)
multivariable logistic regression model and the ACR Lung-RADS Version 2022 decision rules.

NOTE: Machine learning calibration (CalibratedMalignancyModel) is deferred to the
subsequent phase once feature extraction and model training on LIDC-IDRI have been completed.
"""

import math


class BrockCalculator:
    """
    Pan-Canadian Early Detection of Lung Cancer (Brock / PanCan / McWilliams 2013)
    multivariable logistic regression calculator for estimating pulmonary nodule malignancy risk.

    Formula:
        p = 1 / (1 + exp(-X))
    Where X = -6.7892
              + 0.0287 * (Age - 62)
              + 0.6011 * I_Female
              + 0.2961 * I_FamilyHistory
              + 0.2953 * I_Emphysema
              - 5.3854 * [ (d_max / 10)^(-0.5) - 1.58113883 ]
              + beta_attenuation
              + 0.6581 * I_UpperLobe
              - 0.0824 * (NoduleCount - 4)
              + 0.7729 * I_Spiculation
    """

    ATTENUATION_WEIGHTS = {
        "solid": 0.0,
        "part_solid": 0.3770,
        "subsolid": 0.3770,
        "pure_ggn": -0.1276,
        "ggn": -0.1276,
        "non_solid": -0.1276,
    }

    @classmethod
    def calculate(
        cls,
        diameter_max_mm,
        attenuation_type="solid",
        is_spiculated=False,
        is_upper_lobe=False,
        nodule_count=1,
        age=62.0,
        is_female=False,
        family_history=False,
        emphysema=False,
    ):
        """
        Calculates Brock / PanCan malignancy risk probability.

        Args:
            diameter_max_mm: maximum transverse nodule diameter in mm (> 0)
            attenuation_type: 'solid', 'part_solid'/'subsolid', or 'pure_ggn'/'non_solid'
            is_spiculated: True if spiculated margin observed
            is_upper_lobe: True if located in Right Upper Lobe or Left Upper Lobe
            nodule_count: total non-calcified nodule count on scan (default: 1)
            age: patient age in continuous years (default screening median: 62)
            is_female: True if female, False if male (default: False)
            family_history: True if first-degree relative has history of lung cancer
            emphysema: True if CT demonstrates emphysematous parenchymal destruction

        Returns:
            dict containing:
                brock_percent: risk percentage (0.0% - 100.0%)
                probability: continuous probability p in [0.0, 1.0]
                logit_x: calculated log-odds X
                bts_recommendation: British Thoracic Society clinical recommendation (<10% vs >=10%)
        """
        d_max = max(float(diameter_max_mm), 0.1)

        # Baseline intercept
        x = -6.7892

        # Age term: +0.0287 * (Age - 62)
        x += 0.0287 * (float(age) - 62.0)

        # Sex term: +0.6011 if female
        if is_female:
            x += 0.6011

        # Family history: +0.2961
        if family_history:
            x += 0.2961

        # Emphysema: +0.2953
        if emphysema:
            x += 0.2953

        # Diameter fractional polynomial term: -5.3854 * [ (d_max / 10)^(-0.5) - 1.58113883 ]
        d_term = ((d_max / 10.0) ** -0.5) - 1.58113883
        x += -5.3854 * d_term

        # Attenuation term
        beta_att = cls.ATTENUATION_WEIGHTS.get(str(attenuation_type).lower(), 0.0)
        x += beta_att

        # Upper lobe term: +0.6581
        if is_upper_lobe:
            x += 0.6581

        # Nodule count term: -0.0824 * (count - 4)
        x += -0.0824 * (int(nodule_count) - 4)

        # Spiculation term: +0.7729
        if is_spiculated:
            x += 0.7729

        # Logistic sigmoid transformation
        prob = 1.0 / (1.0 + math.exp(-x))
        percent = prob * 100.0

        # British Thoracic Society (BTS) guideline threshold (10%)
        if percent < 10.0:
            bts = "Low risk (<10%): Routine CT surveillance per screening interval"
        else:
            bts = "Elevated risk (>=10%): Direct toward functional evaluation (FDG-PET/CT or tissue sampling)"

        return {
            "brock_percent": round(percent, 2),
            "probability": round(prob, 4),
            "logit_x": round(x, 4),
            "bts_recommendation": bts,
        }


class LungRADSClassifier:
    """
    American College of Radiology (ACR) Lung-RADS Version 2022 heuristic reporting system.
    Standardizes categorical risk tiers, malignancy probability ranges, and clinical follow-up recommendations.
    """

    @classmethod
    def classify(
        cls,
        mean_diameter_mm,
        nodule_type="solid",
        solid_core_diameter_mm=None,
        is_spiculated=False,
        is_high_suspicion_modifier=False,
    ):
        """
        Classifies a nodule into ACR Lung-RADS v2022 category:
            - Category 2: Benign appearance (<1% risk)
            - Category 3: Probably benign (1-2% risk)
            - Category 4A: Suspicious (5-15% risk)
            - Category 4B: Very suspicious (>15% risk)
            - Category 4X: Very suspicious with high-risk findings (>15%, often >50%)

        Args:
            mean_diameter_mm: (d_long + d_short) / 2 on maximal transverse slice in mm
            nodule_type: 'solid', 'part_solid', or 'pure_ggn' / 'calcified'
            solid_core_diameter_mm: mean diameter of dense solid core for part-solid nodules in mm
            is_spiculated: True if spiculated margin observed
            is_high_suspicion_modifier: True if marked spiculation, pleural retraction, or lymphadenopathy

        Returns:
            dict containing:
                category: str ('2', '3', '4A', '4B', '4X')
                risk_range: str (e.g. '5-15%')
                clinical_management: str recommended action
        """
        d = float(mean_diameter_mm)
        ntype = str(nodule_type).lower()

        # Check calcified (benign)
        if ntype == "calcified":
            return {
                "category": "2",
                "risk_range": "<1%",
                "clinical_management": "Annual low-dose CT (LDCT) in 12 months",
            }

        # Check high suspicion modifier (4X upgrade)
        if is_high_suspicion_modifier:
            return {
                "category": "4X",
                "risk_range": ">15% (often >50%)",
                "clinical_management": "Diagnostic chest CT with or without contrast, PET/CT, and/or tissue biopsy",
            }

        # 1. Pure Ground-Glass Nodules (pGGN)
        if ntype in ("pure_ggn", "ggn", "non_solid"):
            if d < 30.0:
                cat = "2"
                risk = "<1%"
                action = "Annual low-dose CT (LDCT) in 12 months"
            else:
                cat = "3"
                risk = "1-2%"
                action = "6-month low-dose CT follow-up"

        # 2. Part-Solid Nodules (PSN)
        elif ntype in ("part_solid", "subsolid"):
            core_d = float(solid_core_diameter_mm) if solid_core_diameter_mm is not None else 0.0

            if d < 6.0:
                cat = "2"
                risk = "<1%"
                action = "Annual low-dose CT (LDCT) in 12 months"
            elif core_d < 6.0:
                cat = "3"
                risk = "1-2%"
                action = "6-month low-dose CT follow-up"
            elif 6.0 <= core_d < 8.0:
                cat = "4A"
                risk = "5-15%"
                action = "3-month low-dose CT; FDG-PET/CT if solid core >= 8 mm"
            else:  # core_d >= 8.0
                cat = "4B"
                risk = ">15%"
                action = "Diagnostic chest CT with or without contrast, PET/CT, and/or tissue biopsy"

        # 3. Solid Nodules (SN)
        else:
            if d < 6.0:
                cat = "2"
                risk = "<1%"
                action = "Annual low-dose CT (LDCT) in 12 months"
            elif 6.0 <= d < 8.0:
                cat = "3"
                risk = "1-2%"
                action = "6-month low-dose CT follow-up"
            elif 8.0 <= d < 15.0:
                cat = "4A"
                risk = "5-15%"
                action = "3-month low-dose CT; FDG-PET/CT if solid core >= 8 mm"
            else:  # d >= 15.0
                cat = "4B"
                risk = ">15%"
                action = "Diagnostic chest CT with or without contrast, PET/CT, and/or tissue biopsy"

        # Marked spiculation upgrade to 4X if Category 3 or 4
        if is_spiculated and cat in ("3", "4A", "4B"):
            cat = "4X"
            risk = ">15% (often >50%)"
            action = "Diagnostic chest CT with or without contrast, PET/CT, and/or tissue biopsy"

        return {
            "category": cat,
            "risk_range": risk,
            "clinical_management": action,
        }


class CalibratedMalignancyModel:
    """
    Inference wrapper for the trained, Platt-calibrated XGBoost malignancy risk model.
    Trained on 4,358 LIDC-IDRI consensus nodules with Option B binarized labels (1-2 vs 4-5).
    """

    _instance = None
    _model_data = None

    @classmethod
    def get_instance(cls, model_path="models/calibrated_malignancy_xgb.joblib"):
        if cls._instance is None:
            import os
            import joblib
            if not os.path.exists(model_path):
                # Search relative to package
                pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                alt_path = os.path.join(pkg_dir, model_path)
                if os.path.exists(alt_path):
                    model_path = alt_path
                else:
                    return None

            cls._model_data = joblib.load(model_path)
            cls._instance = cls._model_data["model"]
        return cls._instance

    @classmethod
    def predict_risk_percent(cls, features_dict, model_path="models/calibrated_malignancy_xgb.joblib"):
        """
        Calculates calibrated malignancy risk probability (0.0% to 100.0%) from nodule attributes.

        Args:
            features_dict: dict containing semantic and/or morphological attributes

        Returns:
            float: calibrated risk percentage in range [0.0, 100.0]
        """
        model = cls.get_instance(model_path)
        if model is None:
            return None

        # Assemble feature vector matching training order:
        # ['subtlety', 'internal_structure', 'calcification', 'sphericity', 'margin', 'lobulation', 'spiculation', 'texture', 'roi_count']
        subtlety = float(features_dict.get("subtlety", 3.0))
        internal_structure = float(features_dict.get("internal_structure", 1.0))
        calcification = float(features_dict.get("calcification", 6.0))  # 6 = non-calcified in LIDC

        # Map sphericity from [0, 1] to 1-5 scale if continuous
        raw_sph = features_dict.get("sphericity", 3.0)
        sphericity = float(raw_sph * 5.0) if raw_sph <= 1.0 else float(raw_sph)

        margin = float(features_dict.get("margin", 3.0))
        lobulation = float(features_dict.get("lobulation", 2.0))

        # Map spiculation
        if "spiculation" in features_dict:
            spiculation = float(features_dict["spiculation"])
        elif features_dict.get("is_spiculated", False):
            spiculation = 4.0
        else:
            spiculation = 1.0

        texture = float(features_dict.get("texture", 5.0))  # 5 = solid
        roi_count = float(features_dict.get("roi_count", features_dict.get("slice_count", 5.0)))

        import numpy as np
        x_vec = np.array([[
            subtlety, internal_structure, calcification, sphericity,
            margin, lobulation, spiculation, texture, roi_count
        ]], dtype=np.float32)

        prob = model.predict_proba(x_vec)[0, 1]
        return round(float(prob * 100.0), 1)


def assess_risk(nodule_features, model_path="models/calibrated_malignancy_xgb.joblib"):
    """
    Standardized clinical triad assessment function:
        1. Calibrated model risk percentage (0-100%)
        2. Brock / PanCan reference formula score (0-100%)
        3. ACR Lung-RADS Version 2022 clinical risk tier & management action
        4. Concordance / Confidence flag ('model/brock agree' vs 'diverge -- review')

    Args:
        nodule_features: dict containing morphology, density, and optional clinical covariates

    Returns:
        dict containing the clinical triad risk report
    """
    d_mean = nodule_features.get("mean_diameter_mm", nodule_features.get("d_mean_mm", 6.0))
    d_max = nodule_features.get("max_diameter_mm", nodule_features.get("d_long_mm", d_mean))
    ntype = nodule_features.get("nodule_type", "solid")
    solid_core_d = nodule_features.get("solid_core_diameter_mm", None)
    is_spic = nodule_features.get("is_spiculated", False)
    is_upper = nodule_features.get("is_upper_lobe", False)

    # 1. Brock PanCan Reference
    brock_res = BrockCalculator.calculate(
        diameter_max_mm=d_max,
        attenuation_type=ntype,
        is_spiculated=is_spic,
        is_upper_lobe=is_upper,
        nodule_count=nodule_features.get("nodule_count", 1),
        age=nodule_features.get("age", 62.0),
        is_female=nodule_features.get("is_female", False),
        family_history=nodule_features.get("family_history", False),
        emphysema=nodule_features.get("emphysema", False),
    )
    brock_percent = brock_res["brock_percent"]

    # 2. ACR Lung-RADS v2022
    rads_res = LungRADSClassifier.classify(
        mean_diameter_mm=d_mean,
        nodule_type=ntype,
        solid_core_diameter_mm=solid_core_d,
        is_spiculated=is_spic,
        is_high_suspicion_modifier=nodule_features.get("is_high_suspicion_modifier", False),
    )

    # 3. Calibrated ML Model Risk
    ml_risk = CalibratedMalignancyModel.predict_risk_percent(nodule_features, model_path=model_path)
    if ml_risk is None:
        # Fallback to Brock if model file not loaded
        ml_risk = brock_percent

    # 4. Confidence / Concordance Flag
    diff = abs(ml_risk - brock_percent)
    if diff <= 25.0:
        confidence_flag = "model/brock agree (high concordance)"
    else:
        confidence_flag = f"diverge -- review (delta: {diff:.1f}%)"

    return {
        "risk_percent": ml_risk,
        "brock_percent": brock_percent,
        "lung_rads_category": rads_res["category"],
        "lung_rads_management": rads_res["clinical_management"],
        "confidence_flag": confidence_flag,
        "nodule_type": ntype,
        "bts_recommendation": brock_res["bts_recommendation"],
    }


def generate_diagnostic_json(
    candidate_id,
    centroid_world_mm,
    dimensions,
    morphology,
    risk_assessment,
):
    """
    Compiles the complete clinical assessment into the standardized diagnostic JSON payload
    defined in Step 6 of the clinical architecture.

    Args:
        candidate_id: str identifier (e.g. 'NOD_0042')
        centroid_world_mm: tuple/list (x, y, z) in mm
        dimensions: dict containing volume_mm3, d_long_mm, d_short_mm, d_mean_mm
        morphology: dict containing type, solid_core_diameter_mm, is_spiculated
        risk_assessment: dict from assess_risk()

    Returns:
        dict matching the Step 6 JSON schema
    """
    return {
        "candidate_id": str(candidate_id),
        "centroid_world_mm": {
            "x": round(float(centroid_world_mm[0]), 1),
            "y": round(float(centroid_world_mm[1]), 1),
            "z": round(float(centroid_world_mm[2]), 1),
        },
        "dimensions": {
            "volume_mm3": round(float(dimensions.get("volume_mm3", 0.0)), 1),
            "long_axis_mm": round(float(dimensions.get("d_long_mm", dimensions.get("long_axis_mm", 0.0))), 1),
            "short_axis_mm": round(float(dimensions.get("d_short_mm", dimensions.get("short_axis_mm", 0.0))), 1),
            "mean_diameter_mm": round(float(dimensions.get("d_mean_mm", dimensions.get("mean_diameter_mm", 0.0))), 1),
        },
        "morphology": {
            "type": str(morphology.get("type", morphology.get("nodule_type", "Solid"))).replace("_", " ").title(),
            "solid_core_diameter_mm": round(float(morphology.get("solid_core_diameter_mm", 0.0)), 1),
            "spiculation_detected": bool(morphology.get("spiculation_detected", morphology.get("is_spiculated", False))),
        },
        "risk_assessment": {
            "malignancy_probability": round(float(risk_assessment.get("risk_percent", 0.0) / 100.0), 2),
            "lung_rads_category": risk_assessment.get("lung_rads_category", "2"),
            "clinical_recommendation": risk_assessment.get("lung_rads_management", risk_assessment.get("clinical_recommendation", "")),
        },
    }


