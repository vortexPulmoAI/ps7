"""
Pulmonary Nodule Risk Assessment AI Core Package.
Provides modules for CT preprocessing, 3D morphology, Hounsfield density profiling,
LUNA16/LIDC dataset parsing, detection & segmentation wrappers, and clinical calculators.
"""

from .preprocessing import AffineTransform, resample_isotropic, apply_hu_window, load_mhd_mask
from .morphology import (
    calculate_volume,
    calculate_diameters_and_axes,
    calculate_sphericity,
    calculate_spiculation_index
)
from .density import decompose_density, classify_nodule_type
from .dataset_parser import (
    LUNA16AnnotationParser,
    LUNA16CandidateParser,
    LIDCXMLParser,
    LUNALungMaskLoader
)
from .detection import (
    NNDetectionDataPrep,
    CandidatePatchExtractor,
    FROCEvaluator
)
from .segmentation import (
    NNUNetDataPrep,
    NoduleSegmentationInferer,
    SegmentationEvaluator
)
from .risk_engine import (
    BrockCalculator,
    LungRADSClassifier,
    CalibratedMalignancyModel,
    assess_risk,
    generate_diagnostic_json
)

__all__ = [
    "AffineTransform",
    "resample_isotropic",
    "apply_hu_window",
    "load_mhd_mask",
    "calculate_volume",
    "calculate_diameters_and_axes",
    "calculate_sphericity",
    "calculate_spiculation_index",
    "decompose_density",
    "classify_nodule_type",
    "LUNA16AnnotationParser",
    "LUNA16CandidateParser",
    "LIDCXMLParser",
    "LUNALungMaskLoader",
    "NNDetectionDataPrep",
    "CandidatePatchExtractor",
    "FROCEvaluator",
    "NNUNetDataPrep",
    "NoduleSegmentationInferer",
    "SegmentationEvaluator",
    "BrockCalculator",
    "LungRADSClassifier",
    "CalibratedMalignancyModel",
    "assess_risk",
    "generate_diagnostic_json",
]
