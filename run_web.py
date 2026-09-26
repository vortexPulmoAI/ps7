"""
Launch script for the Pulmonary Nodule Risk Assessment AI Web Workstation.
Runs FastAPI with Uvicorn on http://127.0.0.1:8000.
"""

import sys
import os
import uvicorn

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    print("=" * 80)
    print("[*] STARTING VORTEX PULMO-AI CLINICAL WEB WORKSTATION")
    print("=" * 80)
    print("  • URL: http://127.0.0.1:8000")
    print("  • Documentation: http://127.0.0.1:8000/docs")
    print("  • Accepted Formats: DICOM (.dcm, .zip), NIfTI (.nii, .nii.gz), MetaImage (.mhd), PNG, JPG, RAW")
    print("  • Press Ctrl+C to terminate.")
    print("=" * 80 + "\n")

    uvicorn.run("web.app:app", host="127.0.0.1", port=8000, reload=False)
