import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def load_and_show_ct(file_path="124.raw", width=512, height=512, window="default"):
    """
    Loads 16-bit signed raw CT slice and displays it.
    Window options: 'default', 'soft_tissue', 'bone', 'lung'
    """
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # Read 16-bit signed integer raw data
    image = np.fromfile(file_path, dtype=np.int16)
    image = image.reshape((height, width))

    # Determine windowing (HU thresholds)
    vmin, vmax = None, None
    title = f"{file_path} (Default Min-Max)"
    if window == "soft_tissue":
        vmin, vmax = -160, 240  # WL: 40, WW: 400
        title = f"{file_path} - Soft Tissue Window (WL: 40, WW: 400)"
    elif window == "bone":
        vmin, vmax = -500, 1300  # WL: 400, WW: 1800
        title = f"{file_path} - Bone Window (WL: 400, WW: 1800)"
    elif window == "lung":
        vmin, vmax = -1350, 150  # WL: -600, WW: 1500
        title = f"{file_path} - Lung Window (WL: -600, WW: 1500)"

    # Display the CT slice
    plt.figure(figsize=(7, 7))
    plt.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "124.raw"
    win = sys.argv[2] if len(sys.argv) > 2 else "default"
    load_and_show_ct(target, window=win)
