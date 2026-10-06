from sample_data import data_archives, cepa_archives
import pytest
from ncu_salsa_rt4 import ScanSet
from services.models import single_scan_annotator_filename
from services.templates import UNet1D
import numpy as np
import os
import matplotlib.pyplot as plt

_de_cat = os.path.dirname(__file__)
plt.style.use("ggplot")

    
    
colors  = {
    0: 'grey',
    1: 'green',
    2: 'red',
    3: 'blue'}
categories = {
    0: 'continuum',
    1: 'emission',
    2: 'rfi',
    3: 'edge'
}

def test_data_loading():
    print(data_archives)

def plot_single_scan_categories(
        raw_data: np.ndarray,
        scan_categories: np.ndarray) -> None:
    plot_directory = os.path.join(_de_cat, "plots")
    os.makedirs(plot_directory, exist_ok=True)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    for category in range(3):
        # ---- manual labels ----
        tmp_data = raw_data.copy()
        indices = scan_categories == category
        tmp_data[~indices] = np.nan
        axes[0].plot(list(range(len(tmp_data))), tmp_data, c=colors[category], label = categories[category])

        # ---- predicted labels ----
        tmp_data = raw_data.copy()
        indices = scan_categories == category
        tmp_data[~indices] = np.nan
        axes[1].plot(list(range(len(tmp_data))), tmp_data, c=colors[category])
    # ---- apperance settings ---- 
    axes[0].legend()
    # axes[0].set_title(f"Scan no. {index}")
    axes[1].set_xlabel("Channel")
    axes[0].set_ylabel("Manual annotations")
    axes[1].set_ylabel("Model predictions")
    fig.tight_layout()


def test_single_scan_annotation():
    for archive in cepa_archives:
        # -- load data --
        scan_set = ScanSet(
            archive_filename=archive,
            on_off=False,
            use_optimized_methods=True
        )

        # -- load all scans --
        scan_data = []
        for scan in scan_set.scans:
            for bbc_index in range(4):
                scan_data.append(scan.spectr_bbc_final[bbc_index])
        scan_data = np.asarray(scan_data)

        # -- load model --
        segmentation_model = UNet1D.from_file(single_scan_annotator_filename)
        predictions = segmentation_model.predict(scan_data)
        print(np.asarray(predictions).shape)

if __name__ == "__main__":
    test_data_loading()
    test_single_scan_annotation()