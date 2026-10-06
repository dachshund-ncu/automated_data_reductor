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

def _get_category_from_proba(categories_proba: np.ndarray):
    category = np.asarray([int(np.argmax(s)) for s in categories_proba])
    return category

def test_data_loading():
    print(data_archives)

def plot_single_scan_categories(
        scan_raw_data: np.ndarray,
        scan_channel_probabilities: np.ndarray,
        scan_number: int) -> None:
    # -- misc: add a save directory --
    plot_directory = os.path.join(_de_cat, "plots")
    os.makedirs(plot_directory, exist_ok=True)

    # -- prepare scan categories --
    scan_channel_categories = _get_category_from_proba(scan_channel_probabilities)

    # -- plot data --
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    for category in range(3):
        # ---- predicted labels ----
        tmp_data = scan_raw_data.copy()
        indices = scan_channel_categories == category
        tmp_data[~indices] = np.nan
        ax.plot(list(range(len(tmp_data))), tmp_data, c=colors[category], label=categories[category])

    # -- apperance settings -- 
    ax.set_title(f"Scan no. {scan_number}")
    ax.set_xlabel("Channel no")
    ax.set_ylabel("Amplitude")
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(plot_directory, f"scan_{scan_number}.png"), dpi=100)
    plt.close(fig)

def test_single_scan_annotation():
    for archive in data_archives:
        # -- load data --
        scan_set = ScanSet(
            archive_filename=archive,
            on_off=False,
            use_optimized_methods=True
        )

        # -- load all scans --
        scan_data = []
        for scan in scan_set.mergedScans:
            for bbc_index in range(4):
                scan_data.append(scan.pols[bbc_index])
        scan_data = np.asarray(scan_data)

        # -- load model --
        segmentation_model = UNet1D.from_file(single_scan_annotator_filename)
        predictions = segmentation_model.predict(scan_data)
        for scan_index in range(len(scan_data)):
            single_scan_raw_data = scan_data[scan_index]
            signle_scan_predictions = predictions[scan_index]
            plot_single_scan_categories(
                scan_raw_data = single_scan_raw_data,
                scan_channel_probabilities=signle_scan_predictions,
                scan_number=scan_index
            )

if __name__ == "__main__":
    test_data_loading()
    test_single_scan_annotation()