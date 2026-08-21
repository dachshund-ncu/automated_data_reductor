import os
import sys
import glob
import tarfile
import tempfile
from pathlib import Path
from datetime import datetime

import requests
import numpy as np
import tensorflow as tf
from tensorflow import keras
import streamlit as st
from platformdirs import user_cache_dir
from streamlit.web import cli as stcli

from services.data.dataReductorMultipleFiles import MultipleDataReductor

# Odpowiednie zarządzanie katalogami w zainstalowanej paczce
CACHE_DIR = Path(user_cache_dir("art4r", "dachshund-ncu"))
MODELS_DIR = CACHE_DIR / "models"

# Wymuszenie CPU dla wnioskowania TF
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def weighted_categorical_crossentropy(weights):
    """
    Creates a weighted categorical crossentropy loss function.
    Important for model loading
    """
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        y_true = tf.cast(y_true, tf.float32)
        weights_per_sample = tf.reduce_sum(y_true * weights, axis=-1)
        loss_val = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1) * weights_per_sample
        return tf.reduce_mean(loss_val)

    return loss


def download_file_requests_basic(url: str, local_filename: Path) -> None:
    """
    Downloads a file from a URL using requests.get() if it doesn't exist.
    """
    if local_filename.exists():
        return
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        local_filename.parent.mkdir(parents=True, exist_ok=True)
        with open(local_filename, 'wb') as f:
            f.write(response.content)
    except Exception as e:
        st.error(f"Nie udało się pobrać pliku {local_filename.name}: {e}")


@st.cache_resource
def load_models() -> None:
    """
    Loads Tensorflow models to the cache directory
    Returns: None
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    scan_annotator_address = "https://box.pionier.net.pl/f/2093ab41430447d8a0a2/?dl=1"
    broken_scan_address = "https://box.pionier.net.pl/f/c7a1bb1e492e4197b70e/?dl=1"
    final_scan_annotator_address = "https://box.pionier.net.pl/f/ab881a6e6c90425486d0/?dl=1"

    download_file_requests_basic(scan_annotator_address, MODELS_DIR / "01_single_scan_annotator.keras")
    download_file_requests_basic(broken_scan_address, MODELS_DIR / "01_broken_scans.keras")
    download_file_requests_basic(final_scan_annotator_address, MODELS_DIR / "01_final_scan_annotator.keras")

    filename_scan_annotator = sorted(MODELS_DIR.glob("*single_scan_annotator.keras"))[-1]
    filename_broken_scans_detector = sorted(MODELS_DIR.glob("*_broken_scans.keras"))[-1]
    filename_final_scan_annotator = sorted(MODELS_DIR.glob("*_final_scan_annotator.keras"))[-1]

    scan_annotator_model = keras.models.load_model(
        filename_scan_annotator,
        custom_objects={'loss': weighted_categorical_crossentropy}
    )
    broken_scans_detector_model = keras.models.load_model(
        filename_broken_scans_detector
    )
    final_scan_annotator_model = keras.models.load_model(
        filename_final_scan_annotator,
        custom_objects={'loss': weighted_categorical_crossentropy}
    )
    return scan_annotator_model, broken_scans_detector_model, final_scan_annotator_model


def generate_timestamp_dirname() -> str:
    now = datetime.now()
    return f"{now.strftime('%Y%m%d_%H%M%S_%f')}_data"


def processUploadedFiles(
        uploadedFiles: list,
        isOnOff: bool,
        isCal: bool,
        BBCLHC: int,
        BBCRHC: int,
        annotator_model: tf.keras.models.Model,
        broken_scan_model: tf.keras.models.Model,
        final_scan_annotator_model: tf.keras.models.Model):
    
    # -- perform all actions using temporary files --
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_reduction_dir = Path(tmp_dir) / generate_timestamp_dirname()
        tmp_reduction_dir.mkdir(parents=True, exist_ok=True)

        data_reduction_files = []
        for uploadedFile in uploadedFiles:
            if uploadedFile is not None:
                fileSavePath = tmp_reduction_dir / uploadedFile.name
                try:
                    fileSavePath.write_bytes(uploadedFile.getvalue())
                    data_reduction_files.append(str(fileSavePath))
                except Exception as e:
                    st.warning(f"Błąd zapisu pliku {uploadedFile.name}: {e}")

        with st.spinner("Processing uploaded files..."):
            reductor = MultipleDataReductor(
                archiveFilenames=data_reduction_files,
                data_tmp_directory=str(tmp_reduction_dir),
                software_path=str(tmp_reduction_dir),
                isOnOff=isOnOff,
                isCal=isCal,
                BBCLHC=BBCLHC,
                BBCRHC=BBCRHC,
                annotator_model=annotator_model,
                broken_scans_detector_model=broken_scan_model,
                final_scan_annotator_model=final_scan_annotator_model
            )
            # -- perform data reduction --
            file_names_to_download = reductor.performDataReduction()

            # -- save all data-reducted files into .tar.bz2 archive -
            archive_filename = f"{tmp_reduction_dir.name}.tar.bz2"
            archive_path = tmp_reduction_dir / archive_filename
            fits_files = list(tmp_reduction_dir.glob("*.fits"))
            with tarfile.open(archive_path, "w:bz2") as tar:
                for fits_file in fits_files:
                    tar.add(fits_file, arcname=fits_file.name)

            # -- allow to download data-reducted archive --
            if archive_path.exists():
                st.download_button(
                    label="Download .fits files",
                    data=archive_path.read_bytes(),
                    file_name=archive_filename,
                    mime="application/x-bzip2",
                    icon=":material/download:"
                )


def archive_uploader(
        annotator_model: tf.keras.models.Model,
        broken_scan_model: tf.keras.models.Model,
        final_scan_annotator_model: tf.keras.models.Model) -> None:
    
    with st.form("Form"):
        uploaded_files = st.file_uploader(
            "Upload .tar.bz2 archives",
            accept_multiple_files=True,
            type=['.tar.bz2']
        )

        selection = {f"BBC {i}": i for i in range(1, 5)}
        selected_bbc_lhc = st.selectbox(
            "Base Band Converter for LHC",
            list(selection.keys()),
            index=0
        )
        selected_bbc_rhc = st.selectbox(
            "Base Band Converter for RHC",
            list(selection.keys()),
            index=1
        )

        use_caltab = st.checkbox("Use caltabs", value=True)
        is_onoff = st.checkbox("On-off reduction", value=False)
        submit = st.form_submit_button("Submit")

    if submit:
        st.write(f"You selected BBC {selection[selected_bbc_lhc]} for LHC and BBC {selection[selected_bbc_rhc]} for RHC")
        if uploaded_files:
            st.write(f"You have uploaded {len(uploaded_files)} files")
        st.write("Caltabs will be used" if use_caltab else "Caltabs will not be used")
        st.write("Reduction using on-off technique" if is_onoff else "Reduction using frequency-switch technique")

        processUploadedFiles(
            uploaded_files,
            isOnOff=is_onoff,
            isCal=use_caltab,
            BBCLHC=int(selection[selected_bbc_lhc]),
            BBCRHC=int(selection[selected_bbc_rhc]),
            annotator_model=annotator_model,
            broken_scan_model=broken_scan_model,
            final_scan_annotator_model=final_scan_annotator_model
        )


def main():
    """
    Main function of the reductor
    Used ONLY by streamlit server
    """
    st.set_page_config(page_title="Torun 32 m radio telescope data reductor", layout='wide')
    scan_annotator_model, broken_scans_detector_model, final_scan_annotator = load_models()
    archive_uploader(
        annotator_model=scan_annotator_model,
        broken_scan_model=broken_scans_detector_model,
        final_scan_annotator_model=final_scan_annotator
    )


def cli():
    """
    Entrance point for Command Line Interface (CLI)
    """
    main_path = Path(__file__).resolve()
    sys.argv = ["streamlit", "run", str(main_path)] + sys.argv[1:]
    sys.exit(stcli.main())


if __name__ == '__main__':
    main()