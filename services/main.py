
"""
Web interface for the automatic SSDDR data reduction pipeline.

This module provides a Streamlit application for processing raw
observational data from the Torun 32-m radio telescope.

The application allows users to:
    - Upload multiple TAR.BZ2 observation archives.
    - Select BBC channels for LHC and RHC polarizations.
    - Enable or disable calibration.
    - Select the observation reduction mode.
    - Perform automatic scan segmentation and data reduction.
    - Download reduced FITS spectra in a compressed archive.

The application can be launched directly through Streamlit
or through its configured CLI entry point.

Temporary observation data and generated FITS files are stored
in an automatically managed temporary directory.
"""

import io
import os
import sys
import tarfile
import tempfile

from datetime import datetime

import streamlit as st

from streamlit.web import cli as stcli

from services.templates.scan_annotation_template import UNet1D
from services.data.dataReductorMultipleFiles import MultipleDataReductor
from services.models import (
    last_scan_annotator_filename,
    single_scan_annotator_filename,
)


# ------------------------------------------------------------
# Application configuration
# ------------------------------------------------------------

# Absolute path to the directory containing this module.
APPLICATION_DIR = os.path.dirname(os.path.abspath(__file__))

# Supported Base Band Converter identifiers.
AVAILABLE_BBC = {
    f"BBC {i}": i
    for i in range(1, 5)
}


# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------

@st.cache_resource
def load_models() -> tuple[UNet1D, UNet1D]:
    """
    Load neural network models used by the reduction pipeline.

    The function is cached using Streamlit's resource cache.
    Models are loaded only once and reused across subsequent
    application reruns.

    Returns:
        tuple[UNet1D, UNet1D]:
            A tuple containing:
                - Single-scan segmentation model.
                - Final-spectrum segmentation model.
    """
    single_scan_model = UNet1D.from_file(
        single_scan_annotator_filename
    )

    final_scan_model = UNet1D.from_file(
        last_scan_annotator_filename
    )

    return single_scan_model, final_scan_model


# ------------------------------------------------------------
# Temporary file management
# ------------------------------------------------------------

def generate_timestamp_dirname() -> str:
    """
    Generate a unique, timestamp-based directory name.

    Microsecond precision reduces the probability of naming
    collisions between consecutive reduction sessions.

    Returns:
        str: Directory name in the format
            YYYYMMDD_HHMMSS_microseconds_data.
    """
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    return f"{timestamp}_data"


def save_uploaded_files(
    uploaded_files: list,
    destination_directory: str,
) -> list[str]:
    """
    Save uploaded observation archives to a local directory.

    Files are written into the temporary reduction directory
    before being passed to the data reduction pipeline.

    Args:
        uploaded_files (list):
            Streamlit UploadedFile objects.

        destination_directory (str):
            Directory where archives should be saved.

    Returns:
        list[str]:
            Absolute paths of successfully saved archives.
    """
    saved_files = []

    for uploaded_file in uploaded_files:
        if uploaded_file is None:
            continue

        filename = os.path.basename(uploaded_file.name)

        # Prevent overwriting an existing file with the same name.
        destination_path = os.path.join(
            destination_directory,
            filename,
        )

        try:
            with open(destination_path, "wb") as file:
                file.write(uploaded_file.getvalue())

            saved_files.append(destination_path)

        except OSError as exc:
            st.warning(
                f"Failed to save {filename}: {exc}"
            )

    return saved_files


# ------------------------------------------------------------
# Output archive management
# ------------------------------------------------------------

def create_reduced_data_archive(
    fits_files: list[str],
) -> bytes:
    """
    Compress reduced FITS files into a TAR.BZ2 archive.

    The archive is created in memory using BytesIO.
    Only files explicitly returned by the reduction pipeline
    are included.

    Args:
        fits_files (list[str]):
            Paths to successfully generated FITS files.

    Returns:
        bytes:
            Binary content of the compressed TAR.BZ2 archive.
    """
    archive_buffer = io.BytesIO()

    with tarfile.open(
        fileobj=archive_buffer,
        mode="w:bz2",
    ) as archive:

        for fits_file in fits_files:
            archive.add(
                fits_file,
                arcname=os.path.basename(fits_file),
            )

    return archive_buffer.getvalue()


# ------------------------------------------------------------
# Data reduction
# ------------------------------------------------------------

def processUploadedFiles(
    uploadedFiles: list,
    isOnOff: bool,
    isCal: bool,
    BBCLHC: int,
    BBCRHC: int,
    annotator_model: UNet1D,
    final_scan_annotator_model: UNet1D,
) -> None:
    """
    Process uploaded observational archives.

    The reduction procedure consists of:
        1. Creating a temporary working directory.
        2. Saving uploaded observation archives.
        3. Initializing the MultipleDataReductor.
        4. Executing automatic data reduction.
        5. Collecting successfully generated FITS files.
        6. Creating a compressed archive.
        7. Displaying a download button.

    All temporary data is automatically removed when
    the function exits.

    Args:
        uploadedFiles (list):
            Uploaded TAR.BZ2 observation archives.

        isOnOff (bool):
            True for ON-OFF observations.
            False for frequency-switching observations.

        isCal (bool):
            Enable calibration using calibration tables.

        BBCLHC (int):
            BBC identifier for left-hand circular polarization.

        BBCRHC (int):
            BBC identifier for right-hand circular polarization.

        annotator_model (UNet1D):
            Neural network for individual scan segmentation.

        final_scan_annotator_model (UNet1D):
            Neural network for final spectrum segmentation.

    Returns:
        None
    """
    if not uploadedFiles:
        st.warning("No observation archives were uploaded.")
        return

    # Create an isolated temporary workspace.
    with tempfile.TemporaryDirectory() as tmp_dir:

        reduction_dir = os.path.join(
            tmp_dir,
            generate_timestamp_dirname(),
        )

        os.makedirs(
            reduction_dir,
            exist_ok=True,
        )

        # Save uploaded archives to the temporary workspace.
        data_reduction_files = save_uploaded_files(
            uploaded_files=uploadedFiles,
            destination_directory=reduction_dir,
        )

        if not data_reduction_files:
            st.error("No valid archives could be saved.")
            return

        # Initialize and execute the reduction pipeline.
        with st.spinner("Processing uploaded files..."):

            reductor = MultipleDataReductor(
                archiveFilenames=data_reduction_files,
                data_tmp_directory=reduction_dir,
                software_path=APPLICATION_DIR,
                isOnOff=isOnOff,
                isCal=isCal,
                BBCLHC=BBCLHC,
                BBCRHC=BBCRHC,
                annotator_model=annotator_model,
                final_scan_annotator_model=(
                    final_scan_annotator_model
                ),
            )

            fits_files = reductor.performDataReduction()

        # Do not create an empty archive.
        if not fits_files:
            st.warning(
                "Data reduction finished, but no valid "
                "FITS files were generated."
            )
            return

        st.success(
            f"Successfully generated {len(fits_files)} "
            f"FITS file(s)."
        )

        # Compress successfully reduced spectra.
        archive_data = create_reduced_data_archive(
            fits_files=fits_files,
        )

        archive_filename = (
            f"{os.path.basename(reduction_dir)}.tar.bz2"
        )

        # Streamlit receives archive bytes, so the temporary
        # workspace can be safely removed after this function.
        st.download_button(
            label="Download .fits files",
            data=archive_data,
            file_name=archive_filename,
            mime="application/x-bzip2",
            icon=":material/download:",
        )


# ------------------------------------------------------------
# Streamlit user interface
# ------------------------------------------------------------

def archive_uploader(
    annotator_model: UNet1D,
    final_scan_annotator_model: UNet1D,
) -> None:
    """
    Render the observation archive uploader interface.

    The form allows users to configure BBC selection,
    calibration and observation mode before starting
    the reduction pipeline.

    Args:
        annotator_model (UNet1D):
            Neural network for single-scan segmentation.

        final_scan_annotator_model (UNet1D):
            Neural network for final-spectrum segmentation.

    Returns:
        None
    """
    with st.form("reduction_form"):

        uploaded_files = st.file_uploader(
            "Upload .tar.bz2 archives",
            accept_multiple_files=True,
            type=["bz2"],
        )

        selected_bbc_lhc = st.selectbox(
            "Base Band Converter for LHC",
            options=list(AVAILABLE_BBC.keys()),
            index=0,
        )

        selected_bbc_rhc = st.selectbox(
            "Base Band Converter for RHC",
            options=list(AVAILABLE_BBC.keys()),
            index=1,
        )

        use_caltab = st.checkbox(
            "Use caltabs",
            value=True,
        )

        is_onoff = st.checkbox(
            "On-off reduction",
            value=False,
        )

        submit = st.form_submit_button(
            "Submit",
        )

    if not submit:
        return

    # Resolve selected BBC labels to their numeric identifiers.
    bbc_lhc = AVAILABLE_BBC[selected_bbc_lhc]
    bbc_rhc = AVAILABLE_BBC[selected_bbc_rhc]

    st.write(
        f"You selected BBC {bbc_lhc} for LHC "
        f"and BBC {bbc_rhc} for RHC."
    )

    if uploaded_files:
        st.write(
            f"You have uploaded {len(uploaded_files)} files."
        )

    st.write(
        "Caltabs will be used"
        if use_caltab
        else "Caltabs will not be used"
    )

    st.write(
        "Reduction using on-off technique"
        if is_onoff
        else "Reduction using frequency-switch technique"
    )

    # Execute the reduction pipeline using the form parameters.
    processUploadedFiles(
        uploadedFiles=uploaded_files,
        isOnOff=is_onoff,
        isCal=use_caltab,
        BBCLHC=bbc_lhc,
        BBCRHC=bbc_rhc,
        annotator_model=annotator_model,
        final_scan_annotator_model=final_scan_annotator_model,
    )


# ------------------------------------------------------------
# Application entry points
# ------------------------------------------------------------

def main() -> None:
    """
    Initialize and run the Streamlit application.

    Configures the application layout, loads cached neural
    network models and renders the archive uploader.
    """
    st.set_page_config(
        page_title="Torun 32 m radio telescope data reductor",
        layout="wide",
    )

    scan_annotator, final_scan_annotator = load_models()

    archive_uploader(
        annotator_model=scan_annotator,
        final_scan_annotator_model=final_scan_annotator,
    )


def cli() -> None:
    """
    Launch the Streamlit application through the CLI.

    Intended as an entry point for installed Python packages,
    including packages managed with uv.

    Additional command-line arguments are forwarded
    to Streamlit.
    """
    main_path = os.path.abspath(__file__)

    sys.argv = [
        "streamlit",
        "run",
        main_path,
        *sys.argv[1:],
    ]

    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
