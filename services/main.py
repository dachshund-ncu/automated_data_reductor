import os
import streamlit as st
from data.dataReductorMultipleFiles import MultipleDataReductor
from datetime import datetime
import glob
import tensorflow as tf
from tensorflow import keras

DE_CAT = os.path.dirname(os.path.abspath(__file__))


def weighted_categorical_crossentropy(weights):
    """
    TLDR: this function definition is required for proper loading of tensorflow models
    Creates a weighted categorical crossentropy loss function.
    Args:
        weights (dict or list): A list where indices correspond to class labels and values are weights.
    Returns:
        A loss function to be used in model compilation.
    """

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)  # Prevent log(0)
        y_true = tf.cast(y_true, tf.float32)

        # Compute per-class weights
        weights_per_sample = tf.reduce_sum(y_true * weights, axis=-1)

        # Compute weighted loss
        loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1) * weights_per_sample

        return tf.reduce_mean(loss)

    return loss


def load_models():
    # get saved model paths using the technique
    filename_scan_annotator = glob.glob(os.path.join(DE_CAT, "models", "*_scan_annotator.keras"))[-1]
    filename_broken_scans_detector = glob.glob(os.path.join(DE_CAT, "models", "*_broken_scans.keras"))[-1]

    # load models using KERAS
    scan_annotator_model = keras.models.load_model(
        filename_scan_annotator,
        custom_objects = {'loss': weighted_categorical_crossentropy})
    broken_scans_detector_model = keras.models.load_model(
        filename_broken_scans_detector)
    return scan_annotator_model, broken_scans_detector_model

def generate_timestamp_dirname():
    """
    Generates a directory name based on the current timestamp, including nanoseconds.
    """
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S_%f")
    return f"{timestamp_str}_data"

def displayMessageOnLoad(uploaded_files, use_caltab, is_onoff):
    if uploaded_files is not None:
        st.write(f"You have uploaded {len(uploaded_files)} files")
    if use_caltab:
        st.write("Caltabs will be used")
    else:
        st.write("Caltabs will not be used")
    if is_onoff:
        st.write(f"Reduction using on-off technique")
    else:
        st.write(f"Reduction using frequency-switch technique")

def processUploadedFiles(
        uploadedFiles: list,
        isOnOff: bool,
        isCal: bool,
        BBCLHC: int,
        BBCRHC: int,
        annotator_model,
        broken_scan_model):
    # -- prepare data --
    tmp_reduction_dir = os.path.join(DE_CAT, "temporary_data", generate_timestamp_dirname())
    os.makedirs(tmp_reduction_dir, exist_ok = True)
    # list with files that were managed to
    data_reduction_files = []
    for uploadedFile in uploadedFiles:
        if uploadedFile is not None:
            fileSavePath = os.path.join(tmp_reduction_dir, uploadedFile.name)
            try:
                fileContent = uploadedFile.getvalue()
                with open(fileSavePath, "wb") as f:
                    f.write(fileContent)
                data_reduction_files.append(fileSavePath)
            except:
                pass

    # -- perform data reduction --
    with st.spinner("Processing uploaded files..."):
        reductor = MultipleDataReductor(
            archiveFilenames = [f for f in data_reduction_files],
            data_tmp_directory = tmp_reduction_dir,
            software_path = DE_CAT,
            isOnOff = isOnOff,
            isCal = isCal,
            BBCLHC = BBCLHC,
            BBCRHC = BBCRHC,
            annotator_model = annotator_model,
            broken_scans_detector_model = broken_scan_model)
        file_names_to_download = reductor.performDataReduction()

        # -- manage files in temporary directory --
        cwd = os.getcwd() # get the current working directory
        os.chdir(tmp_reduction_dir) # change to data save directory
        archive_filename = os.path.basename(tmp_reduction_dir)
        os.system(f"tar -cvjf {archive_filename}.tar.bz2 *.fits") # compress.fits files
        # remove leftover files

        for filename in file_names_to_download:
            os.remove(os.path.join(DE_CAT, filename))
        for filename in data_reduction_files:
            os.remove(filename)
        os.chdir(cwd)

    # set the file to download
    with open(os.path.join(tmp_reduction_dir, f"{archive_filename}.tar.bz2"), "rb") as f:
        st.download_button(
            label = "Download .fits files",
            data = f,
            file_name = f"{archive_filename}.tar.bz2",
            mime = None,
            icon = ":material/download:"
        )
    os.remove(os.path.join(tmp_reduction_dir, f"{archive_filename}.tar.bz2"))
    os.rmdir(tmp_reduction_dir)

def archive_uploader(
        annotator_model,
        broken_scan_model):
    with st.form("Form"):
        uploaded_files = st.file_uploader(
            "Upload .tar.bz2 archives",
            accept_multiple_files=True,
            type = ['.tar.bz2'])

        selection = { f"BBC {i}": i for i in range(1,5)}
        selected_bbc_lhc = st.selectbox(
            "Select a Base Band Converter for LHC",
            selection.keys(),
            index = 0
        )
        selected_bbc_rhc = st.selectbox(
            "Select a Base Band Converter for RHC",
            selection.keys(),
            index = 1
        )

        use_caltab = st.checkbox("Use caltabs", value = True)
        is_onoff = st.checkbox("On-off reduction", value = False)

        submit = st.form_submit_button("Submit")


    if submit:
        st.write(f"You selected BBC {selection[selected_bbc_lhc]} for LHC and BBC {selection[selected_bbc_rhc]} for RHC")
        # -- display message --
        displayMessageOnLoad(uploaded_files, use_caltab, is_onoff)
        # -- process files --
        processUploadedFiles(
            uploaded_files,
            isOnOff = is_onoff,
            isCal = use_caltab,
            BBCLHC = int(selection[selected_bbc_lhc]),
            BBCRHC = int(selection[selected_bbc_rhc]),
            annotator_model = annotator_model,
            broken_scan_model = broken_scan_model)

def main():
    scan_annotator_model, broken_scans_detector_model = load_models()
    st.set_page_config(page_title="Torun 32 m radio telescope data reductor", layout='wide')
    archive_uploader(
        annotator_model = scan_annotator_model,
        broken_scan_model = broken_scans_detector_model
    )

if __name__ == '__main__':
    main()