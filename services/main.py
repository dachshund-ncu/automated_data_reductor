import os,sys
from PIL import Image
import streamlit as st
import numpy as np
import pandas as pd

def archive_uploader():
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

def main():
    st.set_page_config(page_title="Torun 32 m radio telescope data reductor", layout='wide')
    archive_uploader()

if __name__ == '__main__':
    main()