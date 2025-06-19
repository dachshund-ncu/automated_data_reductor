"""
Initializes
"""

from .dataClass import dataContainter

class MultipleDataReductor:
    def __init__(
            self,
            archiveFilenames: list[str],
            data_tmp_directory: str,
            software_path: str = ".",
            isOnOff: bool = False,
            isCal: bool = True,
            BBCLHC: int = 1,
            BBCRHC: int = 2):
        # -- first we need to create objects for data reduction --
        self.archiveFilenames = archiveFilenames
        self.softwarePath = software_path
        self.isOnOff = isOnOff
        self.isCal = isCal
        self.bbcLHC = BBCLHC
        self.bbcRHC = BBCRHC

        # -- download caltabs --
        self.dummyObject = dataContainter(
            self.softwarePath,
            target_filename = None,
        )
        self.dummyObject.download_caltabs()
        del self.dummyObject
        # ----------------------

        self.observationsTab = [dataContainter(
            software_path = software_path,
            target_filename = singleArchiveFilename,
            data_tmp_directory = data_tmp_directory,
            onOff = isOnOff) for singleArchiveFilename in archiveFilenames]
        if self.isCal:
            self.observationsTab[0].download_caltabs()
        self.dataReductedFilenames = []

    def performDataReduction(self):
        saved_filenames: list[str] = []
        for observation in self.observationsTab:
            observation.findCalCoefficients()
            # -- LHC --
            observation.actualBBC = self.bbcLHC
            for i in range(len(observation.obs.mergedScans)):
                observation.addToStack(i)
            # handle calibration
            observation.calculateSpectrumFromStack()
            if self.isCal:
                observation.calibrate(lhc = True)
            observation.clearStack(pol = "LHC")
            observation.bbcs_used.append(self.bbcLHC)

            # -- RHC --
            observation.actualBBC = self.bbcRHC
            for i in range(len(observation.obs.mergedScans)):
                observation.addToStack(i)
            # handle calibration
            observation.calculateSpectrumFromStack()
            if self.isCal:
                observation.calibrate(lhc = False)
            observation.clearStack(pol = "RHC")
            observation.bbcs_used.append(self.bbcRHC)

            saved_filenames.append(observation.saveReducedDataToFits())
        return saved_filenames















# if uploaded_file is not None:
#     # --- 4. Construct the full path for the new file ---
#     original_filename = uploaded_file.name
#     file_save_path = os.path.join(st.session_state.temp_dir_path, original_filename)
#
#     # --- 5. Write the content of the uploaded file to this new path ---
#     try:
#         # Get the file content in bytes
#         file_content = uploaded_file.getvalue()
#
#         # Write bytes to the file
#         with open(file_save_path, "wb") as f:
#             f.write(file_content)
#
#         st.success(f"File '{original_filename}' successfully saved to:\n`{file_save_path}`")
#
#         # Optional: Display content if it's a text file
#         if 'text' in uploaded_file.type or uploaded_file.type.startswith('application/json'):
#             st.subheader("Content Preview:")
#             # Re-read from the saved file to ensure it's written correctly
#             with open(file_save_path, "r", encoding="utf-8", errors="ignore") as f:
#                 st.code(f.read()[:500] + "..." if len(f.read()) > 500 else f.read())
#         elif 'image' in uploaded_file.type:
#             st.subheader("Image Preview:")
#             st.image(file_save_path, caption=original_filename, use_column_width=True)
#
#     except Exception as e:
#         st.error(f"Error saving file: {e}")