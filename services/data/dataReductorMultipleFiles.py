"""
This is a simple wrapper that uses the slightly modified instance of
SSDDR dataClass - in order to perform proper data reduction
"""

from .dataClass import dataContainter

class MultipleDataReductor:
    def __init__(
            self,
            archiveFilenames: list[str],
            data_tmp_directory: str,
            annotator_model,
            broken_scans_detector_model,
            software_path: str = ".",
            isOnOff: bool = False,
            isCal: bool = True,
            BBCLHC: int = 1,
            BBCRHC: int = 2):
        # -- first we need to create attributes for data reduction --
        self.archiveFilenames = archiveFilenames
        self.softwarePath = software_path
        self.isOnOff = isOnOff
        self.isCal = isCal
        self.bbcLHC = BBCLHC
        self.bbcRHC = BBCRHC
        self.annotator_model = annotator_model
        self.broken_scans_detector = broken_scans_detector_model

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
                observation.addToStack(
                    i,
                    annotator = self.annotator_model,
                    broken_scan_detector = self.broken_scans_detector)
            # handle calibration
            observation.calculateSpectrumFromStack()
            if self.isCal:
                observation.calibrate(lhc = True)
            observation.clearStack(pol = "LHC")
            observation.bbcs_used.append(self.bbcLHC)

            # -- RHC --
            observation.actualBBC = self.bbcRHC
            for i in range(len(observation.obs.mergedScans)):
                observation.addToStack(
                    i,
                    annotator = self.annotator_model,
                    broken_scan_detector = self.broken_scans_detector)
            # handle calibration
            observation.calculateSpectrumFromStack()
            if self.isCal:
                observation.calibrate(lhc = False)
            observation.clearStack(pol = "RHC")
            observation.bbcs_used.append(self.bbcRHC)

            saved_filenames.append(observation.saveReducedDataToFits())
        return saved_filenames