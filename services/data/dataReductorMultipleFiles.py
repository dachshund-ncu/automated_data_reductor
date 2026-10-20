
import logging
from pathlib import Path

import numpy as np
import streamlit as st

from .dataClass import dataContainter


logger = logging.getLogger(__name__)


class MultipleDataReductor:

    def __init__(
        self,
        archiveFilenames: list[str],
        data_tmp_directory: str,
        annotator_model,
        final_scan_annotator_model,
        broken_scan_detection_model=None,
        software_path: str = ".",
        isOnOff: bool = False,
        isCal: bool = True,
        BBCLHC: int = 1,
        BBCRHC: int = 2,
        min_continuum_fraction: float = 0.5,
        max_tsys: float = 70.0,
    ):
        self.archiveFilenames = archiveFilenames
        self.dataTmpDirectory = data_tmp_directory
        self.softwarePath = software_path

        self.isOnOff = isOnOff
        self.isCal = isCal

        self.bbcLHC = BBCLHC
        self.bbcRHC = BBCRHC

        self.annotator_model = annotator_model
        self.final_scan_annotator_model = (
            final_scan_annotator_model
        )
        self.broken_scan_detection_model = (
            broken_scan_detection_model
        )

        self.min_continuum_fraction = min_continuum_fraction
        self.max_tsys = max_tsys

        if not 0 <= min_continuum_fraction <= 1:
            raise ValueError(
                "min_continuum_fraction must be in [0, 1]"
            )

        # Download calibration tables once.
        if self.isCal:
            dummy = dataContainter(
                software_path=self.softwarePath,
                target_filename=None,
            )
            dummy.download_caltabs()
            del dummy

    @staticmethod
    def _get_scan_data(
        observation,
        bbc: int,
        indices: np.ndarray,
    ) -> np.ndarray:
        """
        Collect spectra into one batch: (N, 4096).
        """
        return np.stack(
            [
                observation.obs.mergedScans[int(i)].pols[bbc - 1]
                for i in indices
            ],
            axis=0,
        )

    def _get_tsys_mask(
        self,
        observation,
        bbc: int,
    ) -> np.ndarray:
        """
        Reject a merged scan when either source scan
        has TSYS > max_tsys.

        Assumes two consecutive source scans per merged scan.
        """
        n_merged = len(observation.obs.mergedScans)
        tsys = np.asarray(
            observation.tsysTab[bbc - 1],
            dtype=float,
        )

        if len(tsys) != 2 * n_merged:
            raise ValueError(
                "Cannot map source TSYS to merged scans: "
                f"{len(tsys)} source values, "
                f"{n_merged} merged scans."
            )

        # Values are already in K (dataClass divides by 1000).
        tsys_pairs = tsys.reshape(n_merged, 2)

        return np.all(
            np.isfinite(tsys_pairs)
            & (tsys_pairs <= self.max_tsys),
            axis=1,
        )

    def _get_broken_mask(
        self,
        observation,
        bbc: int,
        indices: np.ndarray,
        spectra: np.ndarray,
    ) -> np.ndarray:
        """
        True means broken scan.

        Combines IsolationForest and neural network.
        """
        outliers = np.asarray(
            observation.outlierTable[bbc - 1]
        )[indices] == -1

        if self.broken_scan_detection_model is None:
            return outliers

        # One inference for the entire batch.
        predictions = self.broken_scan_detection_model.predict(
            spectra[..., np.newaxis],
        )

        # Original model convention:
        # class 0 = good, other classes = broken.
        nn_broken = np.argmax(predictions, axis=-1) != 0

        return outliers | nn_broken

    def _segment_batch(
        self,
        spectra: np.ndarray,
    ) -> np.ndarray:
        """
        Segment all candidate spectra in one model call.

        Input:  (N, 4096)
        Output: (N, 4096), integer channel labels.
        """
        predictions = self.annotator_model.predict(
            spectra,
        )

        if predictions.ndim != 3:
            raise ValueError(
                "Expected segmentation output (N, channels, classes), "
                f"got {predictions.shape}"
            )

        return np.argmax(
            predictions,
            axis=-1,
        ).astype(np.uint8)

    def _process_polarization(
        self,
        observation,
        bbc: int,
        pol: str,
        progress_callback=None,
    ) -> int:
        """
        Process one polarization.

        Returns number of successfully stacked scans.
        """
        observation.actualBBC = bbc

        n_scans = len(observation.obs.mergedScans)

        if n_scans == 0:
            return 0

        # --------------------------------------------------
        # 1. Filter TSYS before neural network inference.
        # --------------------------------------------------
        tsys_mask = self._get_tsys_mask(
            observation,
            bbc,
        )

        candidate_indices = np.flatnonzero(tsys_mask)

        for i in np.flatnonzero(~tsys_mask):
            observation.discardFromStack(int(i))

        if progress_callback is not None:
            progress_callback(0.15, f"{pol}: TSYS filtering")

        if len(candidate_indices) == 0:
            logger.warning("%s: no scans pass TSYS filter", pol)
            return 0

        # --------------------------------------------------
        # 2. Prepare the whole candidate batch.
        # --------------------------------------------------
        spectra = self._get_scan_data(
            observation,
            bbc,
            candidate_indices,
        )

        # --------------------------------------------------
        # 3. Batch broken-scan detection.
        # --------------------------------------------------
        broken_mask = self._get_broken_mask(
            observation,
            bbc,
            candidate_indices,
            spectra,
        )

        for i in candidate_indices[broken_mask]:
            observation.discardFromStack(int(i))

        good_mask = ~broken_mask

        candidate_indices = candidate_indices[good_mask]
        spectra = spectra[good_mask]

        if progress_callback is not None:
            progress_callback(0.35, f"{pol}: broken-scan detection")

        if len(candidate_indices) == 0:
            logger.warning(
                "%s: no scans pass broken-scan detection",
                pol,
            )
            return 0

        # --------------------------------------------------
        # 4. One batch segmentation.
        # --------------------------------------------------
        categories = self._segment_batch(spectra)

        if progress_callback is not None:
            progress_callback(0.65, f"{pol}: segmentation")

        # --------------------------------------------------
        # 5. Continuum failsafe.
        # --------------------------------------------------
        continuum_fractions = np.mean(
            categories == 0,
            axis=1,
        )

        continuum_mask = (
            continuum_fractions >= self.min_continuum_fraction
        )

        for i in candidate_indices[~continuum_mask]:
            observation.discardFromStack(int(i))

        candidate_indices = candidate_indices[continuum_mask]
        categories = categories[continuum_mask]

        if len(candidate_indices) == 0:
            logger.warning(
                "%s: no scans pass continuum filter",
                pol,
            )
            return 0

        # --------------------------------------------------
        # 6. RFI removal, polynomial fit, stacking.
        # --------------------------------------------------
        n_added = 0
        n_candidates = len(candidate_indices)

        for j, (scan_index, labels) in enumerate(
            zip(candidate_indices, categories)
        ):
            added = observation.addToStackPresegmented(
                scanIndex=int(scan_index),
                channel_categories=labels,
            )

            if added:
                n_added += 1

            if progress_callback is not None:
                progress_callback(
                    0.65 + 0.30 * (j + 1) / n_candidates,
                    f"{pol}: stacking {j + 1}/{n_candidates}",
                )

        logger.info(
            "%s: accepted %d/%d scans",
            pol,
            n_added,
            n_scans,
        )

        if n_added == 0:
            return 0

        # --------------------------------------------------
        # 7. Final spectrum.
        # --------------------------------------------------
        observation.calculateSpectrumFromStack()

        observation.processFinalSpectrum(
            observation.finalFitRes,
            self.final_scan_annotator_model,
        )

        if self.isCal:
            observation.calibrate(lhc=(pol == "LHC"))

        observation.clearStack(pol=pol)
        observation.bbcs_used.append(bbc)

        if progress_callback is not None:
            progress_callback(1.0, f"{pol}: complete")

        return n_added

    def performDataReduction(self) -> list[str]:
        saved_filenames: list[str] = []
        n_files = len(self.archiveFilenames)

        if n_files == 0:
            return saved_filenames

        progress_bar = st.progress(
            0.0,
            text="Starting data reduction...",
        )

        for file_index, filename in enumerate(
            self.archiveFilenames
        ):
            observation = None

            def update_progress(
                fraction: float,
                description: str,
            ):
                # Fraction of the current file completed.
                overall = (
                    file_index + np.clip(fraction, 0.0, 1.0)
                ) / n_files

                progress_bar.progress(
                    float(overall),
                    text=(
                        f"File {file_index + 1}/{n_files} "
                        f"({Path(filename).name}) — "
                        f"{description}"
                    ),
                )

            update_progress(0.0, "loading")

            try:
                observation = dataContainter(
                    software_path=self.softwarePath,
                    target_filename=filename,
                    data_tmp_directory=self.dataTmpDirectory,
                    onOff=self.isOnOff,
                )

                if self.isCal:
                    observation.findCalCoefficients()

                # LHC occupies 0–50% of file progress.
                n_lhc = self._process_polarization(
                    observation,
                    bbc=self.bbcLHC,
                    pol="LHC",
                    progress_callback=lambda f, msg: update_progress(
                        0.5 * f, msg
                    ),
                )

                if n_lhc == 0:
                    logger.warning(
                        "Skipping %s: no valid LHC scans",
                        filename,
                    )
                    update_progress(1.0, "skipped: no LHC scans")
                    continue

                # RHC occupies 50–100% of file progress.
                n_rhc = self._process_polarization(
                    observation,
                    bbc=self.bbcRHC,
                    pol="RHC",
                    progress_callback=lambda f, msg: update_progress(
                        0.5 + 0.5 * f, msg
                    ),
                )

                if n_rhc == 0:
                    logger.warning(
                        "Skipping %s: no valid RHC scans",
                        filename,
                    )
                    update_progress(1.0, "skipped: no RHC scans")
                    continue

                # Both polarizations must contain valid data.
                result = observation.saveReducedDataToFits()
                saved_filenames.append(result)

                update_progress(1.0, "saved")

            finally:
                del observation

        progress_bar.progress(
            1.0,
            text=(
                f"Data reduction complete: "
                f"{len(saved_filenames)}/{n_files} files saved"
            ),
        )

        return saved_filenames
