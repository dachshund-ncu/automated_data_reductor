"""
Class, that holds the whole data
"""

import tarfile
from .scanObservation import observation
from .caltabClass import caltab
import os
import numpy as np
import configparser
from astropy.io import fits
import platformdirs
from sklearn.ensemble import IsolationForest

class dataContainter:
    def __init__(self,
                 software_path: str,
                 data_tmp_directory: str = ".",
                 target_filename: str | None = None,
                 onOff: bool = False):
        self.isOnOff = onOff
        '''
        CHECK CONFIGURATION FILES
        '''
        self.caltabs = []
        self.DE_CAT = software_path
        self.configDir = platformdirs.user_config_dir('ssddr')
        self.__load_caltabs_wrapper()

        '''
        Initializes dataContainer class
        '''
        self.bbcs_used = []
        self.noOfBBC = 4
        self.actualBBC = 1
        self.fitOrder = 10
        self.tmpDirName = '.tmpSimpleDataReductor'
        self.fitBoundsChannels = [ 
            [10, 824],
            [1224, 2872],
            [3272, 4086]
        ]
        self.dataTmpDirectory = data_tmp_directory
        if target_filename is not None:
            self.__openTheArchive(target_filename)
            self.__processData()
            self.zTab = self.__getZData()
            self.tsysTab = self.__getTsysData()
            self.totalFluxTab = self.__getTotalFluxData()
            self.outlierTable = self.__getOutliers(self.totalFluxTab)
            self.timeTab = self.__getTimeData()
            self.mergedTimeTab = self.__getMergedTimeData()
            self.velTab = self.__generateVelTab()
            self.properCaltabIndex = self.findProperCaltabIndex()
            self.scans_proceed = self.__makeScansProceedTable()
            self.loadedData = True
        else:
            self.loadedData = False

        
        # -- FINAL SPECTRUM --
        self.finalFitBoundChannels = [
            [10,900],
            [1100, 2038]
        ]
        self.stack = []
        self.scansInStack = []
        
        self.meanStack = []
        self.finalFitRes = []
        self.finalRHC = []
        self.finalLHC = []
        # --------------------
        # --- calibration ---
        self.calCoeffLHC = 1.0
        self.calCoeffRHC = 1.0

    def download_caltabs(self):
        """
        Downloads caltabs from the server
        """
        self.caltabs_backup = self.caltabs
        self.caltabs = []
        if self.__tryToLoadCaltabs():
            print("-----> Caltabs downloaded")
            self.__copy_caltabs_to_config(os.path.join(self.configDir, 'caltabs'))
            if self.loadedData:
                self.properCaltabIndex = self.findProperCaltabIndex()
            else:
                self.properCaltabIndex = 0
            return True
        else:
            self.caltabs = self.caltabs_backup
            return False
        
    def __load_caltabs_wrapper(self):
        '''
        Loads the caltab files upon startup
        '''
        self.caltabs = []
        caldir = os.path.join(self.configDir, 'caltabs')
        if not os.path.exists(caldir):
            os.makedirs(caldir, exist_ok=True)
        print(f"-----> Searching for caltabs in {os.path.join(self.configDir, 'caltabs')}...")
        self.__read_caltabs_from_config(os.path.join(self.configDir, 'caltabs'))
        if len(self.caltabs) == 0:
            self.caltabsLoaded = False
            print(f"-----> No caltabs found. I suggest to try download them.")
            print(f"--------> Just go to advanced -> download caltabs")
        else:
            self.caltabsLoaded = True
        print("-----------------------------------------")

    def __read_caltabs_from_config(self, directory: str):
        '''
        Simply reads the caltab from config files
        '''
        dirs = []
        for root, dir, file in os.walk(directory):
            for dir_single in dir:
                dirs.append(os.path.join(directory, dir_single))
        
        for directory_single in dirs:
            self.caltabs.append(caltab(os.path.basename(directory_single), 
                                       [os.path.join(directory_single, 'CALTAB_L1'), os.path.join(directory_single, 'CALTAB_R1')],
                                        np.loadtxt(os.path.join(directory_single, 'freq_ranges')) ) )
        
    def __copy_caltabs_to_config(self, directory):
        '''
        Copies the caltabs to the .config/ssddr directory
        '''
        for c in self.caltabs:
            target_dir = os.path.join(directory, c.label)
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
            c.save_caltab(target_dir)
            with open(os.path.join(target_dir, 'freq_ranges'), 'w+') as freq_file:
                freq_file.write(f"{c.freqRange[0]}\n")
                freq_file.write(f"{c.freqRange[1]}")

    def fitChebyForScan(self, bbc, order, scannr):
        '''
        Fits polynomial for specified BBC, with specified order and for 
        specified scan, returns tables X and Y with polynomial and fit residuals
        '''
        polyTabX, polyTabY, self.polyTabResiduals = self.obs.mergedScans[scannr].fitCheby(bbc, order, self.fitBoundsChannels)
        if not self.isOnOff:
            return polyTabX, polyTabY, self.__halveResiduals(self.polyTabResiduals)
        else:
            return polyTabX, polyTabY, self.polyTabResiduals#self.__halveResiduals(self.polyTabResiduals)

    def __halveResiduals(self, residuals):
        chanCnt = int(len(residuals) / 2)
        self.tmpHalvedRes = (residuals[:chanCnt] - residuals[chanCnt:]) / 2.0
        return self.tmpHalvedRes

    def __openTheArchive(self, tarName):
        """
        Opens the archive and lists the insides of it
        """
        print("-----> Loading archive \"" + tarName + "\"...")
        archive = tarfile.open(tarName, 'r')
        # ---- securing from odd number of scans ---
        self.scansList = archive.getnames()
        self.fullScansList = self.scansList
        if len(self.scansList) % 2 != 0:
            self.scansList = self.scansList[:-1]
        # ------------------------------------------
        # --- securing from an event, where temporary dir was not removed ---
        try:
            os.mkdir(self.tmpDirName)
        except FileExistsError:
            for i in os.listdir(self.tmpDirName):
                os.remove(os.path.join(self.tmpDirName, i))
        # ------------------------------------------
        archive.extractall(path='./' + self.tmpDirName)
    
    def __processData(self):
        '''
        processes the tared data
        '''
        self.obs = observation(self.tmpDirName, self.scansList, self.isOnOff)
        for i in self.fullScansList:
            os.remove(self.tmpDirName + "/" + i)
        os.rmdir(self.tmpDirName)
    
    def calculateFitRMS(self, data):
        sum = 0.0
        no = 0
        '''
        for i in self.fitBoundsChannels:
            sum += np.sum( data[i[0]:i[1]] * data[i[0]:i[1]])
            no += (i[1] - i[0])
        '''
        for i in data:
            sum += i*i
        no = len(data)      
        return np.sqrt( (1.0 / no) * sum)

    def alternateRMSCalc(self, data):
        if len(data) < 1024:
            return -1
        sum = 0.0
        no = 0.0
        bounds = [ [20, 400], [len(data) - 400, len(data) - 20]]
        for i in bounds:
            sum += np.sum( data[i[0]:i[1]] * data[i[0]:i[1]])
            no += (i[1] - i[0])
        return np.sqrt( (1.0 / no) * sum)


    def __getZData(self):
        ztb = [90.0 - scan.EL for scan in self.obs.scans]
        #print(ztb)
        return np.asarray(ztb)
    
    def __getTsysData(self):
        tsystb = []
        for i in range(self.noOfBBC):
            tsystb.append([scan.tsys[i] / 1000.0 for scan in self.obs.scans])
        return np.asarray(tsystb)

    def __getTotalFluxData(self):
        totalFlux = []
        for i in range(self.noOfBBC):
            totalFlux.append( [np.sum(np.sqrt(scan.pols[i] * scan.pols[i])) for scan in self.obs.mergedScans])
        return np.asarray(totalFlux)

    def __getTimeData(self):
        time = np.asarray([i.mjd for i in self.obs.scans])
        self.startEpoch = time.min()
        return (time - self.startEpoch) * 24.0

    def __getMergedTimeData(self):
        time = np.asarray([i.mjd for i in self.obs.mergedScans])
        time -= self.startEpoch
        time *= 24
        return time

    def __getOutliers(self, totalFluxTab):
        outlier_table = []
        for tab in totalFluxTab:
            outlier_table.append(IsolationForest(contamination = 0.2).fit_predict(X = tab.reshape(-1,1), y = None))
        return np.asarray(outlier_table)

    def findBrokenScan(self,
                       scanIndex: int,
                       tmpScanData: np.ndarray,
                       broken_scan_detector):
        """
        Asseses if the scan is broken, using Neural Network
        And total flux outlier finding (computed upon creation of this object with Isolation Forest)
        :param scanIndex:
        :param tmpScanData:
        :param broken_scan_detector:
        :return:
        """
        flag_network = self.checkIfBroken(
                model = broken_scan_detector,
                data = tmpScanData)
        flag_outlier = self.outlierTable[self.actualBBC-1][scanIndex] == -1
        return flag_network or flag_outlier # return True if at least one of these is True

    def addToStack(
            self,
            scanIndex: int,
            annotator,
            broken_scan_detector):
        if self.__checkIfStacked(scanIndex):
            print(f"-----> scan no. {scanIndex+1} is already stacked!")
            return
        # -- prepare for cheby fit --
        # check if scan is broken
        tmp_scan_data = self.obs.mergedScans[scanIndex].pols[self.actualBBC-1]
        is_scan_broken = self.findBrokenScan(
            scanIndex = scanIndex,
            tmpScanData = tmp_scan_data,
            broken_scan_detector = broken_scan_detector)
        if is_scan_broken:
            return

        # label channels
        channel_categories = self.getFitBoundChannels(
            model = annotator,
            data = tmp_scan_data
        )

        # remove RFI
        remove_table = self.extract_category_bounds(channel_categories, cat_to_bound = 2)
        self.obs.mergedScans[scanIndex].removeChannels(self.actualBBC, remove_table)

        # colors = {
        #     0: 'grey',
        #     1: 'green',
        #     2: 'red',
        #     3: 'blue'}

        # -- temporary plot block --
        # import matplotlib.pyplot as plt
        # plt.style.use('ggplot')
        # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 7), sharex=True, sharey=True)
        # for category in range(4):
        #     # ---- predicted labels ----
        #     tmp_data = tmp_scan_data.copy()
        #     indices = channel_categories == category
        #     tmp_data[~indices] = np.nan
        #     axes.plot(list(range(len(tmp_data))), tmp_data, c=colors[category])
        # axes.set_title(f"Is this scan broken? {is_scan_broken}")
        # plt.tight_layout()
        # source_indicator = str(round(self.obs.mjd, 3)).replace(".", "")
        # plt.savefig(f"{scanIndex}_{source_indicator}_bbc_{self.actualBBC}.png", dpi=300)
        # plt.close(fig)
        # # ----------------------------

        self.fitBoundsChannels = self.extract_category_bounds(channel_categories, cat_to_bound = 0)
        self.scans_proceed[scanIndex] = 'ADDED'
        x,y,residuals, = self.fitChebyForScan(self.actualBBC, self.fitOrder, scanIndex)
        self.stack.append(residuals)
        self.scansInStack.append(scanIndex)

    def checkIfBroken(self, model, data: np.ndarray):
        category_labels = model.predict(data.reshape(1, 4096, 1))
        cat = [np.argmax(s) for s in category_labels]
        print(np.asarray(cat))
        if np.asarray(cat)[0] == 0:
            return False # scan is ok
        else:
            return True # scan is broken

    def getFitBoundChannels(self, model, data: np.ndarray):
        category_labels = model.predict(data.reshape(1, 4096, 1))
        category_table = np.asarray([int(np.argmax(s)) for s in category_labels[0]])
        return category_table

    def extract_category_bounds(self, category, cat_to_bound: int = 0):
        new_bounds = []
        i = 25
        while i < len(category) - 25:
            if category[i] == cat_to_bound:
                start = i
                while i < len(category) and category[i] == cat_to_bound:
                    i += 1
                end = i - 1
                new_bounds.append([start, end])
            else:
                i += 1
        return new_bounds

    def discardFromStack(self, scanIndex):
        self.scans_proceed[scanIndex] = 'DISCARDED'

    def calculateSpectrumFromStack(self):
        if len(self.stack) == 0:
            self.meanStack = np.zeros(2048).astype(float)
        else:
            self.meanStack = np.mean(self.stack, axis=0)
        self.finalFitRes = self.meanStack.copy()
        return self.finalFitRes

    def deleteFromStack(self, scanIndex):
        if not self.__checkIfStacked(scanIndex):
            print(f"-----> scan no. {scanIndex+1} was not stacked, so it cannot be removed!")
            return
        i = self.scansInStack.index(scanIndex)
        self.scansInStack.pop(i)
        self.stack.pop(i)
        self.scans_proceed[scanIndex] = 'DISCARDED'

    def __checkIfStacked(self, indexNo):
        if indexNo in self.scansInStack:
            return True
        else:
            return False
    
    def setLHCTab(self):
        self.LHCTab = np.mean(self.stack, axis=0)
    def setRHCTab(self):
        self.RHCTab = np.mean(self.stack, axis=0)

    def __generateVelTab(self):
        '''
        It is to make and return proper array of Radial Velocity, with respect to the local standard of rest (LSR)
        Generally, the data is rotated to this frame upon loading, so here we are just generating the table straightfoward
        Nothing really complicated
        '''
        velocity = self.obs.scans[0].vlsr # km/s
        restfreq = self.obs.scans[0].rest # MHz
        if not self.isOnOff:
            freq_rang = self.obs.scans[0].bw / 2.0 # MHz
            nchans = self.obs.scans[0].NNch / 2.0
        else:
            freq_rang = self.obs.scans[0].bw
            nchans = self.obs.scans[0].NNch

        c = 299792.458 # km/s
        beta = velocity / c
        gamma  = 1.0 / np.sqrt(1.0 - beta**2.0)
        fcentr = restfreq * (gamma * (1.0 - beta))
        fbegin = fcentr - freq_rang / 2.0
        fend = fcentr + freq_rang / 2.0
        #self.auto_prepared_to_fft = zeros((4, self.NN), dtype=complex128) # docelowa
        freqsTab = np.zeros((4, int(nchans)), dtype=np.float64)
        velsTab = np.zeros((4, int(nchans)), dtype=np.float64)
        for i in range(len(fbegin)):
            freqsTab[i] = np.linspace(fbegin[i], fend[i], int(nchans))
            velsTab[i] = - c * ( (freqsTab[i] / restfreq[i] ) - 1.0 )
            velsTab[i] = velsTab[i][::-1]
        return velsTab

    def removeChannels(self, BBC, scanNumber, removeTab):
        self.obs.mergedScans[scanNumber].removeChannels(BBC, removeTab)
    
    def cancelRemoval(self, BBC, scanNumber):
        self.obs.mergedScans[scanNumber].cancelRemove(BBC)
    
    def fitChebyToFinalSpec(self, BBC):
        fitVels, fitData = self.__getDataFromRanges(BBC, self.finalFitBoundChannels)
        poly = np.polyfit(fitVels, fitData, self.fitOrder)
        polytabX = np.linspace(self.velTab.min(), self.velTab.max(), len(self.meanStack))
        polytabY = np.polyval(poly, polytabX)
        self.finalFitRes -= polytabY

    def __getDataFromRanges(self, BBC, ranges):
        fitData = []
        fitVels = []
        for i in ranges:
            fitData.extend(self.finalFitRes[i[0]:i[1]])
            fitVels.extend(self.velTab[BBC-1][i[0]:i[1]])
        #print(len(fitVels), len(fitData))
        return fitVels, fitData
    
    def convertVelsToChannels(self, BBC, velTab):
        chanTab = []
        for i in velTab:
            minChan = -1
            maxChan = -1
            # find minChan
            for j in range(len(self.velTab[BBC])):
                if i[0] < self.velTab[BBC][j]:
                    minChan = j
                    break
            # find maxChan
            for j in range(len(self.velTab[BBC])):
                if i[1] < self.velTab[BBC][j]:
                    maxChan = j
                    break
            
            if i[0] > self.velTab[BBC].max():
                minChan = len(self.velTab[BBC])-1

            if i[1] > self.velTab[BBC].max():
                maxChan = len(self.velTab[BBC])-1

            if minChan != -1 and maxChan != -1:
                chanTab.append([minChan, maxChan])
        return chanTab
    
    def removeChansOnFinalSpectrum(self, channelsToRemove):
        for i in channelsToRemove:
            minChan = i[0]
            maxChan = i[1]
            print(f'------> Removing from channels {minChan} to {maxChan}')
            for j in range(minChan, maxChan, 1):
                self.finalFitRes[j] = self.__interpolateFinal(minChan, maxChan, j)

    def __interpolateFinal(self, minChan, maxChan, j):
        '''
        interpolate between specified channels for channel j
        '''
        y1 = self.finalFitRes[minChan]
        x1 = minChan
        y2 = self.finalFitRes[maxChan]
        x2 = maxChan
        a = (y1-y2) / (x1 - x2)
        b = y1 - a * x1
        return a * j + b
    
    def cancelChangesFinal(self):
        print(f'------> cancelling all of the changes!')
        self.finalFitRes = self.meanStack.copy()
    
    def clearStack(self, pol='LHC'):
        if pol == 'LHC':
            self.finalLHC = self.finalFitRes.copy()
        elif pol == 'RHC':
            self.finalRHC = self.finalFitRes.copy()

        self.clearStackedData()
    
    def clearStackedData(self):
        self.meanStack = []
        self.stack = []
        self.scansInStack = []
        self.finalFitRes = []
        self.scans_proceed = self.__makeScansProceedTable()

    def setActualBBC(self, BBC):
        self.actualBBC = BBC

    def getFinalPols(self):
        I = (self.finalLHC / 2.0) + (self.finalRHC / 2.0)
        V  = self.finalRHC - self.finalLHC
        return I, V, self.finalLHC, self.finalRHC
    
    def __tryToLoadCaltabs(self):
        """
        Tries to load caltab from a server
        adresses are loaded from the config file
        """
        try:
            confile = configparser.ConfigParser()
            confile.read(os.path.join(self.DE_CAT,'caltabPaths.ini'))
            for i in confile.sections():
                tab_paths = [confile[i]['lhcCaltab'], confile[i]['rhcCaltab']]
                freq_ranges = [ float(confile[i]['minFreq']), float(confile[i]['maxFreq'])]
                self.caltabs.append(caltab(i, tab_paths, freq_ranges))
            self.caltabsLoaded = True
            return True
        except:
            self.caltabsLoaded = False
            return False
    
    def findProperCaltabIndex(self):
        '''
        Assumes the data is loaded
        '''
        properIndex = int(1e9)
        for i in range(len(self.caltabs)):
            if self.caltabs[i].inRange(self.obs.scans[0].rest[0] / 1000.0):
                properIndex = i
        return properIndex
    
    def findCalCoefficients(self) -> bool:
        '''
        Finds proper caltab coefficient, based on mean epoch of the acquired data
        returns:
            True - if the caltab is longer than current epoch
            False - if the caltab is shorter
        '''
        date = self.obs.mjd
        if self.properCaltabIndex == int(1e9):
            self.calCoeffLHC = 1.0
            self.calCoeffRHC = 1.0
            return True
        else:
            self.calCoeffLHC = self.caltabs[self.properCaltabIndex].findCoeffs(date)[0]
            self.calCoeffRHC = self.caltabs[self.properCaltabIndex].findCoeffs(date)[1]
            self.printCalibrationMessage(self.calCoeffLHC, self.calCoeffRHC, date, lhc=True)
            if self.caltabs[self.properCaltabIndex].getMaxEpoch() < date:
                return False
            else:
                return True

    def printCalibrationMessage(self, calCoeffLHC, calCoeffRHC, epoch, lhc=True):
        print('-----------------------------------------')
        print(f'-----> CALIBRATION PROMPT:')
        if self.properCaltabIndex == -1:
            print(f'-----> Did not fint any suitable caltab!')
            print('-----------------------------------------')
            return
        print(f'-----> Using {self.caltabs[self.properCaltabIndex].label} to calibrate the data')
        print(f'-----> (LHC): Coefficient is {calCoeffLHC} for MJD {int(epoch)}')
        print(f'-----> (RHC): Coefficient is {calCoeffRHC} for MJD {int(epoch)}')
        if lhc==True:
            maxEpoch = self.caltabs[self.properCaltabIndex].lhcMJDTab.max()
            minEpoch = self.caltabs[self.properCaltabIndex].lhcMJDTab.min()
        else:
            maxEpoch = self.caltabs[self.properCaltabIndex].rhcMJDTab.max()
            minEpoch = self.caltabs[self.properCaltabIndex].rhcMJDTab.min()

        if epoch < maxEpoch and epoch > minEpoch:
            print(f'-----> Epoch is placed between MJD {minEpoch} and {maxEpoch}')
            print(f'-----> Looks OK')
        else:
            print(f'-----> Epoch is placed OUTSIDE caltab (MJD from {minEpoch} to {maxEpoch})')
            print(f'-----> Check CAREFULLY if this is ok.')
        print('-----------------------------------------')
    
    def calibrate(self, lhc = True):
        if lhc:
            self.meanStack *= self.calCoeffLHC
            self.finalFitRes *= self.calCoeffLHC
        else:
            self.meanStack *= self.calCoeffRHC
            self.finalFitRes *= self.calCoeffRHC
        return self.finalFitRes
    
    def uncalibrate(self, lhc = True):
        if lhc:
            self.meanStack = self.meanStack / self.calCoeffLHC
            self.finalFitRes = self.finalFitRes / self.calCoeffLHC
        else:
            self.meanStack = self.meanStack / self.calCoeffRHC
            self.finalFitRes = self.finalFitRes / self.calCoeffRHC
        return self.finalFitRes
    
    def __makeScansProceedTable(self):
        returned_tab = []
        for i in range(len(self.obs.mergedScans)):
            returned_tab.append('NOT_PROCEEDED')
        return returned_tab
    
    def checkIfAllScansProceeded(self):
        if 'NOT_PROCEEDED' in self.scans_proceed:
            return False
        else:
            return True
    
    def setFitOrder(self, fitOrder):
        self.fitOrder = fitOrder
        print("-----> Fit order changed to", fitOrder)
    
    def saveReducedDataToFits(self):
        # -- filename --
        fname = self.obs.scans[0].sourcename + '_' + str(round(self.obs.mjd,3)).replace(".", "") + ".fits"
        result_filename = os.path.join(self.dataTmpDirectory, fname)
        # -- data tables --
        polLHC = np.array(self.finalLHC, dtype=np.float64)
        polRHC = np.array(self.finalRHC, dtype=np.float64)
        columnPol1 = fits.Column(name='Pol 1', format='E', array=polLHC[::-1])
        columnPol2 = fits.Column(name='Pol 2', format='E', array=polRHC[::-1])
        # -- headers --
        primaryHeader = self.__constructPrimaryHeader()
        dataHeader = fits.BinTableHDU.from_columns([columnPol1, columnPol2])
        self.__addToSecondaryHeader(dataHeader.header)
        # -- filesave --
        hdul = fits.HDUList([primaryHeader, dataHeader])
        hdul.writeto(result_filename, overwrite=True)
        return result_filename


    '''
    Methods to help generate FITS file
    '''
    def __constructPrimaryHeader(self):
        hdr = fits.Header()
        hdr['ORIGIN']   = ('TCfA    '             , "Torun Centre for Astronomy" )                                                                 
        hdr['SOFTWARE'] = ('MYLOVE  '             , "16k channel Autocorrelator Spectrometer")              
        hdr['VERSION']  = ('1.0  '             , "Software release version.")                
        hdr['AUTHOR']   = ('Michal Durjasz (TCfA)', "Software author.")                         
        hdr['CONTACT']  = ('md@astro.umk.pl'    , "Contact to software author.")
        primaryHDU = fits.PrimaryHDU(header=hdr)
        return primaryHDU
    
    def __addToSecondaryHeader(self, hdr):
        fscan = self.obs.scans[0]
        hdr['AUTHOR'] = ('Michal Durjasz')
        hdr['INSTRUME'] = ('MYLOVE')
        hdr['TELESCOP'] = ('RT4     ')
        hdr['ORIGIN'] = ('TRAO    ')
        hdr['OBSERVER'] = ('Michal Durjasz')
        hdr['OBJECT'] = (fscan.sourcename, 'Name of the observed object')
        hdr['EQUINOX'] = (2000.0, 'Equinox of celestial coordinate system')
        strRA, strDEC = self.__makeRAandDECstring(fscan)
        hdr['SRC_RA'] = (strRA, 'RA of source')
        hdr['SRC_DEC'] = (strDEC, 'DEC of source')
        hdr['DATE-OBS'] = (fscan.isotime, 'Format: \'yyyy-mm-ddTHH:MM:SS[.sss]\'')
        hdr['FREQ'] = (float(fscan.rest[0]) * 10**6, 'Frequency in HZ')
        FCENTR, FBEGIN, FEND = self.__calculateFbeginAndRest(float(fscan.vlsr[0]), float(fscan.rest[0]) * 10**6, fscan.bw[0] / 2.0)
        hdr['FRQ_BEG'] = (FBEGIN, 'Frequency at the beginning [MHz].')
        hdr['FRQ_MID'] = (FCENTR, 'Frequency at the middle of the spectrum.')
        hdr['FRQ_END'] = (FEND, 'Frequency at the end of the spectrum [MHz].')
        if not self.isOnOff:
            hdr['FRQ_RANG'] = (float(fscan.bw[0]/2.0), 'Bandwidth [MHz] ')
        else:
            hdr['FRQ_RANG'] = (float(fscan.bw[0]), 'Bandwidth [MHz] ')
        hdr['VSYS'] = (float(fscan.vlsr[0]), 'System velocity in km/s.')
        hdr['DOPP_VSU'] = (0.0, 'Suns velocity')
        hdr['DOPP_VOB'] = (0.0, 'Observers velocity.')
        hdr['DOPP_VTO'] = (0.0, 'Finall Doppler velocity for source')
        hdr['RESTFRQ'] = float(fscan.rest[0]) * 10 ** 6
        # --------------- UWAGI MOLEKUŁY ---------
        if fscan.rest[0] < 6034.0:
            hdr['MOLECULE'] = 'exOH 6031'
        elif fscan.rest[0] > 6034.0 and fscan.rest[0] < 6100.0:
            hdr['MOLECULE'] = 'exOH 6035'
        elif fscan.rest[0] > 6100.0 and fscan.rest[0] < 7000.0:
            hdr['MOLECULE'] = 'CH3OH 6668'
        elif fscan.rest[0] > 7000.0 and fscan.rest[0] < 13000.0:
            hdr['MOLECULE'] = 'CH3OH 12178'
        else:
            hdr['MOLECULE'] = 'H20 22235'   
        # -------------------------------------------------
        hdr['TIME'] = fscan.isotime,
        hdr['AZ'] = round(fscan.AZ,4)
        hdr['Z'] = round(90.0 - fscan.EL,4)
        hdr['SCAN_TYP'] = 'FINAL   '
        hdr['TSYS1'] = (float(fscan.tsys[self.bbcs_used[0]-1]) / 1000.0, 'Measured Tsys pol 1')
        hdr['TSYS2'] = (float(fscan.tsys[self.bbcs_used[1]-1]) / 1000.0, 'Measured Tsys pol 2')
    
    def __calculateFbeginAndRest(self, Vlsr, restFreq, bw):
        c = 299792.458
        restFreq  /= 1e6
        beta = Vlsr / c
        gamma = 1.0 / np.sqrt(1.0-beta**2.0)
        FCENTR = float(restFreq) * (gamma * (1.0-beta))
        FBEGIN = float(FCENTR) - float(bw) / 2.0
        FEND = float(FCENTR) + float(bw) / 2.0
        return FCENTR, FBEGIN, FEND

    def __makeRAandDECstring(self, fscan):
        str_rah = self.append0(str(fscan.rah))
        str_ram = self.append0(str(fscan.ram))
        str_ras = self.append0(str(fscan.ras))

        str_decd = self.append0(str(fscan.decd))
        str_decm = self.append0(str(fscan.decm))
        str_decs = self.append0(str(fscan.decs))
        # '06h08m53s'
        # '21d38m29s'
        strRA = str_rah + 'h' + str_ram + 'm' + str_ras + 's'
        strDEC = str_decd + 'd' + str_decm + 'm' + str_decs + 's'
        return strRA, strDEC
    
    def append0(self, napis):
        if napis[0] != '-':
            if len(napis) == 1:
                napis = '0' + napis
            return napis
        else:
            if len(napis) == 2:
                napis = napis[0] + '0' + napis[1]
            return napis
    
    def calculateSNR(self):
        spectr = self.calculateSpectrumFromStack()
        if len(spectr) == 1 and spectr[0] == -1:
            return 0
        noise = self.alternateRMSCalc(spectr)
        snr = spectr.max() / noise
        return snr