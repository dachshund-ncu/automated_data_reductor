'''
Klasa, przechowująca merged skany
Data utworzenia: 05.03.2022
Właściciel: Michał Durjasz
'''

from .scanClass import scan
import numpy as np

class mergedScan:
    
    def __init__(self, scan1, scan2):
        '''
        Prosta metoda inicjująca klasę mergedScan
        pols to polaryzacje z pełną wstęgą
        '''
        self.pols = self.__mergeScans(scan1, scan2)
        self.backupPols = self.pols.copy()
        self.mjd = scan2.mjd
        self.tsys = [scan1.tsys, scan2.tsys]
    
    def fitCheby(self, bbc, order, ranges):
        '''
        metoda ma za zadanie dopasować wielomien
        do danych z zakresów przekazanych w 
        ranges
        zwraca tablice X i Y wielomianu, dodatkowo również rezydua dopasowania
        potrzebne przy generowaniu ostatecznej tablicy z widmem
        '''
        fitCHans, fitData = self.__getDataFromRanges(bbc-1, ranges)
        poly = np.polyfit(fitCHans, fitData, order)
        polyTabX = np.linspace(1, self.numberOfChannels, self.numberOfChannels)
        polyTabY = np.polyval(poly, polyTabX)
        polyTabResiduals = self.pols[bbc-1] - polyTabY
        return polyTabX, polyTabY, polyTabResiduals

    def __getDataFromRanges(self, bbc, ranges):
        fitData = []
        chans = np.linspace(1, self.numberOfChannels, self.numberOfChannels)
        fitChans = []
        for i in ranges:
            if i[0] < 0: 
                i[0] = 0
            if i[0] > len(self.pols[bbc])-1:
                i[0] = len(self.pols[bbc])-1
            if i[1] < 0: 
                i[1] = 0
            if i[1] > len(self.pols[bbc])-1:
                i[1] = len(self.pols[bbc])-1

            if i[0] > i[1]:
                ee1 = i[1]
                ee2 = i[0]
                i[0] = ee1
                i[1] = ee2

            fitData.extend(self.pols[bbc][i[0]:i[1]])
            fitChans.extend(chans[i[0]:i[1]])
        return np.asarray(fitChans), np.asarray(fitData)

    def __mergeScans(self, scan1, scan2):
        '''
        Ta metoda jest wywoływana tylko w __INIT__ 
        i ma na celu połączenie wynikowych widm z dwóch
        kolejnych skanów. Skany finalne przechowywane są w 
        spectr_bbc_final i powinny mieć 4 BBC, ale metoda
        i tak sama sprawdza, czy tak jest w istocie
        '''
        # ZAŁOŻENIE: każde BBC ma równą ilość kanałów!
        self.numberOfBBC = len(scan1.spectr_bbc_final)
        self.numberOfChannels = len(scan1.spectr_bbc_final[0])
        tmpScans = np.zeros( (self.numberOfBBC, self.numberOfChannels), dtype=np.float64 )
        for i in range(self.numberOfBBC):
            tmpScans[i] = (scan1.spectr_bbc_final[i] / 1000.0 - scan2.spectr_bbc_final[i] / 1000.0) / 2.0
        return tmpScans
    
    def removeChannels(self, BBC, removeTab):
        for i in removeTab:
            minChan = i[0]-1
            maxChan = i[1]-1
            print(f'------> Removing from channels {minChan} to {maxChan}')
            for j in range(minChan, maxChan,1):
                self.pols[BBC-1][j] = self.__interpolate(BBC, minChan, maxChan, j)
    
    def __interpolate(self, BBC, minChan, maxChan, j):
        '''
        interpolates between specified channels for channel j (useful in removal)
        '''
        y1 = self.pols[BBC-1][minChan]
        x1 = minChan
        y2 = self.pols[BBC-1][maxChan]
        x2 = maxChan
        a = (y1-y2) / (x1 - x2)
        b = y1 - a * x1
        return a * j + b
    
    def cancelRemove(self, BBC):
        self.pols[BBC-1] = self.backupPols[BBC-1]