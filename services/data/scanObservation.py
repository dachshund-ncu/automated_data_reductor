
'''
Klasa, przechowująca wszystkie skany
Data utworzenia: 05.03.2022
Właściciel: Michał Durjasz
'''


# -- importujemy potrzebne moduły --
# -- numpy --
from numpy import exp, loadtxt, nan, int64, sin, cos, asarray, sqrt, mean, pi, radians, zeros, inf, complex128, linspace
from numpy.fft import fft
# -----------
# -- math i mpmath --
from math import copysign
from cmath import sqrt as math_sqrt
from mpmath import nint
# -----------
# -- astropy --
from astropy.time import Time
from astropy.coordinates import SkyCoord, FK5
import astropy.units as u
# -------------
# -- sys --
from sys import argv, exit
from os import path
# ----------------
from .scanClass import scan
from .mergedScanClass import mergedScan

''' 
DEBUG
'''
import time

class observation:

    def __init__(self, tmpDir, listOfFiles, onOff = False):
        
        self.isOnOff = onOff
        # tablica ze skanami
        self.list_of_filenames = [tmpDir + "/" + i for i in listOfFiles]
        self.scans = self.read_scans(self.list_of_filenames)
        # wstępne ustawienia (współrzędne źródła, obserwatorium, precesja...)
        self.make_initial_settings()
        # parametry
        self.noOfScans = len(self.scans)
        # -----------------------------------
        self.proceed_scans()
        self.mergedScans = self.mergeScans(self.scans)
        # -----------------------------------
        self.__calculateMJD()


    # --- metoda wczytująca listę ---
    def read_list_of_files(self, list_filename):
        '''
        Reads list of files
        OBSOLETE, NOT USED ANYMORE
        SHOULD BE DELETED BY THE VERSION 1.11
        '''
        try:
            flenames = loadtxt(list_filename, dtype=str)
        except FileNotFoundError:
            print("-----> File \"%s\" does not exist! Exiting..." % list_filename)
            print("-----------------------------------------")
            exit()
        return flenames

    # --- metoda zwracająca listę z klasami "scan" ---
    def read_scans(self, list_of_filenames):

        # -- tablica z klasami --
        tab = []    

        # -- tworzymy klasy --
        for i in range(len(list_of_filenames)):
            tab.append(scan(list_of_filenames[i], self.isOnOff))
        
        # -- printujemy powiadomienie --
        print("-----> Loaded", len(tab), "scans")
        print("-----------------------------------------")
        return tab

    def make_initial_settings(self):
        # zestaw stałych, używanych globalnie w programie
        # --------------------
        # LOKALIZACJA OBS. PIWNICE
        self.dl_geog = 18.56406 # stopni
        self.szer_geog = 53.09546 # stopni (geodezyjna)
        self.height = 133.61 # wysokość n.p.m.
        # ---------------------
        # ZARZĄDZANIE WSPÓŁRZĘDNYMI
        # -- deklarujemy obiekt sky coord, w którym będą zawarte współrzędne --
        # -- WAŻNE: współrzędne są wzięte z PIERWSZEGO skanu --
        self.source_J2000 = SkyCoord(ra=self.scans[0].RA*u.hourangle, dec=self.scans[0].DEC*u.degree, frame=FK5, equinox='J2000')
        #(sunc_new.ra*u.degree).value
        # PRECESJA I NUTACJA 
        # -- do precesji deklarujemy nowy frame FK5 z epoką pierwszego skanu --
        self.frame_now = FK5(equinox="J" + str(self.scans[0].decimalyear))
        # -- by wykonać precesję i nutację wystarczy teraz: 
        self.source_JNOW = self.source_J2000.transform_to(self.frame_now)
        # będziemy robić precesję dla każdego skanu w self.proceed_scans
        # ------------------------------------------------------

        # ---------- WSPÓŁRZĘDNE GALAKTYCZNE -------------------
        self.l_ga = self.source_JNOW.galactic
        self.source_L = (self.l_ga.l*u.degree).value
        self.source_B = (self.l_ga.b*u.degree).value
        self.source_ld = int(self.source_L)
        self.source_lm = int(60.0 * (self.source_L % 1))
        self.source_bd = int(self.source_B)
        self.source_bm = int(60.0 * (self.source_B % 1))
        # ------------------------------------------------------

    def proceed_scans(self):

        # -- głwna super pętla robiąca wszystkie najważniejsze rzeczy --
        for i in range(len(self.scans)):

            # ------------- PRECESJA I NUTACJA ---------------
            # -- wykonujemy precesję na czas obecnego skanu --
            # -- do precesji deklarujemy nowy frame FK5 z epoką aktualnego skanu --
            frame_now = FK5(equinox="J" + str(self.scans[i].decimalyear))
            # -- by wykonać precesję i nutację wystarczy teraz: 
            source_JNOW = self.source_J2000.transform_to(frame_now)
            # -- zapisujemy wartości RA i DEC po precesji do nowych zmiennych --
            source_JNOW_RA = (source_JNOW.ra*u.degree).value
            source_JNOW_DEC = (source_JNOW.dec*u.degree).value
            # ------------------------------------------------
            # rzeczy powyżej są deklarowane lokalnie (bez przedrostka self.)
            # gdyż będą zmieniane w każdym obrocie pętli a po zakończeniu metody nie będą potrzebne
            # dlatego zostaną garbage collected

            # -- korekta funkcji autokorelacji --
            # ze względu na 2 i 3 poziomową kwantyzację etc.
            self.scans[i].correct_auto(scannr = i+1)
            
            # -- wygładzanie Hanninga --
            self.scans[i].hanning_smooth()

            # -- korekta na ruch ziemi --
            # obejmuje ona: 
            # 1. ruch wokół własnej osi
            # 2. ruch obiegowy wokół barycentrum US
            # 3. ruch względem lokalnej grupy gwiazd (LSR)
            # dwa pierwsze punkty są zrealizowane z dokładnością do ~ 1 cm/s
            # trzeci opiera się na metodzie, przepisanej żywcem z oryginalnego A2S
            # jej dokładność budzi pewne wątpliwości
            # argumenty doppset: 1: RA po precesji (deg), 2: DEC po precesji (deg)
            # 3: szerokość geograficzna (deg) 4: długość geogradiczna (deg, E > 0, W < 0)
            # 5: wysokość nad geoidą zi-emii
            # doppset wykonuje również rotację f. autokorelacji
            self.scans[i].doppset(source_JNOW_RA, source_JNOW_DEC, self.szer_geog, self.dl_geog, self.height)
            print("-----> scan %d: line rotated by %4.3f channels" % (i+1, round(self.scans[i].fcBBC[0],3)))

            # -- kilka statystyk liczymy --
            self.scans[i].do_statistics()

            # -- skalujemy tsys w mK --
            self.scans[i].scale_tsys_to_mK()

            # --- PRINTOWANIE ---
            # zakomentowane, normalnie tego nie potrzebujemy
            #self.scans[i].extended_print()

            # Najważniejsze!
            # -- robimy transformatę fouriera --
            self.scans[i].make_transformata_furiata()

            # -- kalibrujemy tsys --
            self.scans[i].calibrate_in_tsys()
        # printujemy belkę rozdzielającą
        print("-----------------------------------------")

    def save_to_file(self, flename="WYNIK.DAT"):
        
        # obiekt z zapisywanym plikiem
        fle = open(flename, "w+")

        # wypisujemy stosowne powiadomienie
        print("-----> Saving to file", flename)

        # otwieramy pętlę zapisującą
        # pętla zapisująca 
        for i in range(len(self.scans)): # i to skany (zazwyczaj 0 - 19)
            for ee in range(len(self.scans[i].auto)): # ee to BBC (zazwyczaj 0 - 3) 
                # ---- nagłówek ----
                fle.write("???\n")
                fle.write(repr(self.scans[i].rah).rjust(6) + repr(self.scans[i].ram).rjust(6) + repr(self.scans[i].ras).rjust(6) + repr(self.scans[i].decd).rjust(6) + repr(self.scans[i].decm).rjust(6) + repr(self.scans[i].decs).rjust(6) +"\n" )
                fle.write(repr(self.source_ld).rjust(6) + repr(self.source_lm).rjust(6) + repr(self.source_bd).rjust(6) + repr(self.source_bm).rjust(6) + "\n")
                fle.write(repr(self.scans[i].azd).rjust(6) + repr(self.scans[i].azm).rjust(6) + repr(self.scans[i].eld).rjust(6) + repr(self.scans[i].elm).rjust(6) + "\n" )
                fle.write(self.scans[i].datestring.rjust(10) + "\n")
                fle.write(repr(int(self.scans[i].STh)).rjust(6) + repr(int(self.scans[i].STm)).rjust(6) + repr(int(self.scans[i].STs)).rjust(6) + "\n")
                fle.write(repr(round(self.scans[i].tsys[ee] / 1000.0, 3)).rjust(8)  + "\n")
                fle.write("0".rjust(6) + "\n")
                fle.write(repr(ee).rjust(6) + "\n")
                fle.write("$$$\n")
                fle.write(repr(len(self.scans[i].spectr_bbc_final[ee])).rjust(12) + repr(int(self.scans[i].bw[ee])).rjust(15) + repr(0.25).rjust(15) + repr(self.scans[i].vlsr[ee]).rjust(11) + repr(self.scans[i].rest[ee]).rjust(18) + "\n")
                fle.write(self.scans[i].sourcename + "\n")
                fle.write("***" + "\n")
                fle.write(repr(int(self.scans[i].UTh)).rjust(8) + repr(int(self.scans[i].UTm)).rjust(8) + repr(int(self.scans[i].UTs)).rjust(8) + repr(int(self.scans[i].INT)).rjust(8) + "\n")
                # --- dane ----
                for j in range(len(self.scans[i].spectr_bbc_final[ee])): # j to kanały (zazwyczaj 0 - 2047)
                    # sprawdzamy, czy to co próbujemy zapisać nie jest aby za długie
                    if len(repr( round(self.scans[i].spectr_bbc_final[ee][j] ,1))) < 9:
                        # jeśli tak, zapisujemy
                        fle.write( repr( round(self.scans[i].spectr_bbc_final[ee][j] ,1) ).rjust(10) )
                    else:
                        # jak nie... to może sp......
                        fle.write( repr(000.0).rjust(10) )
                    # co 8 wpisów przechodzimy do nowej linii
                    if (j + 1) % 8 == 0:
                        fle.write("\n")
            # ---------------------------------------------------

        # -- zamykamy plik --
        fle.close()
        print("-----> Completed succesfully. Ending")
        print("-----------------------------------------")
    
    def mergeScans(self, scans):
        mergedScans = []
        for i in range(0,int(self.noOfScans),2):
            mergedScans.append(mergedScan(scans[i], scans[i+1]))
        return mergedScans
    
    def __calculateMJD(self):
        mjd = []
        for i in self.scans:
            mjd.append(i.mjd)
        self.mjd = mean(mjd)