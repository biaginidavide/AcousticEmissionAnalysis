# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 17:37:39 2022

@author: davidebiagini

sample extraction of acoustic emission parameters using Vallene AE python tool 
available at https://pyvallenae.readthedocs.io/en/stable/

in order to run the code vallenae must be properly installed, see instructions https://pyvallenae.readthedocs.io/en/stable/

In this code the .tradb and .pridb are analyzed and acoustic emission features are 
calculated and stored

"""


import os
import numpy as np
import vallenae as vae
from scipy.fft import fft, fftfreq
from scipy import signal
from numpy import asarray
from numpy import savetxt
import pywt
import pywt.data


"""
Functions 

"""

def Find_nearest(array, value):                              # function to find the index of an array in wich the element is closer to a defined value
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def Freqparameters(data: np.ndarray, sf: np.ndarray):        # function to extract form a fourier spectrum
                                                             # peak frequency, centroid frequency, partial power, average peak frequency
                                                            
    numcentr = 0.0
    dencentr = 0.0
    PPw1 = 0.0
    PPw2 = 0.0
    PPw3 = 0.0
    PPw4 = 0.0
    Pt   = 0.0    
    deltafreq = sf[len(sf)-1]-sf[len(sf)-2]    
    for l in range(len(sf)):
          numcentr= numcentr + (sf[l]*(np.abs(data[l])))
          dencentr= dencentr + (np.abs(data[l]))
          Pt = Pt + deltafreq * data[l]
          if sf[l] < 200000:                   # define 4 intervals for partial power 
              PPw1 = PPw1 + deltafreq * data[l]
          if sf[l] > 200000 and sf[l] < 300000:
              PPw2 = PPw2 + deltafreq * data[l]         
          if sf[l] > 300000 and sf[l] < 400000:
              PPw3 = PPw3 + deltafreq * data[l]
          if sf[l] > 400000 and sf[l] < 500000:
              PPw4 = PPw4 + deltafreq * data[l]              
    peakfreq = sf[np.argmax(data)]    
    centroid = numcentr/dencentr
    averagepeak = np.sqrt(centroid * peakfreq)    
    return peakfreq, centroid, averagepeak, PPw1/Pt, PPw2/Pt, PPw3/Pt, PPw4/Pt

def Fourier(y: np.ndarray, t: np.ndarray):                   # function to calculate fourier spectrum
        n = y.size
        Amplitude    = 2*np.abs(fft(y))
        sample_freq  = (fftfreq(n , np.abs(t[n-1] - t[n-2]))) 
        k = Amplitude.size
        return Amplitude[0:int(k/2)], sample_freq[0:int(k/2)]

def Threshold(a: np.ndarray, thresh: float) -> np.ndarray:   # function to set to zero array elements below a defined treshold
    a = [0 if np.abs(a_) < thresh else a_ for a_ in a]
    return a

def WaveletEnergy(data: np.ndarray) -> float:                # function to calculate energy of wavelet decomposition
    agg: float = 0
    for sample in data:
        agg += sample ** 2
    return agg

"""
Main script

"""

# SELECT RAW FILE TO BE ANALYZED

HERE = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
PRIDB = os.path.join(HERE, "AE_rawdata/comp90.pridb")  # path to .pridb file to be analyzed
TRADB = os.path.join(HERE, "AE_rawdata/comp90.tradb")  # path to .tradb file to be analyzed

pridb = vae.io.PriDatabase(PRIDB)
hits = pridb.read_hits()
df_parametric = pridb.read_parametric()
channel_order = hits["channel"].to_numpy()
arrival_times = hits["time"].to_numpy()
trai = hits["trai"].to_numpy()
pa0 = df_parametric["pa0"].to_numpy()
pa1 = df_parametric["pa1"].to_numpy()
timeparametric = df_parametric["time"].to_numpy()
pridb.close()

# Initialize lists for acoustic parameters
Pa0 = []; Pa1 = []; abstime=[]; trai_ch1=[];
PE1 = []; PE2 = []; PE3 = []; PE4 = []; PE5 = []; PE6 = []; PE7 = []; PE8 = [];
PE9 = []; PE10 = []; PE11 = []; PE12 = []; PE13 = []; PE14 = []; PE15 = []; PE16 = [];
E=[]; Time = []; peakf = []; centroidf = []; weightedpeakf = []; 
PPw1 = []; PPw2 = []; PPw3 = []; PPw4 = []; amplitude = []; 
counts = []; rise_time = []; duration = []; totalenergy = [];
index = []; sparsities = [];

for j in range(len(channel_order)): 
            if channel_order[int(j)] == 1: # select channel to be analyzed 
                        if trai[int(j)] != 0 :
                                  n = Find_nearest(timeparametric, arrival_times[int(j)])
                                  trai_ch1.append(trai[int(j)])
                                  abstime.append(arrival_times[int(j)])
                                  Pa1.append(pa1[n])
                                  Pa0.append(pa0[n])

#_____________________________________________________________________________
# Transient waveform analysis

p=0
for q in trai_ch1: 
    if q !=0 :        
       with vae.io.TraDatabase(TRADB) as tradb: 
            dt = 1000/2048   # sampling time
            sr = 2000000     # sample rate (Hz)
            TRAI = q
            yf=[]
            xf=[]
            y,  t =   tradb.read_wave(TRAI)      
            
            sos = signal.butter(10, [0.05,0.99], 'bandpass', output='sos')
            y = signal.sosfilt(sos, y)
                  
#           WAVELET packet TRASFORM
            wavecoeff = pywt.WaveletPacket(y, wavelet='db32', mode='symmetric')         
            
            # filter out all coefficients < 10% max coefficient__________________________________________________
            thresh = 0.1 * np.max([np.max(wavecoeff['aaaa'].data),
                                   np.max(wavecoeff['aaad'].data),
                                   np.max(wavecoeff['aadd'].data),
                                   np.max(wavecoeff['aada'].data),
                                   np.max(wavecoeff['adda'].data),
                                   np.max(wavecoeff['addd'].data),
                                   np.max(wavecoeff['adad'].data),
                                   np.max(wavecoeff['adaa'].data),
                                   np.max(wavecoeff['adaa'].data),
                                   np.max(wavecoeff['ddaa'].data),
                                   np.max(wavecoeff['ddad'].data),
                                   np.max(wavecoeff['dddd'].data),
                                   np.max(wavecoeff['ddda'].data),
                                   np.max(wavecoeff['dada'].data),
                                   np.max(wavecoeff['dadd'].data),
                                   np.max(wavecoeff['daad'].data)])
            
            c1 = Threshold(wavecoeff['aaaa'].data , thresh)
            c2 = Threshold(wavecoeff['aaad'].data , thresh)
            c3 = Threshold(wavecoeff['aadd'].data , thresh)
            c4 = Threshold(wavecoeff['aada'].data , thresh)
            c5 = Threshold(wavecoeff['adda'].data , thresh)
            c6 = Threshold(wavecoeff['addd'].data , thresh)
            c7 = Threshold(wavecoeff['adad'].data , thresh)
            c8 = Threshold(wavecoeff['adaa'].data , thresh)
            c9 = Threshold(wavecoeff['ddaa'].data , thresh)
            c10 = Threshold(wavecoeff['ddad'].data , thresh)
            c11 = Threshold(wavecoeff['dddd'].data , thresh)
            c12 = Threshold(wavecoeff['ddda'].data , thresh)
            c13 = Threshold(wavecoeff['dada'].data , thresh)
            c14 = Threshold(wavecoeff['dadd'].data , thresh)
            c15 = Threshold(wavecoeff['daad'].data , thresh)
            c16 = Threshold(wavecoeff['daaa'].data , thresh)

            wavecoeff.__setitem__('aaaa', c1)
            wavecoeff.__setitem__('aaad', c2)
            wavecoeff.__setitem__('aadd', c3)
            wavecoeff.__setitem__('aada', c4)
            wavecoeff.__setitem__('adda', c5)
            wavecoeff.__setitem__('addd', c6)
            wavecoeff.__setitem__('adad', c7)
            wavecoeff.__setitem__('adaa', c8)
            wavecoeff.__setitem__('ddaa', c9)
            wavecoeff.__setitem__('ddad', c10)
            wavecoeff.__setitem__('dddd', c11)
            wavecoeff.__setitem__('ddda', c12)
            wavecoeff.__setitem__('dada', c13)
            wavecoeff.__setitem__('dadd', c14)
            wavecoeff.__setitem__('daad', c15)
            wavecoeff.__setitem__('daaa', c16)

            sig = pywt.WaveletPacket.reconstruct(wavecoeff)
            #_____________________________________________________________________________________________________
            
            PE1.append(WaveletEnergy(data = c1))
            PE2.append(WaveletEnergy(data = c2))
            PE3.append(WaveletEnergy(data = c3))
            PE4.append(WaveletEnergy(data = c4))
            PE5.append(WaveletEnergy(data = c5))
            PE6.append(WaveletEnergy(data = c6))
            PE7.append(WaveletEnergy(data = c7))
            PE8.append(WaveletEnergy(data = c8))       
            PE9.append(WaveletEnergy(data = c9))  
            PE10.append(WaveletEnergy(data = c10))  
            PE11.append(WaveletEnergy(data = c11))  
            PE12.append(WaveletEnergy(data = c12))  
            PE13.append(WaveletEnergy(data = c13))  
            PE14.append(WaveletEnergy(data = c14))  
            PE15.append(WaveletEnergy(data = c15))  
            PE16.append(WaveletEnergy(data = c16))        

            # Extracting time domain - features
            totalenergy.append(vae.features.acoustic_emission.energy(sig, sr))
            A = vae.features.acoustic_emission.peak_amplitude(sig)      
            rise_time.append(vae.features.acoustic_emission.rise_time(sig, 0.1 * A, sr))
            duration.append(vae.features.acoustic_emission.duration(sig, 0.1 * A, sr))  
            counts.append(vae.features.acoustic_emission.counts(sig, 0.1 * A))
            amplitude.append(vae.features.conversion.amplitude_to_db(A))
          
            # Extracting frequency - features
            yf,  xf =  Fourier(sig,t)                     
            peakfreq_sig, centroid_sig, averagepeak_sig, PPw1_sig, PPw2_sig, PPw3_sig, PPw4_sig =  Freqparameters(data = yf, sf = xf)                  
            peakf.append                (peakfreq_sig)
            centroidf.append            (centroid_sig)
            weightedpeakf.append        (averagepeak_sig)
            PPw1.append                 (PPw1_sig)
            PPw2.append                 (PPw2_sig)
            PPw3.append                 (PPw3_sig)
            PPw4.append                 (PPw4_sig)
            p = p+1
            
# Store data in csv files
savetxt('Pa0.csv', asarray(Pa0), delimiter=',')
savetxt('Pa1.csv', asarray(Pa1), delimiter=',')
savetxt('Peakfreq.csv', asarray(peakf), delimiter=',')
savetxt('Centroid.csv', asarray(centroidf), delimiter=',')
savetxt('Weightedpeak.csv', asarray(weightedpeakf), delimiter=',')
savetxt('Rise.csv', asarray(rise_time), delimiter=',')
savetxt('Duration.csv', asarray(duration), delimiter=',')
savetxt('Counts.csv', asarray(counts), delimiter=',')
savetxt('Amplitude.csv', asarray(amplitude), delimiter=',')
savetxt('Timehits.csv', asarray(abstime), delimiter=',')
savetxt('Energy.csv', asarray(totalenergy), delimiter=',')
savetxt('PPw1.csv', asarray(PPw1), delimiter=',')
savetxt('PPw2.csv', asarray(PPw2), delimiter=',')
savetxt('PPw3.csv', asarray(PPw3), delimiter=',')
savetxt('PPw4.csv', asarray(PPw4), delimiter=',')
savetxt('WLCOMP1_E.csv', asarray(PE1), delimiter=',')
savetxt('WLCOMP2_E.csv', asarray(PE2), delimiter=',')
savetxt('WLCOMP3_E.csv', asarray(PE3), delimiter=',')
savetxt('WLCOMP4_E.csv', asarray(PE4), delimiter=',')
savetxt('WLCOMP5_E.csv', asarray(PE5), delimiter=',')
savetxt('WLCOMP6_E.csv', asarray(PE6), delimiter=',')
savetxt('WLCOMP7_E.csv', asarray(PE7), delimiter=',')
savetxt('WLCOMP8_E.csv', asarray(PE8), delimiter=',')
savetxt('WLCOMP9_E.csv', asarray(PE9), delimiter=',')
savetxt('WLCOMP10_E.csv', asarray(PE10), delimiter=',')
savetxt('WLCOMP11_E.csv', asarray(PE11), delimiter=',')
savetxt('WLCOMP12_E.csv', asarray(PE12), delimiter=',')
savetxt('WLCOMP13_E.csv', asarray(PE13), delimiter=',')
savetxt('WLCOMP14_E.csv', asarray(PE14), delimiter=',')
savetxt('WLCOMP15_E.csv', asarray(PE15), delimiter=',')
savetxt('WLCOMP16_E.csv', asarray(PE16), delimiter=',')