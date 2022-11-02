"""
Read and plot transient data
============================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import vallenae as vae
import pywt
import pycwt
from scipy.fft import fft, fftfreq, fftshift, ifft
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import scaleogram as scg 
import pywt
import scipy
import math
from scipy import stats 

def Find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def Threshold(a: np.ndarray, thresh: float) -> np.ndarray:  
    a = [0 if np.abs(a_) < thresh else a_ for a_ in a]
    return a

def Fourier(y,t):
        y = np.array(y)
        Amplitude = 2*np.abs(fft(y))
        sample_freq  = (fftfreq(y.size , np.abs(t[len(t)-1] - t[len(t)-2])))         
        freq=[]
        yA = []
        for i in range(len(sample_freq)):
            if sample_freq[i] > 0.0:
                freq.append(sample_freq[i])
                yA.append(Amplitude[i])
        return yA, freq

def plot_wavelet(coefficients, frequencies, sig, ax, time, title = '', ylabel = '', xlabel = '' ):
    im = ax.contourf(time, frequencies,coefficients, extend='both', cmap='jet', interpolation = 'none')
    cbar_ax = fig.add_axes([0.75, 0.35, 0.03, 0.45])
    cb = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cb.ax.xaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white', size = 10)
    ax.grid()
    return 

def amplitude_to_db(amplitude: float, reference: float = 1e-6) -> float:
    """
    Convert amplitude from volts to decibel (dB).
    Args:
        amplitude: Amplitude in volts
        reference: Reference amplitude. Defaults to 1 µV for dB(AE)
    Returns:
        Amplitude in dB(ref)
    """
    return 20 * np.log10(amplitude / reference)     

def waveletfilter(sig):
          wavecoeff = pywt.WaveletPacket(sig, wavelet='db32', mode='symmetric')               
          thresh = 0.1 * np.max([np.max(wavecoeff['aaaa'].data),
                                   np.max(wavecoeff['aaad'].data),
                                   np.max(wavecoeff['aadd'].data),
                                   np.max(wavecoeff['aada'].data),
                                   np.max(wavecoeff['adda'].data),
                                   np.max(wavecoeff['addd'].data),
                                   np.max(wavecoeff['adad'].data),
                                   np.max(wavecoeff['adaa'].data)])
          c1 = Threshold(wavecoeff['aaaa'].data , thresh)
          c2 = Threshold(wavecoeff['aaad'].data , thresh)
          c3 = Threshold(wavecoeff['aadd'].data , thresh)
          c4 = Threshold(wavecoeff['aada'].data , thresh)
          c5 = Threshold(wavecoeff['adda'].data , thresh)
          c6 = Threshold(wavecoeff['addd'].data , thresh)
          c7 = Threshold(wavecoeff['adad'].data , thresh)
          c8 = Threshold(wavecoeff['adaa'].data , thresh)
          wavecoeff.__setitem__('aaaa', c1)
          wavecoeff.__setitem__('aaad', c2)
          wavecoeff.__setitem__('aadd', c3)
          wavecoeff.__setitem__('aada', c4)
          wavecoeff.__setitem__('adda', c5)
          wavecoeff.__setitem__('addd', c6)
          wavecoeff.__setitem__('adad', c7)
          wavecoeff.__setitem__('adaa', c8)
          sig = pywt.WaveletPacket.reconstruct(wavecoeff)
          return sig

HERE = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()

# insert desired path
TRADB = os.path.join(HERE, "AE_rawdata/comp90.tradb")
PRIDB = os.path.join(HERE, "AE_rawdata/comp90.pridb")

pridb = vae.io.PriDatabase(PRIDB)
hits = pridb.read_hits()
df_parametric = pridb.read_parametric()
channel_order = hits["channel"].to_numpy()
routmean = hits["rms"].to_numpy()
arrival_times = hits["time"].to_numpy()
amp = hits["amplitude"].to_numpy()
trai = hits["trai"].to_numpy()
pa0 = df_parametric["pa0"].to_numpy()
pa1 = df_parametric["pa1"].to_numpy()
timeparametric = df_parametric["time"].to_numpy()
pridb.close()
Pa0=[]; Pa1=[]; Time = []; ind = []
abstime=[]; trai_ch=[]; abstime1=[];
A = []

for j in range(len(channel_order)):
            if channel_order[int(j)] == 1:             # insert desired channel (sensor number)
               if arrival_times[int(j)] > 380:         # insert desired time of the test
                 if amplitude_to_db(amp[int(j)]) > 85: # insert desired amplitude threshold
                        if trai[int(j)] != 0 :
                                       n = Find_nearest(timeparametric, arrival_times[int(j)])
                                       trai_ch.append(trai[int(j)])
                                       abstime.append(arrival_times[int(j)])
                                       Pa1.append(pa1[n])
                                       Pa0.append(pa0[n])
                                       A.append(amplitude_to_db(amp[int(j)]))

for q in trai_ch:
  with vae.io.TraDatabase(TRADB) as tradb:
          sig, t = tradb.read_wave(q)
          t=t * 1000
          sig = sig * 1000
          sos = signal.butter(10, [0.05, 0.7], 'bandpass', output='sos')
          sig = signal.sosfilt(sos, sig)
          sig = waveletfilter(sig)
          if np.max(sig) > 0.5:

            dt = t[100]-t[99]
            yf,xf = Fourier(sig,t)
            t1 = np.log2(4)
            t2 = np.log2(60)
            vec = np.linspace(t1,t2,80)
            scale_range = vec**2
            coefficients, frequencies = pywt.cwt(sig, scale_range, 'cmor1-1.5', dt)
            power = (np.abs(coefficients))**2    

            if amplitude_to_db(np.max(sig))> 85 :     
                  fig = plt.figure(figsize=(8, 8))
                  grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
                  x_sig = fig.add_subplot(grid[-1, 1:])             
                  x_sig.plot(t, sig)
                  x_sig.set_ylabel('signal [mV]')
                  x_sig.set_xlabel('time [ms]')
                  x_sig.grid()
                  y_freq = fig.add_subplot(grid[:-1, 0]) 
                  y_freq.plot(yf, xf)
                  y_freq.set_xlim(y_freq.get_xlim()[::-1])
                  y_freq.set_ylim([np.min(frequencies), np.max(frequencies)])
                  y_freq.set_ylabel('freq [kHz]')
                  y_freq.grid()
                  main_ax = fig.add_subplot(grid[:-1, 1:], sharex = x_sig)
                  plot_wavelet(np.abs(coefficients), frequencies, sig, main_ax, time = t, title = "CWT coeff", ylabel =' freq [kHz]', xlabel = 'time [ms]')
                  main_ax.set_yticklabels([])
                  main_ax.text(0.6, 670, 'abs(WL Coeff)',  fontsize = 12, color = 'white')
                  main_ax.set_title('Waveform type a', fontsize=16)
                  plt.show()
