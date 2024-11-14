"""
Read transient data and plot continuum wavelet transform
============================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import vallenae as vae
import pywt
from scipy.fft import fft, fftfreq

# Helper Functions
def find_nearest_index(array, value):  
    """Finds the index of the closest element in an array to a specified value."""
    return (np.abs(np.asarray(array) - value)).argmin()

def apply_threshold(arr: np.ndarray, threshold: float) -> np.ndarray:
    """Sets elements in an array to zero if they fall below a threshold."""
    return np.array([x if abs(x) >= threshold else 0 for x in arr])

def compute_fourier(y, t):
    """Calculates Fourier spectrum of a signal."""
    amplitude = 2 * np.abs(fft(y))
    sample_freq = fftfreq(y.size, np.abs(t[-1] - t[-2]))
    return amplitude[sample_freq > 0], sample_freq[sample_freq > 0]

def plot_wavelet(coefficients, frequencies, time, ax, title=''):
    """Plots the continuous wavelet transform coefficients."""
    im = ax.contourf(time, frequencies, np.abs(coefficients), extend='both', cmap='jet')
    cbar_ax = fig.add_axes([0.75, 0.35, 0.03, 0.45])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical").ax.tick_params(color='white')
    ax.grid()

def amplitude_to_db(amplitude: float, reference: float = 1e-6) -> float:
    """Converts amplitude from volts to decibel (dB) using a reference amplitude."""
    return 20 * np.log10(amplitude / reference)

def wavelet_filter(signal):
    """Applies hard threshold wavelet filtering to a signal."""
    coeffs = pywt.WaveletPacket(signal, wavelet='db32', mode='symmetric')
    threshold = 0.1 * max(np.max(leaf.data) for leaf in coeffs.get_level(4))
    for node in coeffs.get_level(4):
        coeffs[node.path] = apply_threshold(node.data, threshold)
    return coeffs.reconstruct()

# Paths and Database Loading
HERE = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
TRADB = os.path.join(HERE, "AE_rawdata/ae_file.tradb") #add path to .tradb file to be analyzed
PRIDB = os.path.join(HERE, "AE_rawdata/ae_file.pridb") #add path to .pridb file to be analyzed

pridb = vae.io.PriDatabase(PRIDB)
hits = pridb.read_hits()
trai = hits["trai"].to_numpy()
pridb.close()

plt.rcParams['font.family'] = 'Times New Roman'

# Signal Processing and Plotting
for q in trai:
    if q == 0:
        continue
    with vae.io.TraDatabase(TRADB) as tradb:
        sig_data, time = tradb.read_wave(q)  
        time, sig_data = time * 1000, sig_data * 1000

        # Filtering (frequency + wavelet)
        sos = signal.butter(10, [0.05, 0.7], 'bandpass', output='sos')  
        sig_data = signal.sosfilt(sos, sig_data)
        sig_data = wavelet_filter(sig_data)

        # Fourier and Morlet Wavelet Transform
        dt = time[1] - time[0]
        yf, xf = compute_fourier(sig_data, time)
        scale_range = np.power(2, np.linspace(np.log2(4), np.log2(80), 80))
        coefficients, frequencies = pywt.cwt(sig_data, scale_range, 'cmor1-1.5', dt)

        # Plotting
        if amplitude_to_db(np.max(sig_data)) > 1:
            fig = plt.figure(figsize=(8, 8))
            grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)

            ax_signal = fig.add_subplot(grid[-1, 1:])
            ax_signal.plot(time, sig_data)
            ax_signal.set_ylabel('Signal [mV]', fontsize=15)
            ax_signal.set_xlabel('Time [ms]', fontsize=15)
            ax_signal.grid()

            ax_freq = fig.add_subplot(grid[:-1, 0])
            ax_freq.plot(yf, xf)
            ax_freq.set_xlim(ax_freq.get_xlim()[::-1])
            ax_freq.set_ylim([frequencies.min(), frequencies.max()])
            ax_freq.set_ylabel('Frequency [kHz]', fontsize=15)
            ax_freq.grid()

            ax_wavelet = fig.add_subplot(grid[:-1, 1:], sharex=ax_signal)
            plot_wavelet(coefficients, frequencies, time, ax_wavelet, title="CWT Coefficients")
            ax_wavelet.set_title('Waveform Type A', fontsize=15)
            ax_wavelet.set_yticklabels([])
            plt.show()
