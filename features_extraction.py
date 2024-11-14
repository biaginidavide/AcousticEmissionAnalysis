# -*- coding: utf-8 -*-
"""
Acoustic Emission Analysis using Vallena AE
Analyzes .tradb and .pridb files to extract acoustic emission features.
Requires Vallena AE library: https://pyvallenae.readthedocs.io/en/stable/

"""

import os
import numpy as np
import vallenae as vae
from scipy.fft import fft, fftfreq
from scipy import signal
import pywt
import matplotlib.pyplot as plt

def find_nearest(array, value):
    """Find index of the element in array closest to a given value."""
    return (np.abs(array - value)).argmin()

def frequency_features(spectrum, freqs):
    """Extract frequency features from Fourier spectrum."""
    peak_freq = freqs[np.argmax(spectrum)]
    return peak_freq

def fourier_transform(y, t):
    """Calculate the Fourier transform and frequency bins for signal y over time t."""
    n = y.size
    amplitudes = 2 * np.abs(fft(y))[:n // 2]
    freqs = fftfreq(n, d=t[1] - t[0])[:n // 2]
    return amplitudes, freqs

def threshold_filter(data, threshold):
    """Apply threshold filter to remove small values in data."""
    return np.array([0 if abs(x) < threshold else x for x in data])

def wavelet_energy(coefficients):
    """Calculate the energy of wavelet coefficients."""
    return np.sum(np.square(coefficients))

# File paths for raw data
BASE_DIR = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
PRIDB_PATH = os.path.join(BASE_DIR, "AE_rawdata/compteflon.pridb")
TRADB_PATH = os.path.join(BASE_DIR, "AE_rawdata/compteflon.tradb")

# Load and parse data
pridb = vae.io.PriDatabase(PRIDB_PATH)
hits = pridb.read_hits()
params = pridb.read_parametric()
arrival_times = hits["time"].to_numpy()
trai_values = hits["trai"].to_numpy()
pridb.close()

# Filter data based on channel and time range
channel_hits = [trai_values[i] for i in range(len(arrival_times)) if 600 < arrival_times[i] < 650 and trai_values[i] != 0]

# Initialize lists for acoustic parameters
energies = [[] for _ in range(8)]
peak_freqs = []
amplitudes, counts, rise_times, total_energies = [], [], [], []

for trai in channel_hits:
    with vae.io.TraDatabase(TRADB_PATH) as tradb:
        dt = 1000 / 2048
        sr = 2000000
        y, t = tradb.read_wave(trai)
        
        # Bandpass filter
        sos = signal.butter(10, [0.05, 0.99], 'bandpass', output='sos')
        y_filtered = signal.sosfilt(sos, y)
        
        # Wavelet packet decomposition
        wp = pywt.WaveletPacket(y_filtered, wavelet='db32', mode='symmetric', maxlevel=3)
        components = ['aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd']
        threshold = 0.1 * max([np.max(wp[node].data) for node in components])
        
        #Threshold and calculate energy for each component
        for i, comp in enumerate(components):
            filtered_data = threshold_filter(wp[comp].data, threshold)
            wp[comp].data = filtered_data
            energies[i].append(wavelet_energy(filtered_data))
        
        # Reconstruct filtered signal
        reconstructed_signal = wp.reconstruct()
        
        # Time-domain features
        total_energies.append(vae.features.acoustic_emission.energy(reconstructed_signal, sr))
        peak_amplitude = vae.features.acoustic_emission.peak_amplitude(reconstructed_signal)
        rise_times.append(vae.features.acoustic_emission.rise_time(reconstructed_signal, 0.1 * peak_amplitude, sr))
        counts.append(vae.features.acoustic_emission.counts(reconstructed_signal, 0.1 * peak_amplitude))
        amplitudes.append(vae.features.conversion.amplitude_to_db(peak_amplitude))
        
        # Frequency-domain features
        yf, xf = fourier_transform(reconstructed_signal, t)
        peak_freq = frequency_features(yf, xf)
        peak_freqs.append(peak_freq)


# Save extracted parameters to CSV files
for i, data in enumerate([peak_freqs, rise_times, counts, amplitudes, total_energies]):
    np.savetxt(f'Parameter_{i + 1}.csv', data, delimiter=',')

for i, energy in enumerate(energies):
    np.savetxt(f'Wavelet_Energy_Component_{i + 1}.csv', energy, delimiter=',')
