# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:15:19 2024

@author: davidebiagini
"""

import os
import numpy as np
import pandas as pd

# Set directory path for wavelet component files
path1 = "sp_2/ch2/" #Path to the csv files hosting the Wavelet packet components (derived in featurs_extractions.y)

# Load wavelet component data files into a list of numpy arrays
wavelet_files = [f"WLCOMP{i}_E.csv" for i in range(1, 9)]
wavelet_data = [np.asarray(pd.read_csv(os.path.join(path1, f))) for f in wavelet_files]

# Compute total wavelet component energy (WLTOT)
WLTOT = sum(wavelet_data)

# Calculate cumulative ratios for each wavelet component
cumulative_ratios = [np.cumsum(WL) / np.cumsum(WLTOT) for WL in wavelet_data]

# Save each cumulative ratio as a separate CSV file
for i, ratio in enumerate(cumulative_ratios, start=1):
    np.savetxt(f'r{i}.csv', ratio, delimiter=',')
