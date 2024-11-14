Acoustic emission analysis:

This Python tool allows to perform different types of analysis of acoustic signals.
It is intended to analyze signals collected using Vallen system http://www.vallen.de/
It can be used in combination with the dataset: https://research.tudelft.nl/en/datasets/acoustic-emission-monitoring-in-cfrp-compression-tests
see the reference paper at: https://journals.sagepub.com/doi/full/10.1177/00219983231163853

All scripts make use of the Vallene tool to open and extract acoustic data (see more details at https://pypi.org/project/vallenae/)

Content: 
 
features_extraction.py - extraction of AE features from .pridb and .tradb files
                         for more info regarding AE features and their meaning refer to Saeedifar M, Zarouchas D. Damage characterization of laminated composites using acoustic emission: A review. Vol. 195, Composites Part B: Engineering. Elsevier Ltd; 2020. 

morlet_wavelet.py -      calculation and plot of Morlet continuum wavelet transform for each waveform 

bvalue_sentryf.py - evaluation of b-value and sentry function 

wavelet_cumulative_ratios.py - script to calculate wavelet packet cumulative ratios 
