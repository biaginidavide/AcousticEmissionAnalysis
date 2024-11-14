Acoustic emission analysis:

This python tool allows to perform different types of analysis of acoustic signals.
It is intended to analyze signals collected using Vallen system http://www.vallen.de/
It can be used in combination with the dataset: 
see the reference paper at: 

All scripts make use of vallene ae tool to open and extract acoustic data (see more details at https://pypi.org/project/vallenae/)
Before running the scripts, store the folder 'valleneae' in local Python libraries

Content: 
 
features_extraction.py - extraction af AE rise time, counts, max amplitude, peak frequency, centroid frequency, weighted peak frequency
                         partial powers, wavelet packet components
                         for more info regarding AE features and their meaning refer to Saeedifar M, Zarouchas D. Damage characterization of laminated composites using acoustic emission: A review. Vol. 195, Composites Part B: Engineering. Elsevier Ltd; 2020. 

morlet_wavelet.py -      calculation and plot of Morlet continuum wavelet transform for each waveform 

bvalue_sentryf.py - evaluation of bvalue and sentry function 

waveletcumratios.py - script to calculate wavelet packet cumulative ratios 
