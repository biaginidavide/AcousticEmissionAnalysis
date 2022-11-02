Acoustic emission analysis:

This pithon tool allows to perform different types of analysis of acoustic signals.
It is intended to analyse signals collected using Vallen system http://www.vallen.de/
It can be used in combination with the dataset: 
see the refernce paper at: 

All scripts make use of vallene ae tool to open and extract acoustic data (see more details at https://pypi.org/project/vallenae/)
The sripts run on python version 3.8.3
The suggested scripting tool is Spyder (https://www.spyder-ide.org/)

Before run the scripts, store the folder 'valleneae' in local python libraries

Content: 
 
features_extraction.py - extraction af AE rise time, counts, max amplitude, peak frequency, centroid frequency, weighted peak frequency
                         partial powers, wavelet packet components
                         for more info regarding AE features and their meaning refer to Saeedifar M, Zarouchas D. Damage characterization of laminated composites using acoustic emission: A review. Vol. 195, Composites Part B: Engineering. Elsevier Ltd; 2020. 

morlet_wavelet.py -      calculation and plot of Morlet continuum wavelet transform for each waveform 

bvalue_sentryf.py - evaluation of bvalue and sentry function 

waveletcumratios.py - script to calculate wavelet packet cumulative ratios 