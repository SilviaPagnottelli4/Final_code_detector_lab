# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:17:42 2023

@author: silvia
"""
#NaI detector

from NaI_guide_functions import fit_spectrum, calibration_curve, linear, resolution_curve, absolute_efficiency_curve
import matplotlib.pyplot as plt


files = [
    # filename, isotope, model, x_fill_ranges, roi, energy
    ('NaI_Cs_0Deg_Gain1.Spe', '137-Cs', 'gaussian_plus_line', (225, 400), (225, 400), '661.657'),
    ('NaI_Ba_0deg_gain1.Spe', '133-Ba_1', 'double_gaussian_plus_line', (10, 22), (10, 22), '383.8485'),
    ('NaI_Ba_0deg_gain1.Spe', '133-Ba_2', 'gaussian_plus_line', (23, 50), (23, 50), '302.85'),
    ('NaI_Ba_0deg_gain1.Spe', '133-Ba_3', 'double_gaussian_plus_line', (100, 190), (100, 190), '356.0129'),
    ('NaI_Am_0Deg_Gain1.Spe', '241-Am', 'gaussian_plus_line', (18, 41), (18, 41), '59.5409'),
]



for filename, isotope, model, x_fill_ranges, ROI, energy in files:
    fit_spectrum(filename, isotope, model, x_fill_ranges, ROI, energy)
   
    

calibration_data = {
    '137-Cs': {'energy': 661.657, 'channel': 275.443},
    '133-Ba_1': {'energy': 356.0129, 'channel': 14.6061},
    '133-Ba_2': {'energy': 302.85, 'channel': 37.200},
    '133-Ba_3': {'energy': 383.8485, 'channel': 125.910},
    '241-Am': {'energy': 59.5409, 'channel': 27.984},
}

energies = []
channels = []


calibration_curve(linear, energies, channels)


resolution_data = {
    '137-Cs': {'energy': 661.657, 'sigma': 15.57},  
    '133-Ba_1': {'energy': 356.0129, 'sigma': 2.49},
    '133-Ba_2': {'energy': 302.85, 'sigma': 4.125},
    '133-Ba_3': {'energy': 383.8485, 'sigma': 9.5905},
    '241-Am': {'energy': 59.5409, 'sigma': 3.432},  
}


sigmas = []

resolution_curve(energies, sigmas)

absolute_efficiency_data = {
    '137-Cs': {'energy': 661.657, 'a': 55410.34, 'current source activity': 150064.6, 'source emission fraction': 0.8499},
    '133-Ba_1': {'energy': 356.0129, 'a': 35765.323, 'current source activity': 23472.5508, 'source emission fraction': 0.6205},
    '133-Ba_2': {'energy': 302.85, 'a': 12719.164, 'current source activity': 23472.5508, 'source emission fraction': 0.1834},
    '133-Ba_3': {'energy': 383.8485, 'a': 5176.66, 'current source activity': 23472.5508, 'source emission fraction': 0.0894},
    '241-Am': {'energy': 59.5409, 'a': 154561.94, 'current source activity': 384822.2, 'source emission fraction': 0.3578},
}

areas = []
current_source_activity = []
source_emission_fraction = []

absolute_efficiency_curve(energies, areas, current_source_activity, source_emission_fraction)

plt.show()

