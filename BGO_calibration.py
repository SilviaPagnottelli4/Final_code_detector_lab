# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:17:42 2023

@author: silvia
"""

#BGO detector

from BGO_guide_functions import fit_spectrum, calibration_curve, linear, resolution_curve, absolute_efficiency_curve
import matplotlib.pyplot as plt


files = [
    # filename, isotope, model, x_fill_ranges, roi, energy
    ('BGO_CS_0_GAIN0.4.Spe', '137-Cs', 'gaussian_plus_line', (440, 600), (440, 600), '661.657'),
    ('BGO_CO_0_GAIN0.4.Spe', '60-Co_1', 'gaussian_plus_line', (840, 970), (840, 970), '1332.492'),
    ('BGO_ba_0_GAIN0.4.Spe', '133-Ba_1', 'gaussian_plus_line', (40, 82), (40, 82), '356.0129'),
    ('BGO_ba_0_GAIN0.4.Spe', '133-Ba_2', 'double_gaussian_plus_line', (200, 340), (200, 340), '383.8485'),
    ('BGO_AM_0_GAIN0.4.Spe', '241-Am', 'gaussian_plus_line', (13, 75), (13, 75), '59.5409'),
]


for filename, isotope, model, x_fill_ranges, ROI, energy in files:
    fit_spectrum(filename, isotope, model, x_fill_ranges, ROI, energy)
   
    

calibration_data = {
    '137-Cs': {'energy': 661.657, 'channel': 521.681},
    '60-Co_1': {'energy': 1332.492, 'channel': 907.96},
    '133-Ba_1': {'energy': 356.0129, 'channel': 57.759},
    '133-Ba_2': {'energy': 383.8485, 'channel': 275.925},
    '241-Am': {'energy': 59.5409, 'channel':  40.1738},
}

energies = []
channels = []


calibration_curve(linear, energies, channels)


resolution_data = {
    '137-Cs': {'energy': 661.657, 'sigma': 26.584},  
    '60-Co_1': {'energy': 1332.492, 'sigma': 37.217},
    '133-Ba_1': {'energy': 356.0129, 'sigma': 9.5977},
    '133-Ba_2': {'energy': 383.8485, 'sigma': 20.995},
    '241-Am': {'energy': 59.5409, 'sigma': 8.49485},  
}


sigmas = []

resolution_curve(energies, sigmas)

absolute_efficiency_data = {
    '137-Cs': {'energy': 661.657, 'a': 118333.74, 'current source activity': 150064.6, 'source emission fraction': 0.8499},
    '60-Co_1': {'energy': 1332.492, 'a': 1153.166, 'current source activity': 1406.5857, 'source emission fraction': 0.999826},
    '133-Ba_1': {'energy': 356.0129, 'a': 27124.61, 'current source activity': 23472.5508, 'source emission fraction': 0.6205},
    '133-Ba_2': {'energy': 383.8485, 'a': 38571.92, 'current source activity': 23472.5508, 'source emission fraction': 0.0894},
    '241-Am': {'energy': 59.5409, 'a': 1047799.46, 'current source activity': 384822.2, 'source emission fraction': 0.3578},
}

areas = []
current_source_activity = []
source_emission_fraction = []

absolute_efficiency_curve(energies, areas, current_source_activity, source_emission_fraction)

plt.show()

