# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 19:30:15 2023

@author: silvia
"""
#This code was adapted and extended from the code provided in Brightspace
#BGO detector

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy import inf as INF
import os


files = [
    # filename, isotope, model, x_fill_ranges, roi, energy
    ('BGO_CS_0_GAIN0.4.Spe', '137-Cs', 'gaussian_plus_line', (440, 600), (440, 600), '661.657'),
    ('BGO_CO_0_GAIN0.4.Spe', '60-Co_1', 'gaussian_plus_line', (840, 970), (840, 970), '1332.492'),
    ('BGO_ba_0_GAIN0.4.Spe', '133-Ba_1', 'gaussian_plus_line', (40, 82), (40, 82), '356.0129'),
    ('BGO_ba_0_GAIN0.4.Spe', '133-Ba_2', 'double_gaussian_plus_line', (200, 340), (200, 340), '383.8485'),
    ('BGO_AM_0_GAIN0.4.Spe', '241-Am', 'gaussian_plus_line', (13, 75), (13, 75), '59.5409'),
]

background = 'background_BGO.Spe'


def read_spe_file(filename, background):
    def read_spe_data(file):
        initial_data = []
        start_reading = False
        for line in file:
            line = line.strip()
            if line.startswith('$DATA:'):
                start_reading = True
                continue
            if start_reading and line.isdigit():
                initial_data.append(float(line))
        return initial_data

    with open(filename) as speFile, open(background) as backgroundFile:
        spectrum_data = read_spe_data(speFile)
        background_data = read_spe_data(backgroundFile)

        # Ensure that both data arrays have the same length
        min_length = min(len(spectrum_data), len(background_data))
        spectrum_data = spectrum_data[:min_length]
        background_data = background_data[:min_length]

        # Subtract the background data from the spectrum data
        data = np.array(spectrum_data) - np.array(background_data)

    return data  


def plot_spectrum(ax, channels, counts, xlabel='Channels', ylabel='Counts', **kwargs):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1000)
    ax.grid(True)
    return ax.scatter(channels, counts, **kwargs)


def fill_graph_between_x(ax, channels, counts, x_fill, alpha=0.5, color='purple'):
    fill_mask = np.zeros(len(channels), dtype=bool)
    x_start = x_fill[0]
    x_end = x_fill[1]
    mask = (channels >= x_start) & (channels <= x_end)
    fill_mask |= mask
    ax.fill_between(channels, 0, counts, where=mask, alpha=alpha, color=color)
    return fill_mask


def gaussian(x, mu, sig, a):
    two_pi = np.pi * 2
    return (a / np.sqrt(two_pi * sig**2)) * np.exp(-0.5 * (x-mu)**2 / sig**2)


def in_interval(x, xmin=-INF, xmax=INF):
    """Boolean mask with value True for x in [xmin, xmax) ."""
    _x = np.asarray(x)
    return np.logical_and(xmin <= _x, _x < xmax)


def filter_in_interval(x, y, xmin, xmax):
    """Selects only elements of x and y where xmin <= x < xmax."""
    _mask = in_interval(x, xmin, xmax)
    return np.asarray(x)[_mask], np.asarray(y)[_mask]


def simple_model_fit(model, channels, counts, roi, **kwargs):
    """Least squares estimate of model parameters."""
    # select relevant channels & counts
    _channels, _counts = filter_in_interval(channels, counts, *roi)

    # fit the model to the data
    popt, pcov = curve_fit(model, _channels, _counts, **kwargs, maxfev=100000)
    return popt, pcov


def format_result(params, popt, pcov):
    """Display parameter best estimates and uncertainties."""
    # extract the uncertainties from the covariance matrix
    perr = np.sqrt(np.diag(pcov))

    # format parameters with best estimates and uncertainties
    # TODO: should probably round these to a sensible precision!
    _lines = (f"{p} = {o} ± {e}" for p, o, e in zip(params, popt, perr))
    return "\n".join(_lines)


def plot_model(ax, model, xrange, ps, npoints=1001, **kwargs):
    """Plots a 1d model on an Axes smoothly over xrange."""
    _channels = np.linspace(*xrange, npoints)
    _counts = model(_channels, *ps)

    return ax.plot(_channels, _counts, **kwargs)


def first_moment(x, y):
    return np.sum(x * y) / np.sum(y)


def second_moment(x, y):
    x0 = first_moment(x, y)
    return np.sum((x-x0)**2 * y) / np.sum(y)


def gaussian_initial_estimates(channels, counts):
    """Estimates of the three parameters of the gaussian distribution."""
    mu0 = first_moment(channels, counts)
    sig0 = np.sqrt(second_moment(channels, counts))
    a0 = np.sum(counts)

    return (mu0, sig0, a0)


def colourmask(x, x_start=-INF, x_end=INF, cin='red', cout='gray'):
    """Colour cin if within region of interest, cout otherwise."""
    # compute mask as integers 0 or 1
    _mask = np.array(in_interval(x, x_start, x_end), dtype=int)

    # convert to colours
    colourmap = np.array([cout, cin])
    return colourmap[_mask]


def line(x, c0, c1):
    return c0 + c1 * x


def gaussian_plus_line_components(x, mu, sig, a, c0, c1):
    components = [
        gaussian(x, mu, sig, a),
        line(x, c0, c1),
    ]
    return components


def gaussian_plus_line(x, mu, sig, a, c0, c1):
    """A gaussian on a linear background."""
    _components = gaussian_plus_line_components(x, mu, sig, a, c0, c1)
    return sum(_components)


def fit_gaussian(CHANNELS, COUNTS, p0, bounds):
    popt, pcov = curve_fit(gaussian, CHANNELS, COUNTS, p0=p0, bounds=bounds)
    return popt, pcov


def fit_double_gaussian(CHANNELS, COUNTS, p0, bounds):
    popt, pcov = curve_fit(double_gaussian, CHANNELS,
                           COUNTS, p0=p0, bounds=bounds)
    return popt, pcov


def double_gaussian(x, mu1, sig1, a1, mu2, sig2, a2):
    return gaussian(x, mu1, sig1, a1) + gaussian(x, mu2, sig2, a2)


def fit_double_gaussian_plus_line(x, mu1, sig1, a1, mu2, sig2, a2, c0, c1):
    return double_gaussian(x, mu1, sig1, a1, mu2, sig2, a2) + line(x, c0, c1)


def fit_gaussian_plus_line(x, mu, sig, a, c0, c1):
    return gaussian(x, mu, sig, a) + line(x, c0, c1)


def fit_double_gaussian_plus_line(x, mu1, sig1, a1, mu2, sig2, a2, c0, c1):
    # Fit a double Gaussian plus line
    return double_gaussian(x, mu1, sig1, a1, mu2, sig2, a2) + line(x, c0, c1)


def fit_spectrum(filename, isotope, model, x_fill_ranges, ROI, energy):
    data = read_spe_file(filename, background)
    COUNTS = np.array(list(data))
    CHANNELS = np.array(list(range(len(data))))
    fig, ax = plt.subplots(1)
    plt.ylim(0, 70000)
    plot_spectrum(ax, CHANNELS, COUNTS, marker='+')
    ax.set_title(os.path.splitext(os.path.basename(filename))[0])
    fill_mask = fill_graph_between_x(ax, CHANNELS, COUNTS, x_fill_ranges)

    GAUSSIAN_PARAMS = ('mu', 'sigma', 'a', 'c0', 'c1')


    # Make initial estimates
    _channels, _counts = filter_in_interval(CHANNELS, COUNTS, *ROI)
    _p0_gaussian = gaussian_initial_estimates(_channels, _counts)

    if model == 'gaussian_plus_line':
        # Provide initial guesses and bounds for Gaussian plus line within ROI
        popt, pcov = curve_fit(fit_gaussian_plus_line, _channels, _counts, p0=(
            *_p0_gaussian, 0, 0), bounds=([0, 0, 0, -INF, -INF], [INF, INF, INF, INF, INF]), maxfev=100000)
        mu1, sigma1, a1, c0, c1 = popt
       
    elif model == 'double_gaussian_plus_line':
        # Provide initial guesses and bounds for double Gaussian plus line within ROI
        popt, pcov = curve_fit(fit_double_gaussian_plus_line, _channels, _counts, p0=(*_p0_gaussian, *_p0_gaussian, 0, 0),
                               bounds=([0, 0, 0, 0, 0, 0, -INF, -INF], [INF, INF, INF, INF, INF, INF, INF, INF]), maxfev=100000)
        mu1, sigma1, a1, mu2, sigma2, a2, c0, c1 = popt

   
    print(filename)
    print("> the final fitted estimates:")
    print(format_result(GAUSSIAN_PARAMS, popt, pcov))

    # Create a mask to select the ROI in the full channel range
    fill_mask = fill_graph_between_x(ax, CHANNELS, COUNTS, x_fill_ranges)

    x_start = x_fill_ranges[0]
    x_end = x_fill_ranges[1]

    fig, ax = plt.subplots(1)
    colours = colourmask(CHANNELS, x_start, x_end)

    # Plot the data, showing the ROI
    plot_spectrum(ax, CHANNELS, COUNTS, c=colours, marker='+')
    plt.xlim(0, 1000)
    plt.ylim(0, 70000)
    ax.set_title(os.path.splitext(os.path.basename(filename))[0])

    # Plot the model with its parameters within the ROI
    if model == 'gaussian_plus_line':
        plot_model(ax, fit_gaussian_plus_line, (x_start, x_end), popt, c='k')
    elif model == 'double_gaussian_plus_line':
        plot_model(ax, fit_double_gaussian_plus_line,
                   (x_start, x_end), popt, c='k')

    return isotope, popt, pcov

def linear(x, m, b):
    return np.array(m) * np.array(x) + np.array(b)


def calibration_curve(linear, energies, channels):
    energies = np.array(energies, dtype=float)
    channels = np.array(channels, dtype=float)

    
    calibration_data = {
        '137-Cs': {'energy': 661.657, 'channel': 521.681},
        '60-Co_1': {'energy': 1332.492, 'channel': 907.96},
        '133-Ba_1': {'energy': 356.0129, 'channel': 57.759},
        '133-Ba_2': {'energy': 383.8485, 'channel': 275.925},
        '241-Am': {'energy': 59.5409, 'channel':  40.1738},
    }
    
    # Extract calibration data and create the calibration curve
    energies = []
    channels = []
    
    for isotope, data in calibration_data.items():
        energies.append(data['energy'])
        channels.append(data['channel'])
    
    m_initial = (energies[-1] - energies[0]) / (channels[-1] - channels[0])
    b_initial = energies[0] - m_initial * channels[0]
    initial_guess = [m_initial, b_initial]
    
    bounds = ([0, -np.inf], [np.inf, np.inf])
    popt, _ = curve_fit(linear, channels, energies, p0=initial_guess, bounds=bounds)
    m, b = popt
    
    
    fig, ax = plt.subplots()
    ax.scatter(channels, energies, label="Data")
    ax.plot(channels, linear(channels, m, b), 'black', label="Fitted Polynomial")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Energy (keV)")
    ax.legend()
    plt.title("BGO Energy vs. Channel Calibration Curve")
    plt.grid(True)
    plt.show()



def resolution_curve(energies, sigmas):
   
    resolution_data = {
        '137-Cs': {'energy': 661.657, 'sigma': 26.584},  
        '60-Co_1': {'energy': 1332.492, 'sigma': 37.217},
        '133-Ba_1': {'energy': 356.0129, 'sigma': 9.5977},
        '133-Ba_2': {'energy': 383.8485, 'sigma': 20.995},
        '241-Am': {'energy': 59.5409, 'sigma': 8.49485},  
    }
    
   
    energies = np.array([data['energy'] for data in resolution_data.values()], dtype=float)
    sigmas = np.array([data['sigma'] for data in resolution_data.values()], dtype=float)
    
   
    resolutions = (2 * sigmas * np.sqrt(2 * np.log(2)) / energies) * 100
    
    print("Resolutions:")
    print(resolutions)
    
    degree = 2  
    coefficients = np.polyfit(energies, resolutions, degree)
    fitted_polynomial = np.poly1d(coefficients)
    
    # Generate a range of energy values for the fitted curve
    energy_range = np.linspace(min(energies), max(energies), 100)
    fitted_resolutions = fitted_polynomial(energy_range)
  
    
    fig, ax = plt.subplots()
    ax.scatter(energies, resolutions, label="Data")
    ax.plot(energy_range, fitted_resolutions, 'black', label="Fitted Polynomial)")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Resolution (percentage)")
    ax.legend()
    plt.title("BGO Resolution vs. Energy (Polynomial Fitting)")
    plt.grid(True)
    plt.show()

def fit_function(log_energy, a, b, c):
        return a + b * log_energy + c * log_energy ** 2


def absolute_efficiency_curve(energies, areas, current_source_activity, source_emission_fraction):
    
    absolute_efficiency_data = {
        '137-Cs': {'energy': 661.657, 'a': 118333.74, 'current source activity': 150064.6, 'source emission fraction': 0.8499},
        '60-Co_1': {'energy': 1332.492, 'a': 1153.166, 'current source activity': 1406.5857, 'source emission fraction': 0.999826},
        '133-Ba_1': {'energy': 356.0129, 'a': 27124.61, 'current source activity': 23472.5508, 'source emission fraction': 0.6205},
        '133-Ba_2': {'energy': 383.8485, 'a': 38571.92, 'current source activity': 23472.5508, 'source emission fraction': 0.0894},
        '241-Am': {'energy': 59.5409, 'a': 1047799.46, 'current source activity': 384822.2, 'source emission fraction': 0.3578},
    }

    

    # Extract energies,current_source_activity and source_emission_fraction from the data
    energies = [data['energy'] for data in absolute_efficiency_data.values()]
    areas = [data['a'] for data in absolute_efficiency_data.values()]
    current_source_activity = [data['current source activity'] for data in absolute_efficiency_data.values()]
    source_emission_fraction = [data['source emission fraction'] for data in absolute_efficiency_data.values()]

    energies = np.array(energies, dtype=float)
    areas = np.array(areas, dtype=float)
    current_source_activity = np.array(current_source_activity, dtype=float)
    source_emission_fraction = np.array(source_emission_fraction, dtype=float)

    absolute_efficiency = np.array(areas) / (np.array(current_source_activity) * np.array(source_emission_fraction))

    print("Absolute efficiencies:")
    print(absolute_efficiency)

    log_energies = np.log(energies)
    log_efficiency = np.log(absolute_efficiency)

    popt, _ = curve_fit(fit_function, log_energies, log_efficiency, p0=(0, 0, 0))
    a, b, c = popt
    
    # Create a range of energies for the best fit curve
    energy_range = np.linspace(min(energies), max(energies), 100)
    log_energy_range = np.log(energy_range)
    best_fit_curve = a + b * log_energy_range + c * log_energy_range ** 2

    
    fig, ax = plt.subplots()
    ax.scatter(energies, absolute_efficiency, label="Data")
    ax.plot(energy_range, np.exp(best_fit_curve), label="Fitted Low-Order Polynomial", color='black')
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Efficiency")
    ax.legend()
    plt.title("BGO Efficiency vs. Energy (log-log)")
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.grid(True)
    plt.show()

    return absolute_efficiency
