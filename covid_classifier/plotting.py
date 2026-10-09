"""
shared matplotlib styling so figures look consistent
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'

# categorical slots in fixed order: one per class (normal, viral, covid)
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']

# cingle-hue sequential ramp (light to dark) for magnitudes such as confusion matrix
BLUES = LinearSegmentedColormap.from_list('blues', [
    '#f4f8fd', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])

plt.rcParams.update({
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'font.family': ['Segoe UI', 'DejaVu Sans', 'sans-serif'],
    'font.size': 10,
    'text.color': INK,
    'axes.labelcolor': INK_SECONDARY,
    'axes.titlecolor': INK,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.edgecolor': AXIS,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': GRID,
    'grid.linewidth': 0.8,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'xtick.labelcolor': INK_SECONDARY,
    'ytick.labelcolor': INK_SECONDARY,
    'legend.frameon': False,
    'lines.linewidth': 2,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})
