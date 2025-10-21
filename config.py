"""
Configuration settings for the Width vs Depth visualization application
"""

import multiprocessing as mp

# Platform-independent multiprocessing setup
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# PLOTLY THEME SETTINGS
PLOTLY_TEMPLATE = "plotly_white"  # or "plotly", "seaborn", "ggplot2"

PLOTLY_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'width_vs_depth_plot',
        'height': 1000,
        'width': 1400,
        'scale': 2
    }
}

# Plotly qualitative color palette
QUAL_PLOTLY = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
]
