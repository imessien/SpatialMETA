from ._plotting import (
    showionimage,
    plot_spot_sm_st,
    plot_newdot_sm_st,
    plot_marker_gene_metabolite,
    plot_markerfeature,
    plot_corrcoef_stsm_inall,
    plot_corrcoef_stsm_ingroup,
    plot_spatial_deconvolution,
    plot_gene_corrcoef_sm_ingroup,
    plot_metabolite_corrcoef_st_ingroup,
    plot_volcano_corrcoef_gene,
    plot_volcano_corrcoef_metabolite,
    plot_trajectory_with_arrows,
    plot_clustermap_with_smoothing,
    plot_features_trajectory,
    plot_network,
    make_colormap,
    create_fig,
    create_subplots,
)

from .core.wrapper import (
    Wrapper
)

from matplotlib import font_manager
import os
import matplotlib.pyplot as plt

font_files = font_manager.findSystemFonts(fontpaths=['./spatialtk/plotting/fonts/'])
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)


os.system('export LC_CTYPE="en_US.UTF-8"')
os.environ["LC_CTYPE"] = "en_US.UTF-8"

plt.rcParams['font.family'] = 'arial'
plt.rcParams['font.size'] = 12