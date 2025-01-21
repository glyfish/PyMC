import numpy
from matplotlib import pyplot

from lib.utils import get_param_default_if_missing
from lib.plots import comp
from scipy.stats import gaussian_kde


def ridge_plot(cats, raw_data, **kwargs):
    """
    Ridge plot for dist over categories cats.

    Parameters
    ----------
    cats : numpy.ndarray[float]
        Categories for ridge plot.
    raw_data : numpy.ndarray[float]
        Data to plot.
    nbins : int
        Number of bins for histogram.
    title : string, optional
        Plot title (default is None)
    title_offset : float (default is 0.0)
        Plot title off set from top of plot.
    xlabel : string, optional
        Plot x-axis label (default is 'x')
    ylabel : string, optional
        Plot y-axis label (default is 'y')
    xlim : (float, float)
        Specify the limits for the x axis. (default None)
    ylim : (float, float)
        Specify the limits for the y axis. (default None)
    figsize : (int, int)
        Figure size.
    """

    figsize = get_param_default_if_missing("figsize", (10, 8), **kwargs)
    nbins = get_param_default_if_missing("nbins", 10, **kwargs)
    xlabel = get_param_default_if_missing("xlabel", None, **kwargs)  
    title = get_param_default_if_missing("title", None, **kwargs)  
    title_offset = get_param_default_if_missing("title_offset", 0.0, **kwargs)

    nplot = len(cats)
    _, axis = pyplot.subplots(nplot, sharex=True, figsize=figsize)

    max_data = numpy.max([numpy.max(raw_data[i]) for i in range(nplot)])
    min_data = numpy.min([numpy.min(raw_data[i]) for i in range(nplot)]) 

    axis[0].set_title(title, y=1.0 + title_offset)

    for i in range(nplot):
        kde = gaussian_kde(raw_data[i])
        xkde = numpy.linspace(min(raw_data[i]), max(raw_data[i]), 100)
        hist, bins = numpy.histogram(raw_data[i], bins=nbins, range=(min_data, max_data), density=True)
        hist = hist * (bins[-1] - bins[0]) / nbins
        comp.bar_comparison(axis[i], hist, kde(xkde), bins[:-1], xkde, color='blue', alpha=0.5, xlabel=None)
        axis[i].set_yticklabels([])
        axis[i].set_ylabel(cats[i], rotation=0)

    axis[nplot-1].set_xlabel(xlabel)