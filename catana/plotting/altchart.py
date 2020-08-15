"""
AltChart
--------

API wrapper to utilise template altair plots.

Data types:

    Data Type	Shorthand Code	Description
    quantitative	         Q	a continuous real-valued quantity
    ordinal 	             O	a discrete ordered quantity
    nominal	                 N	a discrete unordered category
    temporal	             T	a time or date value

If types are not specified for data input as a DataFrame, Altair defaults to quantitative for any numeric data, temporal
for date/time data, and nominal for string data, but be aware that these defaults are by no means always the correct choice!

https://altair-viz.github.io/user_guide/encoding.html#encoding-data-types
"""
import altair as alt
from copy import copy
import numpy as np

from catana.plotting import altplot
from catana.plotting import constants


class AltChart(object):
    """Base chart object

    """
    def __init__(self, data=None, size=None, meta=''):
        self.data = data
        self._base_plot = alt.Chart(self.data).mark_point()
        self.plot_grid = np.full(size, self._base_plot) if size else [[self._base_plot]]
        self._selections = []
        self.meta = meta

    def _combine_plots(self):
        row_plots = []
        for row in self.plot_grid:
            row_plots.append(alt.hconcat(*row))
        joined_plots = alt.vconcat(*row_plots)

        return joined_plots

    def _plot_grid_cell(self, obj, pos, plot):
        obj.plot_grid[pos[0], pos[1]] = plot
        # TODO: If plot is called on the same cell multiple times, store in an array and layer on rendering: alt.layer(eta_bars, eta_lines)
        # grid_cell = obj.plot_grid[pos[0], pos[1]]
        # if grid_cell == self._base_plot:
        #     obj.plot_grid[pos[0], pos[1]] = plot
        # else:
        #     obj.plot_grid[pos[0], pos[1]].append(plot)

    def _add_plot(self, plot_function, **kwargs):
        """Add a plot to the chart."""
        # Process kwargs
        kwargs['data'] = kwargs.get('data', self.data)
        pos = kwargs.pop('pos') if 'pos' in kwargs.keys() else (0, 0)  # TODO: Remove magic (0, 0)
        brush = kwargs.pop('brush') if 'brush' in kwargs.keys() else None
        selection = kwargs.pop('selection') if 'selection' in kwargs.keys() else None
        meta = kwargs.pop('meta') if 'meta' in kwargs.keys() else None  # TODO: Store this with the plot object or a tooltip

        if brush:
            kwargs['interactive'] = False

        obj = copy(self)
        plot = plot_function(**kwargs)
        if selection:
            plot = self._add_selection(plot, selection)
        self._plot_grid_cell(obj, pos, plot)  # TODO: This needs to be handled more elegantly by a chart class
        return obj

    # TODO: Apply selection on Tables before other selections are applied or apply max_row selection at end
    def _add_selection(self, plot, selection):
        # TODO: MAKE THIS WORK FOR EVERY PLOT TYPE (INCLUDING TABLE, FACET)
        if str(type(plot)).find('RepeatChart') > -1:
            plot = plot.add_selection(selection).transform_filter(selection)
        elif str(type(plot)).find('ConcatChart') > -1:
            plot = plot.add_selection(selection).transform_filter(selection)
        else:
            background = plot.encode(opacity=alt.value(constants.DESELECTED_OPACITY)).add_selection(selection)
            highlight = plot.transform_filter(selection)
            plot = alt.layer(
                background,
                highlight,
            )
            # else:
            #     plot = plot.add_selection(selection)
        self._selections.append(selection)

        return plot

    # def add_plot(self, plot):
    #     """Add one or more selections to the chart."""
    #     if not plot:
    #         return self
    #     copy = self.copy(deep=['selection'])
    #     if copy.selection is alt.api.Undefined:
    #         copy.selection = {}
    #
    #     for s in selections:
    #         copy.selection[s.name] = s.selection
    #     return copy

    """Alternatively we can setup link seaborn:
    https://github.com/mwaskom/seaborn/blob/09e7d61eaeed932c743976c877df2edd321a6c05/seaborn/categorical.py#L2221

    1. Define a base class for each plot type
    2. explicitly define all the args in the function definition

    def boxplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None,
            orient=None, color=None, palette=None, saturation=.75,
            width=.8, dodge=True, fliersize=5, linewidth=None,
            whis=1.5, notch=False, ax=None, **kwargs):

    plotter = _BoxPlotter(x, y, hue, data, order, hue_order,
                          orient, color, palette, saturation,
                          width, dodge, fliersize, linewidth)

    if ax is None:
        ax = plt.gca()
    kwargs.update(dict(whis=whis, notch=notch))

    plotter.plot(ax, kwargs)
    return ax

    """

    def plot(self, kind='scatter', **kwargs):
        """Create a chart of the specified type, defaults to scatter plot.

        Args:
            kind:
            **kwargs:

        Returns:

        """
        plot_function = getattr(self, kind)
        return self._add_plot(plot_function, **kwargs)

    @alt.utils.use_signature(altplot.scatter)
    def scatter(self, **kwargs):
        return self._add_plot(altplot.scatter, **kwargs)

    @alt.utils.use_signature(altplot.line)
    def line(self, **kwargs):
        return self._add_plot(altplot.line, **kwargs)

    @alt.utils.use_signature(altplot.histogram)
    def histogram(self, **kwargs):
        return self._add_plot(altplot.histogram, **kwargs)

    @alt.utils.use_signature(altplot.histogram2d)
    def histogram2d(self, **kwargs):
        return self._add_plot(altplot.histogram2d, **kwargs)

    @alt.utils.use_signature(altplot.pie)
    def pie(self, **kwargs):
        return self._add_plot(altplot.pie, **kwargs)

    @alt.utils.use_signature(altplot.table)
    def table(self, **kwargs):
        return self._add_plot(altplot.table, **kwargs)

    @alt.utils.use_signature(altplot.pairgrid)
    def pairgrid(self, **kwargs):
        return self._add_plot(altplot.pairgrid, **kwargs)

    @alt.utils.use_signature(altplot.facet)
    def facet(self, **kwargs):
        return self._add_plot(altplot.facet, **kwargs)

    @alt.utils.use_signature(alt.api.TopLevelMixin.serve)
    def serve(self, **kwargs):
        self._combine_plots().serve(**kwargs)
        return

    @alt.utils.use_signature(alt.api.TopLevelMixin.display)
    def display(self, **kwargs):
        self._combine_plots().display(**kwargs)
        return
