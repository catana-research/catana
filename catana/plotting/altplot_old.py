# -*- coding: utf-8 -*-
"""
Altair plotting
---------------

Provides wrapper functionality to easily utilise altair plots.
"""
import altair as alt


class AltPlot(object):
    """

    """

    def __init__(self, save_dir=None, context=''):
        self.context = context
        self._figs = {}
        self._axs = {}
        self._save_dir = save_dir
        pass

    def point(self, df, param, group=None, kwargs=None):

        """
        fixed recipes (histogram ratio, facet grid), configurable API

        Layers, transforms, filters, selections, facet, trellis
        hconcat, vconcat, link selections

        use partial/decorators to evaluate?

        point, bar, box

        aggregations: count(*)

        :param df:
        :param param:
        :param group:
        :param kwargs:
        :return:
        """
        if kwargs:
            chart_kwargs = kwargs.get('chart', {})
            selection_kwargs = kwargs.get('selection', {'type': 'interval'})

        selections = alt.selection(**selection_kwargs)
        #selections = []
        transform = None

        if group:
            color = alt.condition(selections, f'{group}:N', alt.value('lightgray'))
        else:
            color = None

        # Make a point chart
        chart = alt.Chart(data=df, **chart_kwargs).mark_point().encode(
            x=f'{param[0]}:Q',
            y=f'{param[1]}:Q',
            color=color
        ).add_selection(selections).facet(columns=f'{group}:N')
        # if brush:
        #     chart.add_selection(brush)
        if transform:
            chart = chart.add_transform(transform)
        chart.serve()  # This will open the plot in a web browser, not required in jupyter lab




    def point(self, df, param, group=None, kwargs=None):

        """
        fixed recipes (histogram ratio, facet grid), configurable API

        Layers, transforms, filters, selections, facet, trellis
        hconcat, vconcat, link selections

        use partial/decorators to evaluate?

        point, bar, box

        aggregations: count(*)

        :param df:
        :param param:
        :param group:
        :param kwargs:
        :return:
        """
        if kwargs:
            chart_kwargs = kwargs.get('chart', {})
            selection_kwargs = kwargs.get('selection', {'type': 'interval'})

        selections = alt.selection(**selection_kwargs)
        #selections = []
        transform = None

        if group:
            color = alt.condition(selections, f'{group}:N', alt.value('lightgray'))
        else:
            color = None

        # Make a point chart
        chart = alt.Chart(data=df, **chart_kwargs).mark_point().encode(
            x=f'{param[0]}:Q',
            y=f'{param[1]}:Q',
            color=color
        ).add_selection(selections).facet(columns=f'{group}:N')
        # if brush:
        #     chart.add_selection(brush)
        if transform:
            chart = chart.add_transform(transform)
        chart.serve()  # This will open the plot in a web browser, not required in jupyter lab


class AltPlot(object):
    """

    """

    def __init__(self, save_dir=None, context=''):
        self.context = context
        self._figs = {}
        self._axs = {}
        self._save_dir = save_dir
        pass



def main():
    # https://altair-viz.github.io/user_guide/generated/toplevel/altair.Chart.html#altair.Chart
    from vega_datasets import data
    df = data.iris()

    import altair as alt

    chart = AltChart(size=(2, 2)).scatter(x='sepalLength', y='sepalWidth', color='species',
                                          pos=(0, 0))
    chart.serve()

    scatter(df, 'sepalLength', 'sepalWidth', color='species').serve()

    return




    source = data.cars()

    brush = alt.selection(type='interval')

    points = alt.Chart(source).mark_point().encode(
        x='Horsepower:Q',
        y='Miles_per_Gallon:Q',
        color=alt.condition(brush, 'Origin:N', alt.value('lightgray'))
    ).add_selection(
        brush
    )

    bars = alt.Chart(source).mark_bar().encode(
        y='Origin:N',
        color='Origin:N',
        x='count(Origin):Q'
    ).transform_filter(
        brush
    )

    a =  points & bars
    a.serve()

    plot = AltPlot()
    #plot.point(df, ['sepalLength', 'sepalWidth'], group='species', kwargs={'chart': {'title': 'test'}})

if __name__ == '__main__':
    main()
