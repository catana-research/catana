# -*- coding: utf-8 -*-
"""
Altair plotting
---------------

Provides wrapper functionality to easily utilise altair plots.

In addition to configuring the plots via the provided keyword arguments, the underlying Altair/Vega-Lite chart instance
can be customised via `**kwargs`. Full documentation of these options can be found in the `Altair documentation <
https://altair-viz.github.io/user_guide/generated/toplevel/altair.Chart.html#>`_.

"""
import altair as alt


def _customise(chart, interactive=False):
    if interactive:
        chart = chart.interactive()
    return chart

# TODO: Factorise documentation like seaborn? https://github.com/mwaskom/seaborn/blob/09e7d61eaeed932c743976c877df2edd321a6c05/seaborn/categorical.py#L2091
# TODO: Perhaps implement an establish variables like seaborn: https://github.com/mwaskom/seaborn/blob/09e7d61eaeed932c743976c877df2edd321a6c05/seaborn/categorical.py#L31
def scatter(data=None, x=None, y=None, color=None, interactive=False):
    """Create a scatter plot.

    Parameters
    ----------
    x, y, color : names of variables in ``data``
        Inputs for plotting long-form data. See examples for interpretation.
    data : pandas.DataFrame
        Dataset for plotting. If ``x`` and ``y`` are absent, this is
        interpreted as wide-form. Otherwise it is expected to be long-form.
    **kwargs :
        Additional keywords will be used to customise the underlying Altair/Vega-Lite plot instance.
    """
    chart = alt.Chart(data).mark_circle(size=10).encode(
        x=f'{x}:Q',
        y=f'{y}:Q',
        color=f'{color}:N',
        #tooltip=['charge', 'm']
    )

    return _customise(chart, interactive=interactive)

def line(data, x, y, color=None, line_width=2):

    chart = alt.Chart(data).mark_line(size=line_width).encode(
        x=f'{x}',
        y=f'{y}',
        color=color,
        )
    return _customise(chart, interactive=False)

def histogram(data, x, bins=30, color=None):
    chart = alt.Chart(data).mark_bar().encode(
        x=f'{x}:Q',
        y=f'{y}:Q',
        color=f'{color}:N',
        #tooltip=['charge', 'm']
    )#.interactive()
    return chart


def histogram2d(data, x, bins=30, color=None):
    return chart

def table(data, columns, labels=None, max_rows=10):

    labels = labels if labels else columns  # TODO: Write function to strip ':' if present

    # Base chart for data tables
    table_column = alt.Chart(data).mark_text().encode(
        y=alt.Y('row_number:O', axis=None)
    ).transform_window(
        row_number='row_number()'
    ).transform_window(
        rank='rank(row_number)'
    ).transform_filter(
        alt.datum.rank <= max_rows
    )

    table_columns = [table_column.encode(text=column).properties(title=label) for column, label in zip(columns, labels)]
    table = alt.hconcat(*table_columns)
    return table


def facet(data, rows, columns, color=None, width=200, height=200):
    # TODO: make the type of chart configurable
    chart = alt.Chart(data).mark_circle().encode(
        alt.X(alt.repeat("column"), type='quantitative'),
        alt.Y(alt.repeat("row"), type='quantitative'),
        color=color
    ).properties(
        width=width,
        height=height
    ).repeat(  # <<< Specify rows and columns to repeat
        row=rows,
        column=columns
    ).interactive()
    return chart

def legend2d(data, color=None):
    legend = alt.Chart(data).mark_rect().encode(
        y=alt.Y('eta_bin:N', axis=alt.Axis(orient='right')),
        x='name:O',
        color=color
    )
    return legend
