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
import numpy as np


def _customise(chart, interactive=False, title=None, color=None):
    if interactive:
        chart = chart.interactive()
    if title:
        chart = chart.properties(title=title)
    if color:
        chart = chart.encode(color=color)

    return chart

# TODO: Factorise documentation like seaborn? https://github.com/mwaskom/seaborn/blob/09e7d61eaeed932c743976c877df2edd321a6c05/seaborn/categorical.py#L2091
# TODO: Perhaps implement an establish variables like seaborn: https://github.com/mwaskom/seaborn/blob/09e7d61eaeed932c743976c877df2edd321a6c05/seaborn/categorical.py#L31
def scatter(data=None, x=None, y=None, color=None, title=None, interactive=False):
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
        x=x,
        y=y,
        color=color,
    )

    return _customise(chart, interactive=interactive, title=title)


def line(data, x, y, color=None, line_width=2, title=None, interactive=False, ci_y=None):

    kwargs = {}
    if color:
        kwargs['color'] = color

    chart = alt.Chart(data).mark_line(size=line_width).encode(
        x=x,
        y=y,
        **kwargs
        )
    if ci_y:
        ci = alt.Chart(data).mark_area(opacity=0.3).encode(
            x=x,
            y='ci0({})'.format(ci_y),  # ci0 - Lower CI
            y2='ci1({})'.format(ci_y),  # ci1 - Upper CI
            **kwargs
        )
        chart = alt.layer(ci, chart)

    return _customise(chart, interactive=interactive, title=title)


def histogram(data, x, bins=30, color=None, title=None, interactive=False, stack=True, fill=True, alpha=1):
    bins = alt.Bin(maxbins=bins)
    if fill:
        chart = alt.Chart(data).mark_bar(opacity=alpha).encode(
            x=alt.X(x, bin=bins),
            y=alt.Y('count()', stack=stack),
            )
    else:
        chart = alt.Chart(data).mark_line(interpolate='step-after', opacity=alpha).encode(
            x=alt.X(x, bin=bins),
            y=alt.Y('count()', stack=stack),
        )

    return _customise(chart, interactive=interactive, color=color, title=title)


def histogram2d(data=None, x=None, y=None, z=None, bins=30, color=None, title=None, interactive=False):
    bins = alt.Bin(maxbins=bins)
    chart = alt.Chart(data).mark_rect().encode(
        x=x,
        y=y,
        color=z,
        )

    return _customise(chart, interactive=interactive, color=color, title=title)

def pie(data=None, column=None, color=None):
    """
    Reference:
    https://github.com/vega/vega-lite/issues/408#issuecomment-373870307

    """
    r = 90
    df = data.groupby([color]).count()

    labels = df.index
    values = df[column].values / df[column].sum() * 360.

    piechart = {}
    current_angle = 0
    for label, value in zip(labels, values):
        piechart.update({label: (current_angle, current_angle + value)})
        current_angle += value

    features = [{
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, r], [v[1], -(r - 1)], [v[0], -(r - 1)], [0, r]]]
        },
        "properties": {"cat": k}} for k, v in piechart.items()]

    data = {"values": {
            "type": "FeatureCollection",
            "features": features},
            "format": {"type": "json", "property": "features"}
            }

    color = alt.Color(field="properties.cat",
                      type="nominal",
                      legend={"title": None}
                      )
    chart = alt.Chart(data).mark_geoshape(stroke="#fff").encode(
        color={'field': "properties.cat",
               'type': "nominal",
               'legend': {"title": None}}
    ).project(type="azimuthalEquidistant", rotate=[0, 90, 0])
    return chart


def table(data, columns, labels=None, max_rows=10, filter=None):
    labels = labels if labels else columns  # TODO: Write function to strip ':' if present

    # Base chart for data tables
    table_column = alt.Chart(data).mark_text().encode(
        y=alt.Y('row_number:O', axis=None)
    ).transform_window(
        row_number='row_number()'
    ).transform_window(
        rank='rank(row_number)'
    )
    if filter:
        table_column = table_column.transform_filter(
            filter
        )

    table_column = table_column.transform_filter(
        alt.datum.rank <= max_rows
    )

    table_columns = [table_column.encode(text=column).properties(title=label) for column, label in zip(columns, labels)]
    table = alt.hconcat(*table_columns)
    return table


def pairgrid(data, variables, color=None, title=None, width=200, height=200, upper='scatter', diag='hist', lower='scatter',
             interactive=False, stack=True, alpha=0.3):
    """Plot pair-wise relationship of input variables

    Or copy this:
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.plotting.scatter_matrix.html

    Example:
    https://seaborn.pydata.org/tutorial/axis_grids.html#plotting-pairwise-data-relationships
    """
    size = len(variables)
    grid = np.full((size, size), None)


    for i, variable_1 in enumerate(variables):
        for j in range(i, size):
            variable_2 = variables[j]

            if i == j:
                # Diagonal plots
                grid[i, i] = histogram(data, variable_1, color=color, stack=stack, alpha=alpha)
            else:
                # Off-diagonal plots
                grid[j, i] = scatter(data, x=variable_1, y=variable_2, color=color)
                grid[i, j] = scatter(data, x=variable_2, y=variable_1, color=color)

    row_plots = []
    for row in grid:
        row_plots.append(alt.hconcat(*row))
    joined_plots = alt.vconcat(*row_plots)
    return joined_plots

def facet(data, rows, columns, color=None, title=None, width=200, height=200, interactive=False):
    """

    TODO: RENAME THIS REPEATGRID OR PAIRGRID, A FACET HAS THE SAME AXIS BUT DIFFERENT SELECTIONS PER PLOT

    RepeatGrid are limited, not sure how to implement selections

    :return:
    """
    # TODO: make the type of chart configurable
    chart = alt.Chart(data).mark_circle().encode(
        alt.X(alt.repeat("column"), type='quantitative'),
        alt.Y(alt.repeat("row"), type='quantitative'),
        color=color
    ).properties(
        width=width,
        height=height
    ).repeat(
        row=rows,
        column=columns
    )

    return _customise(chart, interactive=interactive, title=title)


def legend2d(data, color=None):
    legend = alt.Chart(data).mark_rect().encode(
        y=alt.Y('eta_bin:N', axis=alt.Axis(orient='right')),
        x='name:O',
        color=color
    )
    return legend
