# -*- coding: utf-8 -*-
"""
Development for altplot
------------------------

Iris schema:

     sepalLength  sepalWidth  petalLength  petalWidth    species
0            5.1         3.5          1.4         0.2     setosa
1            4.9         3.0          1.4         0.2     setosa
2            4.7         3.2          1.3         0.2     setosa

# To add:
kwargs['color'] = alt.condition(brush, kwargs['color'], alt.ColorValue('gray'))


"""
import altair as alt
import numpy as np
import pandas as pd
from vega_datasets import data

from catana.plotting.altchart import AltChart


def example1():
    """Plot a grid of four plots with selections"""
    df = data.iris()
    selection_1 = alt.selection(type='interval', encodings=['x'])
    selection_2 = alt.selection(type='interval', resolve='global')

    chart = AltChart(data=df, size=(2, 2)
        ).histogram(
            meta='Histogram with color',
            x='sepalLength', pos=(0, 0), interactive=True,
            brush=selection_1,  # Enables you to draw on the chart
            selection=selection_1,
            color='species',
        ).histogram(
            meta='Histogram without color',
            x='sepalWidth', pos=(0, 1), interactive=True,
            brush=selection_1,
            selection=selection_1,
        ).scatter(
            meta='2D relationship',
            x='sepalLength', y='sepalWidth', color='species', pos=(1, 1), interactive=True,
            selection=selection_1,
            brush=selection_1,  # Enables you to draw on the chart
        ).line(
            meta='1D projection of relationship, shows new projections with selections',
            x='sepalLength', y='sepalWidth', color='species', pos=(1, 0),
            selection=selection_1, brush=selection_1,
        )
    chart.serve()


def example2():
    """Plot a grid of four plots with selections"""
    df = data.iris()
    selection_1 = alt.selection(type='interval', resolve='global')
    selection_2 = alt.selection(type='interval', resolve='global')
    x, y = np.meshgrid(range(-5, 5), range(-5, 5))
    z = x ** 2 + y ** 2
    df2 = pd.DataFrame({'x': x.ravel().astype('i'),  # Cast to data type in to avoid Altair asking for the type
                       'y': y.ravel().astype('i'),
                       'z': z.ravel().astype('d')})

    # chart = AltChart(data=df, size=(2, 2)
    #     ).histogram2d(
    #         meta='2D Histogram with color',
    #         x='x:O', y='y:O', z='z:Q', pos=(0, 0), interactive=True,
    #         brush=selection_1,  # Enables you to draw on the chart
    #         selection=selection_1,
    #         color='species',
    #     )
# sepalLength  sepalWidth  petalLength  petalWidth

    chart = AltChart(data=df, size=(2, 2)
        ).histogram2d(
            meta='2D Histogram with color',
            x='sepalLength', y='sepalWidth', z='count(sepalWidth)', pos=(0, 0),
            brush=selection_1, selection=selection_1,
        ).histogram(
            meta='1D Histogram with color',
            x='species', bins=True, selection=selection_1, pos=(0, 1),
        ).histogram2d(
            data=df2,
            meta='2D Histogram with color',
            x='x:O', y='y:O', z='z:Q',
            brush=selection_2, selection=selection_2, pos=(1, 1),
    )

    chart.serve()

# chart = AltChart(data=df, size=(2, 2)
#     ).scatter(
#         x='sepalLength', y='sepalWidth', color='species', pos=(0, 0), interactive=True,
#         brush=alt.selection(type='interval', resolve='global'),  # Enables you to draw on the chart
#     ).scatter(
#         x='petalLength', y='petalWidth', color='species', pos=(0, 1), interactive=False,
#         brush=alt.selection(type='interval', resolve='global')
#     )
#     ).line(
#         x='sepalLength', y='sepalWidth', color='species', pos=(1, 0)
#     ).table(
#         columns=['sepalLength', 'sepalWidth', 'species'], pos=(1, 1)
#     )

#
# import pandas as pd
# import numpy as np
# x = np.arange(100)
# data = pd.DataFrame({
#   'x': x,
#   'f(x)': np.sin(x / 5)

# })
#
#
# chart = AltChart(data=data, size=(2, 2)
#     ).line(
#         x='x:Q', y='f(x):Q', color='x:Q', pos=(0, 0)
#     )

"""Facet"""
# df = data.iris()
# chart = AltChart(data=df, size=(2, 2)
#     ).facet(rows=['sepalLength', 'sepalWidth'], columns=['petalLength', 'petalWidth'], color='species', width=200, height=200, pos=(0, 0))


def example5():
    """Line plot with confidence intervals"""

    bins = np.linspace(0, 0.9, 10)
    x = np.linspace(0, 1, 1000)
    x = np.digitize(x, bins)
    y = x + np.random.normal(0, 3, 1000)
    df = pd.DataFrame({'x': x, 'y': y})
    chart = AltChart(data=df, size=(1, 1)).line(
        x='x', y='mean(y)', ci_y='y',
        interactive=True,
    )

    chart.serve()


def main():
    example1()
    #example2()
    #example5()
    return


if __name__ == '__main__':
    main()

