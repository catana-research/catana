"""
Analysis
--------


Read this:
https://medium.com/dunder-data/from-pandas-to-scikit-learn-a-new-exciting-workflow-e88e2271ef62
"""

# 0. Setup
import numpy as np
np.random.seed(42)

from catana.analysis.transform import ndarray_to_df
from sklearn.datasets import samples_generator
X, y = samples_generator.make_classification(n_samples=1000, n_informative=2, n_redundant=0, n_features=3,
                                             random_state=42)

df = ndarray_to_df(X, y)

from sklearn.preprocessing import OneHotEncoder


# 0.1. Visualise features

import altair as alt
from catana.plotting.altchart import AltChart

selection_1 = alt.selection(type='interval', resolve='global')
#selection_1 = alt.selection(type='interval', encodings=['x'])

chart = AltChart(data=df, size=(1, 1)
                 ).facet(
    meta='Histogram with color',
    rows=df.columns.to_list(), columns=df.columns.to_list(), pos=(0, 0), interactive=False,
    # brush=selection_1,  # Enables you to draw on the chart
    # selection=selection_1,
    color='y0:O',
)


chart = AltChart(data=df, size=(2, 2)
                 ).pairgrid(
    meta='Histogram with color',
    variables=df.columns.to_list(), pos=(0, 0), interactive=True,
    brush=selection_1,  # Enables you to draw on the chart
    selection=selection_1,
    color='y0:O',
)
chart.serve()



chart = AltChart(data=df, size=(2, 2)
                 ).histogram(
    meta='Histogram with color',
    x='x0', pos=(0, 0), interactive=True,
    brush=selection_1,  # Enables you to draw on the chart
    selection=selection_1,
    color='y0:O',
)
chart.serve()


from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

n_jobs = 4
# TODO: Get index function:     rooms_ix, bedrooms_ix, population_ix, household_ix = [
#         list(housing.columns).index(col)
#         for col in ("total_rooms", "total_bedrooms", "population", "households")]

full_pipeline = ColumnTransformer([
        #("pca", PCA(n_components=0.95, random_state=42), ['x0', 'x1', 'x2', 'x3', 'x4']),  # Select on Variance
        ("pca", PCA(n_components=2, random_state=42), ['x0', 'x1', 'x2', 'x3', 'x4']),  # Select on dimension
    ], n_jobs=n_jobs, remainder='drop', verbose=True)

X_pca = full_pipeline.fit_transform(df)

"""
Bin features
------------
  n_bins : int or array-like, shape (n_features,) (default=5)
 encode : {'onehot', 'onehot-dense', 'ordinal'}, (default='onehot')
  strategy : {'uniform', 'quantile', 'kmeans'}, (default='quantile')
"""
from sklearn.preprocessing import KBinsDiscretizer
kbd = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')

discrete_pipeline = make_column_transformer((KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform'), [0, 1]))

discrete_pipeline.fit_transform(X_pca)



# 1. Visualise features


# 2. PCA features

from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline

pca_tsne = Pipeline([
    ("pca", PCA(n_components=0.95, random_state=42)),
    ("tsne", TSNE(n_components=2, random_state=42)),
])
t0 = time.time()
X_pca_tsne_reduced = pca_tsne.fit_transform(X)
t1 = time.time()
print("PCA+t-SNE took {:.1f}s.".format(t1 - t0))
plot_digits(X_pca_tsne_reduced, y)
plt.show()
save_fig("PCAt-SNE-MNIST")



# 2.1 Feature importance

# 3. Classification fit


# 4. Model performance
# 4.1 CV


