"""
Analysis
--------


Read this:
https://medium.com/dunder-data/from-pandas-to-scikit-learn-a-new-exciting-workflow-e88e2271ef62
"""
from sklearn.compose import ColumnTransformer

# 0. Setup
import numpy as np
np.random.seed(42)

from catana.analysis.transform import ndarray_to_df
from sklearn.datasets import samples_generator
X, y = samples_generator.make_classification(n_samples=1000, n_informative=2, n_redundant=0, n_features=3,
                                             random_state=42)

df = ndarray_to_df(X, y)

from sklearn.decomposition import PCA

def df_pipeline(steps):
    """
    TODO: make a pipeline that handles dataframes instead of np.ndarray by maintaining a dictionary of columns and tracking
    the mapping through the pipeline
    """
    from sklearn.compose import ColumnTransformer

    columns = [step[2] for step in steps]

    transformer = ColumnTransformer(df, f, columns)
    return

pipeline = df_pipeline([("pca", PCA(n_components=2, random_state=42), ['x0', 'x1', 'x2', 'x3', 'x4'])

])


from sklearn.compose import make_column_transformer


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
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE

pca_tsne = Pipeline([
    ("pca", PCA(n_components=0.95, random_state=42)),
    ("tsne", TSNE(n_components=2, random_state=42)),
])


"""
k-Nearest neighbors
"""



"""
Linear regression
"""

"""
Logistic regression
"""

"""
Support Vector Machine
"""

"""
Random forest
"""

"""
Neural network
"""
