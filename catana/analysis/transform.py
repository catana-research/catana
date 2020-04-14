"""
Transform
---------

Transform data
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


def ndarray_to_df(X, y=None, x_columns=None, y_columns=None, index=None):
    def get_size(x):
        if len(x.shape) == 1:
            return 1
        else:
            return x.shape[1]

    data = np.column_stack([X, y]) if y is not None else X
    if not x_columns:
        x_columns = ['x{}'.format(i) for i in range(get_size(X))]
    if not y_columns:
        y_columns = ['y{}'.format(i) for i in range(get_size(y))]
    columns = x_columns + y_columns
    df = pd.DataFrame(data=data, index=index, columns=columns)
    return df

def df_to_ndarray(df, columns):
    x = df[columns].values
    return x

def pca(x, n_components):

    x_pca = PCA(n_components=n_components).fit_transform(x)
    return x_pca

def polynomial(X, degree):
    from sklearn.preprocessing import PolynomialFeatures
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    return X_poly


def test():
    """
    Pandas/Numpy Transform
    """
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import Normalizer
    ct = ColumnTransformer(
        [("norm1", Normalizer(norm='l1'), [0, 1]),
         ("norm2", Normalizer(norm='l1'), slice(2, 4))])
    X = np.array([[0., 1., 2., 2.],
                  [1., 1., 0., 1.]])
    # Normalizer scales each row of X to unit norm. A separate scaling
    # is applied for the two first and two last elements of each
    # row independently.
    ct.fit_transform(X)
    """
    array([[0. , 1. , 0.5, 0.5],
           [0.5, 0.5, 0. , 1. ]])
    """


    """
    Custom Transforms
    -----------------
    
    Constructs a transformer from an arbitrary callable.
    
    A FunctionTransformer forwards its X (and optionally y) arguments to a
    user-defined function or function object and returns the result of this
    function. This is useful for stateless transformations such as taking the
    log of frequencies, doing custom scaling, etc.
    
    Useful parameters
    -----------------
    n_jobs 
        Parallelize
    transformers 
        Estimator (implements fit and transform) or string: 'drops' or 'passthrough'
    columns
        Either column index or column name
    remainder
        Treatment of columns that are not specified by `columns`. Can be either 
        Estimator, 'drop' or 'passthrough', defaults to 'drop'.
    transformer_weights
    
    
    Useful macros
    -------------
    make_column_transformer
    
    make_column_selector
        Select column by dtype or regex of column name. Only available for 0.23 (not stable yet)
    
    More examples: https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#sphx-glr-auto-examples-compose-plot-column-transformer-mixed-types-py
    """
    # Example 1
    def all_but_first_column(X):
        return X[:, 1:]

    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import FunctionTransformer

    pipeline = make_pipeline(
        PCA(), FunctionTransformer(all_but_first_column),  # Apply the custom function
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    pipeline.fit(X_train, y_train)



    # Example 2

    # Get the index of particular columns
    rooms_ix, bedrooms_ix, population_ix, household_ix = [
        list(housing.columns).index(col)
        for col in ("total_rooms", "total_bedrooms", "population", "households")]


    def add_extra_features(X, add_bedrooms_per_room=True):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

    attr_adder = FunctionTransformer(add_extra_features, validate=False,
                                     kw_args={"add_bedrooms_per_room": False})
    housing_extra_attribs = attr_adder.fit_transform(housing.values)

