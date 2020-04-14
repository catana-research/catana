"""
Regression models
"""
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression


def target_scaling():
    """

    https://scikit-learn.org/dev/auto_examples/compose/plot_transformed_target.html#sphx-glr-auto-examples-compose-plot-transformed-target-py
    Returns
    -------

    """
    # Can specify arbitrary transforms provided an inverse_func is specified
    regr_trans = TransformedTargetRegressor(regressor=RidgeCV(),
                                            func=np.log1p,
                                            inverse_func=np.expm1)

def regressor(X_train, y_train):
    from sklearn import preprocessing
    from sklearn.linear_model import LinearRegression
    from sklearn.compose import TransformedTargetRegressor
    # Why you should scale the target: https://scikit-learn.org/dev/auto_examples/compose/plot_transformed_target.html#sphx-glr-auto-examples-compose-plot-transformed-target-py



    # Transform the targets as part of the workflow
    # https://scikit-learn.org/dev/modules/compose.html#transforming-target-in-regression
    transformer = preprocessing.StandardScaler()
    regressor = LinearRegression()  # Regressor can be a Pipeline
    regr = TransformedTargetRegressor(regressor=regressor,
                                      transformer=preprocessing.StandardScaler())

    model = regr
    model.fit(X_train, y_train)
    model.regressor_.intercept_, model.regressor_.coef_  # You need to get the regressor parameters through the regressor_ method
    return


def linear_regression():
    """Ordinary least squares Linear Regression model.

    This is a paragraph describing the linear regression model.

    Here is some more text.

    Read more in the :ref:`User guide<models>`

    Parameters
    ----------
    param_1 : int, optional(default=100)
        The number of samples.

    param_2 : float between 0 and 1
        The weight.

    Returns
    -------
    x: array of shape [n, n]
        The input samples


    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> # y = 1 * x_0 + 2 * x_1 + 3
    >>> y = np.dot(X, np.array([1, 2])) + 3
    >>> reg = LinearRegression().fit(X, y)
    >>> reg.score(X, y)
    1.0
    >>> reg.coef_
    array([1., 2.])
    >>> reg.intercept_
    3.0000...
    >>> reg.predict(np.array([[3, 5]]))
    array([16.])

    Notes
    -----
    From the implementation point of view, this is just plain Ordinary
    Least Squares (scipy.linalg.lstsq) wrapped as a predictor object.
    """
    model = LinearRegression()
    return model

