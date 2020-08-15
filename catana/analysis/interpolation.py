import numpy as np
from scipy.interpolate import BarycentricInterpolator


def chebyshev_points(n, a=-1, b=1):
    """Returns Chebyshev nodes of specified dimensions for the range [a, b].

    Further reading: https://en.wikipedia.org/wiki/Chebyshev_nodes

    Args:
        n (int): Number of Chebyshev nodes.
        a (float, optional): Interval minimum.
        b (float, optional): Interval maximum.

    Returns:
        numpy.ndarray: 1D array of Chebyshev nodes.
    """
    x_i = 0.5 * (b + a) + 0.5 * (b - a) * np.cos(np.pi * (np.arange(1, n + 1) - 0.5) / n)
    return np.array(x_i)


def build_chebyshev_tensor(dimensions):
    """Build a Chebyshev tensor of specified dimensions.

    Args:
        dimensions (tuple): Number of points for each dimension of the Chebyshev tensor.

    Returns:
        numpy.ndarray: N-Dimensional array of Chebyshev nodes.
    """
    return np.array([chebyshev_points(dimension) for dimension in dimensions])


def _ndm(*args):
    return np.array([x[(None,)*i + (slice(None),) + (None,)*(len(args)-i-1)] for i, x in enumerate(args)])


def compute_chebyshev_tensor(grid, f):
    """Compute Chebyshev tensor points and evaluated values"""
    points = np.array(_ndm(*grid))
    values = f(*_ndm(*grid))
    return points, values


def offline_polynomial_tensor(grid, values):
    """Compute polynomial tensor, dimensions D - 1"""
    value_col = values.reshape(-1, len(grid[-1]))
    poly = []
    for value in value_col:
        poly.append(BarycentricInterpolator(grid[-1], value))
    poly = np.array(poly).reshape(values.shape[:-1])
    return poly


def interpolate_chebyshev_tensor(poly, grid, evaluation_vector):
    """Evaluate

    Interpolates the grid points of the highest remaining dimension and evaluate polynomials at each evaluation point.
    Reduces grid and point dimension by one and repeats until all points have been interpolated and evaluated.

    Args:
        poly (numpy.ndarray): Vector of offline interpolation polynomial.
        grid (numpy.ndarray): Positions of grid points for interpolation
        evaluation_vector:

    Returns:
        float: Interpolated value at evaluation vector.

    Raises:


    Examples:
        >>> def f_3d_cos(x, y, z):
        >>>     return np.cos(0.5 * np.pi * x) * np.sin(0.5 * np.pi * y) * np.sin(0.5 * np.pi * z)

    """
    # TODO: Investigate https://www.boost.org/doc/libs/1_65_0/libs/math/doc/html/math_toolkit/interpolate/barycentric.html
    # TODO: Check evaluation_vector lies inside the grid otherwise throw exception
    # TODO: Add dask delayed
    import cProfile
    profile = cProfile.Profile()

    profile.enable()

    # Get current point (z) and reduced grid (2d)
    eval_point = evaluation_vector[-1:]
    eval_vector = evaluation_vector[:-1]
    grid_current = np.array(grid)
    poly_current = poly.reshape(-1)

    dimensions = len(eval_vector)

    for d in range(dimensions):
        # TODO: Investigate possibility of changing yi (set_yi) instead of creating a new object
        # TODO: Stop using grid_current[-1]

        #value_point = np.array([p(eval_point) for p in poly_current]) # 5ms per call - 0.2 ms per evaluation
        value_point = np.array([p._evaluate(eval_point) for p in poly_current])  # 2.4 ms per call - 0.1 ms per evaluation
        #np.vectorize(p._evaluate)(eval_point) for p in poly_current])
        value_point = value_point.reshape(-1, grid_current[-1].shape[-1])  # 10 μs

        poly_current = []
        poly_new = BarycentricInterpolator(grid_current[-1], value_point[0])
        for value in value_point:  # 0.2 ms per constructor call
            poly_current.append(BarycentricInterpolator(grid_current[-1], value))
            # TODO: Investigate possibiility of changing yi (set_yi) instead of creating a new object
            # poly_new.set_yi(value_point[1])  # Both take ~0.250 ms
            # poly_new(eval_point)

        grid_current = grid_current[:-1]

        # 2nd iteration
        if eval_vector.shape[0] == 1:  # terminate
            eval_point = eval_vector
            #value = poly_current[0](eval_point)[0]  # Final value
            value = poly_current[0]._evaluate(eval_point)[0][0]  # Final value
        else:
            eval_point = eval_vector[-1]
            eval_vector = eval_vector[:-1]

    profile.disable()
    profile.print_stats(sort='cumtime')


    return value


if __name__ == '__main__':

    def f_3d(x, y, z):
        return np.cos(0.5 * np.pi * x) * np.sin(0.5 * np.pi * y) * np.sin(0.5 * np.pi * z)

    # Interpolation
    f = f_3d
    dimensions = (11, 11, 11)
    grid = build_chebyshev_tensor(dimensions)
    points, values = compute_chebyshev_tensor(grid, f)
    poly = offline_polynomial_tensor(grid, values)

    # Evaluation
    x_eval = np.linspace(-0.9, 0.9, 5)
    y_eval = np.linspace(-0.9, 0.9, 5)
    z_eval = np.linspace(-0.9, 0.9, 5)
    grids = (x_eval, y_eval, z_eval)

    eval_tensor = np.array(np.meshgrid(*grids))
    eval_col = np.array(eval_tensor).T.reshape(-1, eval_tensor.shape[0])

    # Evaluate a single points
    from catana.core.timer import Timer


    with Timer() as t:
        value = interpolate_chebyshev_tensor(poly, grid, eval_col[0])

    print(value, t.elapsed())

    # # Evaluate across a grid of points
    # values = np.array([interpolate_chebyshev_tensor(poly, grid, eval_vector) for eval_vector in eval_col])
    # values_true = np.array([f(*eval_vector) for eval_vector in eval_col])
    # error = np.abs(values - values_true)
    # print("Max error = ", np.max(error))
