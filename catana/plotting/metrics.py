import matplotlib.pyplot as plt
import numpy as np
import sklearn
import json
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score




def plot_training_fit(model, X_train, y_train):
    """
    ----------
    model
    X
    y
    step
    random_state
    test_size

    Returns
    -------

    """
    y_train_predict = model.predict(X_train)

    plt.scatter(X_train, y_train, s=0.4, label="Training set")
    plt.plot(X_train, y_train_predict, "r", linewidth=2, label="Model fit")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("X", fontsize=14)
    plt.ylabel("y", fontsize=14)


def plot_gridsearch_metrics(grid_search, max_params=10):
    """Plot the test and train CV scores ordered by rank and the fit times.

    Args:
        grid_search: GridSearch result.
        max_params: Maximum number of parameters to show on plot, show all if None.

    Returns:
        Matplotlib plots
    """
    if not max_params:
        max_params = -1

    rank = grid_search.cv_results_['rank_test_score']
    # param_labels = [f'{k}: {v}' for param in grid_search.cv_results_['params'] for k, v in param.items()]
    # param_labels = [x for y, x in sorted(zip(rank, param_labels))][:max_params]
    param_labels = json.dumps(grid_search.cv_results_['params'])[1:-1].split('}, ')
    param_labels = [param + '}' if param[-1] != '}' else param for param in param_labels][:max_params]

    mean_train_score = [x for y, x in sorted(zip(rank, grid_search.cv_results_['mean_train_score']))][:max_params]
    std_train_score = [x for y, x in sorted(zip(rank, grid_search.cv_results_['std_train_score']))][:max_params]
    mean_test_score = [x for y, x in sorted(zip(rank, grid_search.cv_results_['mean_test_score']))][:max_params]
    std_test_score = [x for y, x in sorted(zip(rank, grid_search.cv_results_['std_test_score']))][:max_params]
    mean_fit_time = [x for y, x in sorted(zip(rank, grid_search.cv_results_['mean_fit_time']))][:max_params]
    std_fit_time = [x for y, x in sorted(zip(rank, grid_search.cv_results_['std_fit_time']))][:max_params]

    f, axs = plt.subplots(1, 2)
    axs[0].errorbar(param_labels, mean_test_score, yerr=std_test_score, ls='none', alpha=0.9, capsize=4, linewidth=1)
    axs[0].scatter(param_labels, mean_test_score, label='Test')
    axs[0].errorbar(param_labels, mean_train_score, yerr=std_train_score, ls='none', alpha=0.9, capsize=4, linewidth=2, color='r')
    axs[0].scatter(param_labels, mean_train_score, label='Train', color='r')
    axs[0].set_xticklabels(param_labels, rotation=90)
    axs[0].set_ylabel('CV score')
    axs[0].legend()

    axs[1].errorbar(param_labels, mean_fit_time, yerr=std_fit_time, ls='none', alpha=0.9, capsize=4, linewidth=2)
    axs[1].scatter(param_labels, mean_fit_time)
    axs[1].set_ylabel('Fit time (s)')
    axs[1].set_xticklabels(param_labels, rotation=90)

    plt.tight_layout()


def plot_fit_residual(model, X_train, y_train, X_val, y_val):
    """Plot the model fit residual nnd correlation with the validation set

    Args:
        model:
        X_train:
        y_train:
        X_val:
        y_val:

    Returns:
        matplotlib plot object
    """
    # Measure the fit residual and correlation using the validation set
    from sklearn.metrics import median_absolute_error, r2_score
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    residual = y_pred - y_val

    f, (ax0, ax1) = plt.subplots(1, 2)
    x_min_true, x_max_true = min(y_val), max(y_val)
    x_min_pred, x_max_pred = min(y_pred), max(y_pred)
    x_min, x_max = min(x_min_true, x_min_pred), max(x_max_true, x_max_pred)

    ax0.scatter(y_val, residual, alpha=0.1)
    ax0.plot([x_min_true, x_max_true], [0, 0], '--k')
    ax0.set_ylabel('Residual')
    ax0.set_xlabel('True Target')
    ax0.set_title('Regression \n residual')

    ax1.scatter(y_val, y_pred, alpha=0.1)
    ax1.plot([x_min_true, x_max_true], [x_min_pred, x_max_pred], '--k')
    ax1.set_ylabel('Target predicted')
    ax1.set_xlabel('True Target')
    ax1.set_title('Regression \n with target')
    ax1.text(x_min_true, 0.9 * x_max_pred, r'$R^2$=%.2f, MAE=%.2f' % (
        r2_score(y_val, y_pred), median_absolute_error(y_val, y_pred)))

    f.suptitle("Regression metrics", y=0.035)
    f.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])


def model_cv(model, X, y, n_splits=10, random_state=42, scoring='neg_mean_squared_error'):
    """

    Parameters
    ----------
    model
    X
    y
    n_splits
    random_state

    Returns
    -------

    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, scoring=scoring, cv=cv, n_jobs=-1)
    # convert scores to positive
    scores = np.absolute(scores)
    print('{} = {:.3f} ± {:.3f}'.format(scoring, np.mean(scores), np.std(scores)))


def plot_learning_curves(model, X, y, m_min=1, rmse_min=0, rmse_max=3, step=10, random_state=42, test_size=0.2):
    """
    Splits X into train and validation set and plots model RMSE with sample size.
    Parameters
    ----------
    model
    X
    y
    step
    random_state
    test_size

    Returns
    -------

    """

    # if type(model) is sklearn.compose._target.TransformedTargetRegressor:
    #     model = model.regressor_

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    train_errors, val_errors, training_size = [], [], []
    for m in range(m_min, len(X_train), step):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
        training_size.append(m)

    plt.plot(training_size, np.sqrt(train_errors), "r-", linewidth=2, label="Train")
    plt.plot(training_size, np.sqrt(val_errors), "b-", linewidth=2, label="Validation")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
    plt.ylim([rmse_min, rmse_max])
