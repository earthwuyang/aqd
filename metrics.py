import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

import copy

import numpy as np
from sklearn.metrics import mean_squared_error


class Metric:
    """
    Abstract class defining a metric used to evaluate the zero-shot cost model performance (e.g., Q-error)
    """

    def __init__(self, metric_prefix='val_', metric_name='metric', maximize=True, early_stopping_metric=False):
        self.maximize = maximize
        self.default_value = -np.inf
        if not self.maximize:
            self.default_value = np.inf
        self.best_seen_value = self.default_value
        self.last_seen_value = self.default_value
        self.metric_name = metric_prefix + metric_name
        self.best_model = None
        self.early_stopping_metric = early_stopping_metric

    def evaluate(self, model=None, metrics_dict=None, **kwargs):
        metric = self.default_value
        try:
            metric = self.evaluate_metric(**kwargs)
        except ValueError as e:
            print(f"Observed ValueError in metrics calculation {e}")
        self.last_seen_value = metric

        metrics_dict[self.metric_name] = metric
        print(f"{self.metric_name}: {metric:.4f} [best: {self.best_seen_value:.4f}]")

        best_seen = False
        if (self.maximize and metric > self.best_seen_value) or (not self.maximize and metric < self.best_seen_value):
            self.best_seen_value = metric
            best_seen = True
            if model is not None:
                self.best_model = copy.deepcopy(model.state_dict())
        return best_seen


class RMSE(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='rmse', maximize=False, **kwargs)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        val_mse = np.sqrt(mean_squared_error(labels, preds))
        return val_mse


class MRE(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='mre', maximize=False, **kwargs)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        mre = np.mean(np.abs((labels - preds) / labels))
        return mre


class QError(Metric):
    def __init__(self, percentile=50, min_val=0.1, **kwargs):
        super().__init__(metric_name=f'median_q_error_{percentile}', maximize=False, **kwargs)
        self.percentile = percentile
        self.min_val = min_val

    # def evaluate_metric(self, labels=None, preds=None, probs=None):
    #     if not np.all(labels >= self.min_val):
    #         print("WARNING: some labels are smaller than min_val")
    #     preds = np.abs(preds)
    #     # preds = np.clip(preds, self.min_val, np.inf)

    #     q_errors = np.maximum(labels / preds, preds / labels)
    #     q_errors = np.nan_to_num(q_errors, nan=np.inf)
    #     median_q = np.percentile(q_errors, self.percentile)
    #     return median_q

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        # Avoid division by zero by adding a small epsilon where necessary
        epsilon = 1e-10
        y_test_safe = np.where(labels == 0, epsilon, labels)
        y_pred_safe = np.where(preds <= 0, epsilon, preds)
        
        # Q-Error Calculation
        qerror = np.maximum(y_pred_safe / y_test_safe, y_test_safe / y_pred_safe)
        return np.percentile(qerror, self.percentile)
    
class MeanQError(Metric):
    def __init__(self, min_val=0.1, **kwargs):
        super().__init__(metric_name=f'mean_q_error', maximize=False, **kwargs)
        self.min_val = min_val

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        # Avoid division by zero by adding a small epsilon where necessary
        epsilon = 1e-10
        y_test_safe = np.where(labels == 0, epsilon, labels)
        y_pred_safe = np.where(preds <= 0, epsilon, preds)
        
        # Q-Error Calculation
        qerror = np.maximum(y_pred_safe / y_test_safe, y_test_safe / y_pred_safe)
        return np.mean(qerror)


def compute_metrics(y_test, y_pred):
    """
    Compute various regression metrics between y_test and y_pred.

    Parameters:
    - y_test (np.array): True values.
    - y_pred (np.array): Predicted values.

    Returns:
    - metrics (dict): Dictionary containing all computed metrics.
    """
    # Ensure inputs are NumPy arrays
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero by adding a small epsilon where necessary
    epsilon = 1e-10
    y_test_safe = np.where(y_test == 0, epsilon, y_test)
    y_pred_safe = np.where(y_pred <= 0, epsilon, y_pred)

    qerror_0 = QError(percentile=0, min_val=0.1).evaluate_metric(y_test_safe, y_pred_safe)
    qerror_50 = QError(percentile=50, min_val=0.1).evaluate_metric(y_test_safe, y_pred_safe)
    qerror_95 = QError(percentile=95, min_val=0.1).evaluate_metric(y_test_safe, y_pred_safe)
    qerror_max = QError(percentile=100, min_val=0.1).evaluate_metric(y_test_safe, y_pred_safe)
    mean_qerror = MeanQError(min_val=0.1).evaluate_metric(y_test_safe, y_pred_safe)
    
    mre = np.mean(np.abs((y_pred - y_test) / y_test))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    metrics = {
        'qerror_50 (Median)': qerror_50,
        'qerror_95': qerror_95,
        'qerror_max': qerror_max,
        'mean_qerror': mean_qerror,
        'qerror_0': qerror_0,
        'mre': mre,
        'rmse': rmse
    }
    
    return metrics

   