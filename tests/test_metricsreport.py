import pytest
import numpy as np

from metricsreport import MetricsReport


@pytest.fixture
def binary_classification_data():
    y_true = [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    y_pred = [0.2, 0.8, 0.9, 0.1, 0.6, 0.3, 0.4, 0.7, 0.2, 0.9, 0.8, 0.4, 0.9]
    return y_true, y_pred

@pytest.fixture
def regression_data():
    y_true = [1, 2, 3, 4, 5]
    y_pred = [1.1, 2.3, 3.4, 4.2, 4.8]
    return y_true, y_pred

########### tests ############

def test_metrics_report_instance(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred, threshold=0.5)
    assert report.task_type == "classification"
    assert np.array_equal(report.y_true, np.array(y_true))
    assert np.array_equal(report.y_pred, np.array(y_pred))
    assert report.threshold == 0.5

def test_determine_task_type(binary_classification_data, regression_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred)
    assert report._determine_task_type(y_true) == "classification"
    
    y_true, y_pred = regression_data
    report = MetricsReport(y_true, y_pred)
    assert report._determine_task_type(y_true) == "regression"


def test_classification_metrics(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred, threshold=0.5)
    assert isinstance(report.metrics, dict)
    assert set(report.metrics.keys()) == {
    'AUC', 'Log Loss', 'Average_Precision', 
    'Accuracy', 'Precision', 'Recall',
    'F1 Score', 'MCC', 'TN', 'FP', 'FN', 'TP'
    }

def test_regression_metrics(regression_data):
    y_true, y_pred = regression_data
    report = MetricsReport(y_true, y_pred)
    assert isinstance(report.metrics, dict)
    assert set(report.metrics.keys()) == {
        'Mean Squared Error', 'Mean Squared Log Error', 
        'Mean Absolute Error', 'R^2', 
        'Explained Variance Score', 
        'Max Error', 
        'Mean Absolute Percentage Error'
    }

def test_classification_metrics_values(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred, threshold=0.5)
    assert report.metrics['AUC'] == 0.7857
    assert report.metrics['Log Loss'] == 0.5807
    assert report.metrics['Average_Precision'] == 0.6635
    assert report.metrics['Accuracy'] == 0.7692
    assert report.metrics['Precision'] == 0.7784
    assert report.metrics['Recall'] == 0.7692
    assert report.metrics['F1 Score'] == 0.7692
    assert report.metrics['MCC'] == 0.5476
    assert report.metrics['TN'] == 5
    assert report.metrics['FP'] == 2
    assert report.metrics['FN'] == 1
    assert report.metrics['TP'] == 5