# Metrics Report

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/metricsreport) 
![PyPI](https://img.shields.io/pypi/v/metricsreport) 
[![Telegram](https://img.shields.io/badge/chat-on%20Telegram-2ba2d9.svg)](https://t.me/automlalex) 
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE) 

-------------

MetricsReport is a Python package that generates classification and regression metrics report for machine learning models.


<img width=600 src="./exemples/metricsrepor_html.gif" alt="sample">

## Features
- AutoDetect the type of task
- Save report in .html and .md format
- Has several plotting functions


## Installation

You can install MetricsReport using pip:

```bash
pip install metricsreport
```

## Usage

```python
from metricsreport import MetricsReport  

# sample classification data 
y_true = [1, 0, 0, 1, 0, 1, 0, 1] 
y_pred = [0.8, 0.3, 0.1, 0.9, 0.4, 0.7, 0.2, 0.6]  

# generate report 
report = MetricsReport(y_true, y_pred, threshold=0.5)  

# print all metrics 
print(report.metrics)  

# plot ROC curve 
report.plot_roc_curve()

# saved MetricsReport (html) in folder: report_metrics
report.save_report()
```

More examples in the folder ./examples:


### Constructor

```python
MetricsReport(y_true, y_pred, threshold: float = 0.5)
```

*   `y_true` : list
    *   A list of true target values.
*   `y_pred` : list
    *   A list of predicted target values.
*   `threshold` : float
    *   Threshold for generating binary classification metrics. Default is 0.5.


## Plots

following methods can be used to generate plots:

*   `plot_roc_curve()`: Generates a ROC curve plot.
*   `plot_all_count_metrics()`: Generates a count metrics plot.
*   `plot_precision_recall_curve()`: Generates a precision-recall curve plot.
*   `plot_confusion_matrix()`: Generates a confusion matrix plot.
*   `plot_class_distribution()`: Generates a class distribution plot.
*   `plot_class_hist()`: Generates a class histogram plot.
*   `plot_calibration_curve()`: Generates a calibration curve plot.
*   `plot_lift_curve()`: Generates a lift curve plot.
*   `plot_cumulative_gain()`: Generates a cumulative gain curve plot.

### Dependencies

*   numpy
*   pandas
*   matplotlib
*   scikit-learn
*   scikit-plot

### License

This project is licensed under the MIT License.