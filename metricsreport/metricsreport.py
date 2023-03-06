import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from sklearn.metrics import (
    log_loss,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    max_error,
    mean_squared_log_error,
    mean_poisson_deviance
)
from sklearn.metrics import classification_report
import scikitplot as skplt
from plot_metric.functions import BinaryClassification
import matplotlib.pyplot as plt
from markdownify import markdownify

from .custom_metrics import lift


class MetricsReport:
    """
    Class for generating reports on the metrics of a machine learning model.

    Args:
        y_true (List): A list of true target values.
        y_pred (List): A list of predicted target values.
        threshold (float): Threshold for generating binary classification metrics.

    Attributes:
        task_type: Type of task, either "classification" or "regression".
        y_true: A list of true target values.
        y_pred: A list of predicted target values.
        threshold: Threshold for generating binary classification metrics.
        metrics: A dictionary containing all metrics generated.
        target_info: A dictionary containing information about the target variable.
    """
    def __init__(self, y_true, y_pred, threshold: float = 0.5):
        """
        Initializes the MetricsReport object.

        Args:
            y_true: A list of true target values.
            y_pred: A list of predicted target values.
            threshold: Threshold for generating binary classification metrics.

        Returns:
            None
        """
        self.task_type = self._determine_task_type(y_true)
        print(f'Detecting {self.task_type} task type')
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.threshold = threshold
        self.metrics = {}
        self.target_info = {}

        if self.task_type == "classification":
            self.y_pred_binary = (self.y_pred > self.threshold).astype(int)
            # fix for skplt
            self.probas_reval = pd.DataFrame({"proba_0": 1 - self.y_pred.ravel(), "proba_1": self.y_pred.ravel()})
            self.metrics = self._generate_classification_metrics()
        else:
            # assuming y_pred is a numpy array
            self.y_pred_nonnegative = np.maximum(self.y_pred, 0)
            self.metrics = self._generate_regression_metrics()

    def _determine_task_type(self, y_true) -> str:
        """
        Determines the type of task based on the number of unique values in y_true.

        Args:
            y_true: A list of true target values.

        Returns:
            The type of task, either "classification" or "regression".
        """
        if len(np.unique(y_true)) > 2:
            return "regression"
        else:
            return "classification"

    ########## classification ###################################################

    def _generate_classification_metrics(self):
        """
        Generates a dictionary of classification metrics.

        Returns:
            A dictionary of classification metrics.
        """
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred_binary).ravel()

        metrics = {
            'AUC': round(roc_auc_score(self.y_true, self.y_pred), 4),
            'Log Loss': round(log_loss(self.y_true, self.y_pred), 4),
            'Average_Precision': round(average_precision_score(self.y_true, self.y_pred), 4),
            'Accuracy': round(accuracy_score(self.y_true, self.y_pred_binary), 4),
            'Precision': round(precision_score(self.y_true, self.y_pred_binary, average='weighted'), 4),
            'Recall': round(recall_score(self.y_true, self.y_pred_binary, average='weighted'), 4),
            'F1 Score': round(f1_score(self.y_true, self.y_pred_binary, average='weighted'), 4),
            'MCC': round(matthews_corrcoef(self.y_true, self.y_pred_binary), 4),
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'TP': tp,
        }
        return metrics
    
    def plot_roc_curve(self, figsize = (12, 10)) -> plt:
        """
        Generates a ROC curve plot.

        Args:
            figsize: the width and height of the figure.

        Returns:
            A ROC curve plot.
        """
        bc = BinaryClassification(self.y_true, self.y_pred, labels=["Class 1", "Class 2"])
        plt.figure(figsize=figsize)
        bc.plot_roc_curve()
        return plt
    
    def plot_precision_recall_curve(self, figsize = (12, 10)) -> plt:
        """
        Generates a precision recall curve plot.

        Args:
            figsize: A tuple of the width and height of the figure.

        Returns:
            A precision recall curve plot.
        """
        bc = BinaryClassification(self.y_true, self.y_pred, labels=["Class 1", "Class 2"])
        plt.figure(figsize=figsize)
        bc.plot_precision_recall_curve()
        return plt
    
    def plot_confusion_matrix(self, figsize = (12, 10)) -> plt:
        """
        Generates a confusion matrix plot.

        Args:
            figsize: A tuple of the width and height of the figure.

        Returns:
            A confusion matrix plot.
        """
        bc = BinaryClassification(self.y_true, self.y_pred, labels=["Class 1", "Class 2"])
        plt.figure(figsize=figsize)
        bc.plot_confusion_matrix()
        return plt
    
    def plot_class_distribution(self, figsize = (12, 10)) -> plt:
        """
        Generates a class distribution plot.

        Args:
            figsize: A tuple of the width and height of the figure.

        Returns:
            A class distribution plot.
        """
        bc = BinaryClassification(self.y_true, self.y_pred, labels=["Class 1", "Class 2"])
        plt.figure(figsize=figsize)
        bc.plot_class_distribution()
        return plt
    
    def plot_calibration_curve(self, figsize = (12, 10)) -> plt:
        """
        Generates a calibration curve plot.

        Args:
            figsize: A tuple of the width and height of the figure.

        Returns:
            A calibration curve plot.
        """
        skplt.metrics.plot_calibration_curve(self.y_true, [self.probas_reval], n_bins=10, figsize=figsize)
        return plt
    
    def plot_lift_curve(self, figsize = (12, 10)) -> plt:
        """
        Generates a lift curve plot.

        Args:
            figsize: A tuple of the width and height of the figure.

        Returns:
            A lift curve plot.
        """
        skplt.metrics.plot_lift_curve(self.y_true, self.probas_reval, figsize=figsize)
        return plt
    
    def plot_cumulative_gain(self, figsize = (12, 10)) -> plt:
        """
        Generates a cumulative gain curve plot.

        Args:
            figsize: A tuple of the width and height of the figure.

        Returns:
            A cumulative gain curve plot.
        """
        skplt.metrics.plot_cumulative_gain(self.y_true, self.probas_reval, figsize=figsize)
        return plt

    def plot_ks_statistic(self, figsize=(12,10)):
        """
        Generates a KS statistic plot.

        Args:
            figsize: A tuple of the width and height of the figure.

        Returns:
            A KS statistic plot.
        """
        skplt.metrics.plot_ks_statistic(self.y_true, self.probas_reval, figsize=figsize)
        return plt

    def _classification_plots(self, save: bool = False, folder: str = '.') -> None:
        """
        Generates a dictionary of classification plots.

        Args:
            save: A boolean indicating whether to save the plots.
            folder: The folder to save the plots in.

        Returns:
            None.
        """
        if save:
            if os.path.exists(folder+'/plots'):
                shutil.rmtree(folder+'/plots')
            os.makedirs(folder+'/plots')

        plots = {
            "class_distribution": self.plot_class_distribution,
            "confusion_matrix": self.plot_confusion_matrix,
            "precision_recall_curve": self.plot_precision_recall_curve,
            "roc_curve": self.plot_roc_curve,
            "ks_statistic": self.plot_ks_statistic,
            "calibration_curve": self.plot_calibration_curve,
            "cumulative_gain": self.plot_cumulative_gain,
            "lift_curve": self.plot_lift_curve
            }

        for plot_name, plot_func in plots.items():
            plt = plot_func()
            if save:
                plt.savefig(f'{folder}/plots/{plot_name}.png')
            else:
                plt.show()
            plt.close()

    ########## regression ###################################################

    def _generate_regression_metrics(self) -> dict:
        """
        Generates a dictionary of regression metrics.

        Returns:
            A dictionary of regression metrics.
        """
        metrics = {
            'Mean Squared Error': round(mean_squared_error(self.y_true, self.y_pred), 4),
            'Mean Squared Log Error': round(mean_squared_log_error(self.y_true, self.y_pred_nonnegative), 4),
            'Mean Absolute Error': round(mean_absolute_error(self.y_true, self.y_pred), 4),
            'R^2': round(r2_score(self.y_true, self.y_pred), 4),
            'Explained Variance Score': round(explained_variance_score(self.y_true, self.y_pred), 4),
            'Max Error': round(max_error(self.y_true, self.y_pred), 4),
            'Mean Absolute Percentage Error': round(np.mean(np.abs((self.y_true - self.y_pred) / self.y_true)) * 100, 1),
        }
        return metrics
    
    def plot_residual_plot(self, figsize = (12, 10)) -> plt:
        """
        Generates a residual plot.

        Args:
            figsize: Figure size for plot.

        Returns:
            A residual plot.
        """
        plt.figure(figsize=figsize)
        plt.scatter(self.y_pred, self.y_true - self.y_pred)
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        return plt
    
    def plot_predicted_vs_actual(self, figsize = (12, 10)) -> plt:
        """
        Generates a predicted vs actual plot.

        Args:
            figsize: Figure size for plot.

        Returns:
            A predicted vs actual plot.
        """
        plt.figure(figsize=figsize)
        plt.scatter(self.y_pred, self.y_true)
        plt.xlabel("Predicted Values")
        plt.ylabel("Actual Values")
        plt.title("Predicted vs Actual")
        return plt
    
    def _regression_plots(self, save: bool = False, folder: str = '.') -> None:
        """
        Generates a dictionary of regression plots.

        Args:
            save: Whether to save the plots to disk.
            folder: Folder path where to save the plots.
        """
        if save:
            if os.path.exists(folder+'/plots'):
                shutil.rmtree(folder+'/plots')
            os.makedirs(folder+'/plots')
        plots = {
            "residual_plot": self.plot_residual_plot,
            "predicted_vs_actual": self.plot_predicted_vs_actual
            }
        for plot_name, plot_func in plots.items():
            plt = plot_func()
            if save:
                plt.savefig(f'{folder}/plots/{plot_name}.png')
            else:
                plt.show()
            plt.close()

    ########## HTML Report #################################################

    def __target_info(self):
        """
        Generates a dictionary of target information.

        Returns:
            A dictionary of target information.
        """
        if self.task_type == 'classification':
            target_info = {
                'Count of samples': self.y_true.shape[0],
                'Count True class': sum(self.y_true),
                'Count False class': (len(self.y_true) - sum(self.y_true)),
                'Class balance %': round((sum(self.y_true) / len(self.y_true)) * 100, 1),
            }
        if self.task_type == 'regression':
            target_info = {
                'Count of samples': self.y_true.shape[0],
                'Mean of target': round(np.mean(self.y_true), 2),
                'Std of target': round(np.std(self.y_true), 2),
                'Min of target': round(np.min(self.y_true), 2),
                'Max of target': round(np.max(self.y_true), 2),
            }
        self.target_info = target_info

    def _generate_html_report(self, folder='report_metrics', add_css=True) -> str:
        """
        Generates an HTML report.

        Args:
            folder (str): The folder to save the report in. Defaults to 'report_metrics'.
            add_css (bool): Whether to add CSS styles to the report. Defaults to True.

        Returns:
            A string containing the HTML report.
        """
        css = """
        <style>
            body {
                font-family: Arial, sans-serif;
                font-size: 16px;
                line-height: 1.5;
            }

            h1, h2 {
                margin-top: 40px;
                margin-bottom: 20px;
            }

            table {
                border-collapse: collapse;
                margin-bottom: 40px;
            }

            th, td {
                border: 1px solid #ccc;
                padding: 8px;
            }

            th {
                background-color: #f2f2f2;
            }

            img {
                max-width: 100%;
                height: auto;
            }
        </style>
        """ if add_css else ""

        html = f"""
        <!DOCTYPE html>
        <html>
            <head>
                {css}
            </head>
            <body>
                <h1>Metrics Report</h1>
                <h4>Type: {self.task_type}</h4>
                <h2>Data info</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Info</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        {self.__generate_html_rows(self.target_info)}
                    </tbody>
                </table>
                <h2>Metrics</h2>
                <p><b>threshold: {self.threshold}</b></p>
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        {self.__generate_html_rows(self.metrics)}
                    </tbody>
                </table>
                <h2>Plots</h2>
                {self.__add_plot_images_to_report(folder)}
            </body>
        </html>
        """
        return html

    def __generate_html_rows(self, data: dict) -> str:
        """
        Generates HTML rows.

        Args:
            data: A dictionary containing the data to be displayed.

        Returns:
            A string containing the HTML rows.
        """
        rows = ''
        for name, value in data.items():
            rows += f'<tr><td>{name}</td><td>{value}</td></tr>\n' if isinstance(value, float) else f'<tr><td>{name}</td><td>{int(value)}</td></tr>\n'
        return rows

    def __add_plot_images_to_report(self, folder='report_metrics',) -> str:
        """
        Generates HTML image tags for each plot.

        Returns:
            A string containing the HTML or markdown image tags for each plot.
        """
        images = ''
        directory = f'{folder}/plots/'
        png_files = [file for file in os.listdir(directory) if file.endswith('.png')]

        for name in png_files:
            images += f'<img src="./plots/{name}"></br>\n'
        return images

    def save_report(self, folder: str = 'report_metrics', name: str = 'report_metrics') -> None:
        """
        Creates and saves a report in HTML or markdown format.

        Args:
            folder (str): The folder to save the report to.
            name (str): The name of the report.
        """
        # Create the report directory
        if folder != '.':
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder)
        # Get target info
        self.__target_info()
        # Generate plots based on task type
        if self.task_type == 'classification':
            self._classification_plots(save=True, folder=folder)
        elif self.task_type == 'regression':
            self._regression_plots(save=True, folder=folder)
        # Generate HTML report
        html = self._generate_html_report(folder, add_css=True)
        with open(f'{folder}/{name}.html', 'w') as f:
            f.write(html)
        # Convert HTML report to markdown
        md = markdownify(self._generate_html_report(folder, add_css=False), heading_style="ATX")
        with open(f'{folder}/{name}.md', 'w') as f:
            f.write(md)
        print(f'Report saved in folder: {folder}')

    def print_metrics(self) -> None:
        """
        Prints the metrics dictionary.
        """
        print(pd.DataFrame(self.metrics, index=['score']).T)

    def plot_metrics(self) -> None:
        """
        Plots classification or regression metrics based on task type.
        """
        if self.task_type == 'classification':
            self._classification_plots(save=False,)
        elif self.task_type == 'regression':
            self._regression_plots(save=False,)

    def print_report(self):
        """
        Prints the metrics and plots generated by MetricsReport.
        """
        if self.task_type == 'classification':
            print(f'threshold={self.threshold}')
            print("\n                  |  Classification Report | \n")
            print(classification_report(self.y_true, self.y_pred_binary, target_names=["Class 0", "Class 1"]))
            print("\n                  |  Metrics Report: | \n")
            self.print_metrics()
            print("\n                  |  Lift: | \n")
            print(lift(self.y_true, self.y_pred))
            print("\n                  |  Plots: | \n")
            self.plot_metrics()

        elif self.task_type == 'regression':
            print("\n                  |  Metrics Report: | \n")
            self.print_metrics()
            print("\n                  |  Plots: | \n")
            self.plot_metrics()
