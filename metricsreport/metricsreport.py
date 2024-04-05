import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from sklearn.metrics import (
    log_loss,
    #f1_score,
    accuracy_score,
    precision_score,
    #recall_score,
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
    mean_poisson_deviance,
    classification_report,
    precision_recall_curve,
    roc_curve
)

import scikitplot as skplt
from plot_metric.functions import BinaryClassification
import matplotlib.pyplot as plt
from io import BytesIO

from .custom_metrics import lift, recall_score, f1_score


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
            self.probas_reval = pd.DataFrame(data={"proba_0": 1 - self.y_pred.ravel(), "proba_1": self.y_pred.ravel()})
            self.metrics = self._generate_classification_metrics()
        else:
            # assuming y_pred is a numpy array
            self.y_pred_nonnegative = np.maximum(self.y_pred, 0)
            self.metrics = self._generate_regression_metrics()

        self.binary_plots = {
            "all_count_metrics": self.plot_all_count_metrics,
            "class_hist": self.plot_class_hist,
            "tp_fp_with_optimal_threshold": self.plot_tp_fp_with_optimal_threshold,
            "class_distribution": self.plot_class_distribution,
            "confusion_matrix": self.plot_confusion_matrix,
            "precision_recall_curve": self.plot_precision_recall_curve,
            "roc_curve": self.plot_roc_curve,
            "ks_statistic": self.plot_ks_statistic,
            "calibration_curve": self.plot_calibration_curve,
            "cumulative_gain": self.plot_cumulative_gain,
            "precision_recall_vs_threshold": self.plot_precision_recall_vs_threshold,
            "lift_curve": self.plot_lift_curve
            }
        
        self.reg_plots = {
            "residual_plot": self.plot_residual_plot,
            "predicted_vs_actual": self.plot_predicted_vs_actual
            }

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

    def _generate_classification_metrics(self) -> dict:
        """
        Generates a dictionary of classification metrics.

        Returns:
            A dictionary of classification metrics.
        """
        tn, fp, fn, tp = confusion_matrix(y_true=self.y_true, y_pred=self.y_pred_binary).ravel()
        report = classification_report(
            y_true=self.y_true, 
            y_pred=self.y_pred_binary, 
            output_dict=True, 
            labels=[0, 1], 
            target_names=['negative', 'positive'], 
            zero_division='warn'
            )
        report = pd.json_normalize(report, sep=' ').to_dict(orient='records')[0]

        metrics = {
            'AP': round(average_precision_score(self.y_true, self.y_pred), 4),
            'AUC': round(roc_auc_score(self.y_true, self.y_pred), 4),
            'Log Loss': round(log_loss(self.y_true, self.y_pred), 4),
            'MSE': round(mean_squared_error(self.y_true, self.y_pred), 4),
            'Accuracy': round(accuracy_score(self.y_true, self.y_pred_binary,), 4),
            'Precision_weighted': round(precision_score(self.y_true, self.y_pred_binary, average='weighted'), 4),
            'MCC': round(matthews_corrcoef(self.y_true, self.y_pred_binary), 4),
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'TP': tp,
            'P precision': round(report['positive precision'], 4),
            'P recall': round(report['positive recall'], 4),
            'P f1-score': round(report['positive f1-score'], 4),
            'P support': report['positive support'],
            'N precision': round(report['negative precision'], 4),
            'N recall': round(report['negative recall'], 4),
            'N f1-score': round(report['negative f1-score'], 4),
            'N support': report['negative support'],
            #'Recall_weighted': round(recall_score(self.y_true, self.y_pred_binary, average='weighted'), 4),
            #'F1_weighted': round(f1_score(self.y_true, self.y_pred_binary, average='weighted'), 4),
        }
        return metrics
    
    def plot_roc_curve(self, figsize = (15, 10)) -> plt:
        """
        Generates a ROC curve plot.

        Args:
            figsize: the width and height of the figure.

        Returns:
            A ROC curve plot.
        """
        bc = BinaryClassification(y_true=self.y_true, y_pred=self.y_pred, labels=["Class 1", "Class 2"])
        plt.figure(figsize=figsize)
        bc.plot_roc_curve()
        return plt
    
    def plot_precision_recall_curve(self, figsize = (15, 10)) -> plt:
        """
        Generates a precision recall curve plot.

        Args:
            figsize: A tuple of the width and height of the figure.

        Returns:
            A precision recall curve plot.
        """
        bc = BinaryClassification(y_true=self.y_true, y_pred=self.y_pred, labels=["Class 1", "Class 2"])
        plt.figure(figsize=figsize)
        bc.plot_precision_recall_curve()
        return plt
    
    def plot_confusion_matrix(self, figsize = (15, 10)) -> plt:
        """
        Generates a confusion matrix plot.

        Args:
            figsize: A tuple of the width and height of the figure.

        Returns:
            A confusion matrix plot.
        """
        bc = BinaryClassification(y_true=self.y_true, y_pred=self.y_pred, labels=["Class 1", "Class 2"])
        plt.figure(figsize=figsize)
        bc.plot_confusion_matrix()
        return plt
    
    def plot_class_distribution(self, figsize = (15, 10)) -> plt:
        """
        Generates a class distribution plot.

        Args:
            figsize: A tuple of the width and height of the figure.

        Returns:
            A class distribution plot.
        """
        bc = BinaryClassification(y_true=self.y_true, y_pred=self.y_pred, labels=["Class 1", "Class 2"])
        plt.figure(figsize=figsize)
        bc.plot_class_distribution()
        return plt
    
    def plot_class_hist(self, figsize = (15, 10)) -> plt:
        """
        Generates a class histogram plot, showing the distribution of predicted probabilities
        for each actual class label.

        Args:
            figsize: A tuple of the width and height of the figure.

        Returns:
            A matplotlib figure with the histogram plot.
        """
        plt.style.use('ggplot')
        plt.figure(figsize=figsize)

        # Отдельные предсказания для классов 0 и 1
        preds_for_true_0 = [pred for pred, true in zip(self.y_pred, self.y_true) if true == 0]
        preds_for_true_1 = [pred for pred, true in zip(self.y_pred, self.y_true) if true == 1]

        # Гистограмма для класса 0
        plt.hist(preds_for_true_0, bins=100, edgecolor='black', alpha=0.5, label='Class 0')

        # Гистограмма для класса 1
        plt.hist(preds_for_true_1, bins=100, edgecolor='black', alpha=0.5, label='Class 1')

        plt.axvline(x=self.threshold, color='r', linestyle='--', label=f'Threshold: {self.threshold}')

        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Predicted Probability Histogram by Class')
        plt.legend()
        return plt
    
    def plot_all_count_metrics(self, step=101, plot_count_coef=1e-2, figsize=(15, 10)) -> plt:
        """
        Generates a plot of accuracy, precision, recall, and class distribution as a function of the decision threshold.

        Args:
            step: The number of steps to take between 0 and 1.
            plot_count_coef: The coefficient to multiply the count by in the scoring rule.
            figsize: A tuple of the width and height of the figure.

        Returns:
            A plot of accuracy, precision, recall, and class distribution as a function of the decision threshold.
        """
        plt.style.use('ggplot')
        plt.figure(figsize=figsize)
        accuracy_score_list = []
        precision_score_list = []
        recall_score_list = []
        list_classes = []
        list_counts = []

        pred_prob = np.array(self.y_pred)
        target = np.array(self.y_true, dtype=int)

        thresholds = np.linspace(0, 1, step)[:-1]
        for i in thresholds:
            predicted_labels = pred_prob > i

            accuracy_score_list.append(accuracy_score(target, predicted_labels))
            precision_score_list.append(precision_score(target, predicted_labels,))
            recall_score_list.append(recall_score(target, predicted_labels,))
            list_classes.append(predicted_labels.sum() / len(predicted_labels))
            list_counts.append(predicted_labels.sum())

        plt.plot(thresholds, accuracy_score_list, label='Accuracy')
        plt.plot(thresholds, precision_score_list, label='Precision')
        plt.plot(thresholds, recall_score_list, label='Recall')
        plt.plot(thresholds, list_classes, label='Class 1 count', color='black', linestyle='--')
        plt.axvline(x=self.threshold, color='r', linestyle='--', label=f'Threshold: {self.threshold}')

        min_count, max_count = min(list_counts), max(list_counts)
        for i, count in enumerate(list_counts):
            if (i % (len(list_counts) // 80) == 0 or count in (min_count, max_count)) and (list_counts[i-1] / list_counts[i]) - 1 > plot_count_coef:
                y_offset = list_classes[i] + (max(list_classes) - min(list_classes)) * 0.02
                plt.text(thresholds[i], y_offset, str(count), fontsize=7, rotation=90, fontweight='bold')

        plt.xlabel('Threshold')
        plt.ylabel('Scores')
        plt.title(f'accuracy, precision, recall, and class distribution')
        plt.legend()
        plt.grid(True)
        return plt
    
    def plot_calibration_curve(self, figsize = (15, 10)) -> plt:
        """
        Generates a calibration curve plot.

        Args:
            figsize: A tuple of the width and height of the figure.

        Returns:
            A calibration curve plot.
        """
        skplt.metrics.plot_calibration_curve(self.y_true, [self.probas_reval], n_bins=10, figsize=figsize)
        return plt
    
    def plot_lift_curve(self, figsize = (15, 10)) -> plt:
        """
        Generates a lift curve plot.

        Args:
            figsize: A tuple of the width and height of the figure.

        Returns:
            A lift curve plot.
        """
        skplt.metrics.plot_lift_curve(self.y_true, self.probas_reval, figsize=figsize)
        return plt
    
    def plot_cumulative_gain(self, figsize = (15, 10)) -> plt:
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
    

    def plot_precision_recall_vs_threshold(self, fp_coefficient: int =1, figsize=(15, 10)):
        """
        Plots Precision and Recall as a function of the decision threshold.

        Args:
            fp_coefficient (int): The coefficient to multiply FP by in the scoring rule.
            figsize (tuple): Figure size.

        Returns:
            matplotlib.pyplot: The matplotlib plot object.

        Raises:
            ValueError: If y_true and probas_pred do not have the same length.
            ValueError: If y_true and probas_pred are not 1-dimensional arrays.
        """
        y_true, probas_pred = self.y_true, self.y_pred
        # Validate inputs
        if len(y_true) != len(probas_pred):
            raise ValueError("y_true and probas_pred must have the same length.")
        if len(y_true.shape) != 1 or len(probas_pred.shape) != 1:
            raise ValueError("y_true and probas_pred must be 1-dimensional arrays.")
        
        thresholds = np.linspace(0, 1, 100)
        TP_list, FP_list, Scores_list = [], [], []
        
        for thresh in thresholds:
            pred_thresh = (probas_pred >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred_thresh).ravel()
            TP_list.append(tp)
            FP_list.append(fp)
            Scores_list.append(tp - (fp_coefficient*fp))  # Custom scoring criteria
        
        optimal_idx = np.argmax(Scores_list)
        optimal_threshold = thresholds[optimal_idx]

        # Calculate precision and recall for various thresholds
        precision, recall, thresholds = precision_recall_curve(y_true, probas_pred)
        
        # Create the plot
        plt.figure(figsize=figsize)
        plt.plot(thresholds, precision[:-1], "b--", label="Precision")
        plt.plot(thresholds, recall[:-1], "g-", label="Recall")
        # Highlighting the best threshold
        plt.scatter([optimal_threshold], [precision[optimal_idx]], color="blue", marker='o', label=f"Best for Precision: {optimal_threshold:.2f}")
        plt.scatter([optimal_threshold], [recall[optimal_idx]], color="green", marker='x', label=f"Best for Recall: {optimal_threshold:.2f}")
        plt.axvline(x=optimal_threshold, color='grey', linestyle='--', label=f'Best Threshold: {optimal_threshold:.2f}')
        
        plt.xlabel("Threshold")
        plt.ylabel("Metrics")
        plt.legend(loc="best")
        plt.title("Precision and Recall as a function of the decision threshold")
        plt.grid(True)
        
        return plt
    
    def plot_tp_fp_with_optimal_threshold(self, fp_coefficient: int =1, figsize=(15, 10)):
        """
        Plots the True Positives (TP) and False Positives (FP) rates across different thresholds and
        identifies the optimal threshold based on a scoring rule (TP - 2*FP).

        Args:
            fp_coefficient (int): The coefficient to multiply FP by in the scoring rule.
            figsize (tuple): Figure size.

        Returns:
            matplotlib.pyplot: The matplotlib plot object.

        Raises:
            ValueError: If y_true and probas_pred do not have the same length.
            ValueError: If y_true and probas_pred are not 1-dimensional arrays.
        """
        y_true, probas_pred = self.y_true, self.y_pred

        if len(y_true) != len(probas_pred):
            raise ValueError("y_true and probas_pred must have the same length.")
        if len(y_true.shape) != 1 or len(probas_pred.shape) != 1:
            raise ValueError("y_true and probas_pred must be 1-dimensional arrays.")
        
        thresholds = np.linspace(0, 1, 100)
        TP_list, FP_list, Scores_list = [], [], []
        
        for thresh in thresholds:
            pred_thresh = (probas_pred >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred_thresh).ravel()
            TP_list.append(tp)
            FP_list.append(fp)
            Scores_list.append(tp - (fp_coefficient*fp))  # Custom scoring criteria
        
        optimal_idx = np.argmax(Scores_list)
        optimal_threshold = thresholds[optimal_idx]

        # Create the plot
        plt.figure(figsize=figsize)
        plt.plot(thresholds, TP_list, "b--", label="TP (True Positives)")
        plt.plot(thresholds, FP_list, "r-", label="FP (False Positives)")
        plt.axvline(x=optimal_threshold, color='grey', linestyle='--', label=f'Optimal Threshold: {optimal_threshold:.2f}')
        plt.scatter([optimal_threshold], [TP_list[optimal_idx]], color="green", label="Optimal TP Threshold")
        plt.scatter([optimal_threshold], [FP_list[optimal_idx]], color="orange", label="Optimal FP Threshold")
        
        plt.xlabel("Threshold")
        plt.ylabel("Count")
        plt.legend(loc="best")
        plt.title("TP and FP counts as a function of the decision threshold")
        plt.grid(True)
        
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

        for plot_name, plot_func in self.binary_plots.items():
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
    
    def plot_residual_plot(self, figsize = (15, 10)) -> plt:
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
    
    def plot_predicted_vs_actual(self, figsize = (15, 10)) -> plt:
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
            if not os.path.exists(folder+'/plots'):
            #    shutil.rmtree(folder+'/plots')
                os.makedirs(folder+'/plots')

        for plot_name, plot_func in self.reg_plots.items():
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
                {self.add_svg_plots_to_html_rows()}
            </body>
        </html>
        """
        return html
    
    def add_svg_plots_to_html_rows(self, figsize = (15, 10)) -> str:
        """
        Adds SVG plots to HTML rows.

        Args:
            plots: A dictionary containing the SVG plots.

        Returns:
            A string containing the HTML rows with the SVG plots.
        """
        rows = ''
        if self.task_type == "classification":
            for name, plot in self.binary_plots.items():
                plt = plot(figsize=figsize)
                # Создаем объект BytesIO в памяти
                svg_io = BytesIO()
                # Сохраняем график в формате SVG в объект BytesIO
                plt.savefig(svg_io, format='svg', bbox_inches='tight')
                # Получаем содержимое объекта BytesIO и декодируем его в строку
                svg = '<svg' + svg_io.getvalue().decode('utf-8').split('<svg')[1]
                plt.close()
                rows += f'<tr><td>{svg}<br></td></tr>\n'
        elif self.task_type == "regression":
            for name, plot in self.reg_plots.items():
                plt = plot(figsize=figsize)
                # Создаем объект BytesIO в памяти
                svg_io = BytesIO()
                # Сохраняем график в формате SVG в объект BytesIO
                plt.savefig(svg_io, format='svg', bbox_inches='tight')
                # Получаем содержимое объекта BytesIO и декодируем его в строку
                svg = '<svg' + svg_io.getvalue().decode('utf-8').split('<svg')[1]
                plt.close()
                rows += f'<tr><td>{svg}<br></td></tr>\n'
        return rows

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

    def save_report(self, folder: str = 'report_metrics', name: str = 'report_metrics', verbose=0) -> None:
        """
        Creates and saves a report in HTML or markdown format.

        Args:
            folder (str): The folder to save the report to.
            name (str): The name of the report.
        """
        # Create the report directory
        if folder != '.':
            if not os.path.exists(folder):
                os.makedirs(folder)
        # Get target info
        self.__target_info()
        # Generate HTML report
        html = self._generate_html_report(folder, add_css=True)

        file_path = f'{folder}/{name}.html'
        with open(file_path, 'w') as f:
            f.write(html)

        if verbose > 0:
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
