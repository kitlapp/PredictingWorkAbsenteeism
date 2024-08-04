import pandas as pd
import numpy as np


def range_monthdays(day):
    """
    Categorize a day of the month into one of six ranges.

    This function assigns a given day of the month into one of six categories:
    1. Days 1 to 5
    2. Days 6 to 10
    3. Days 11 to 15
    4. Days 16 to 20
    5. Days 21 to 25
    6. Days 26 to 31

    Parameters:
    day (int): The day of the month to categorize (1 to 31).

    Returns:
    int: The category number (1 to 6) that the day falls into.
    """
    # Check if the day is between 1 and 5
    if 1 <= day <= 5:
        return 1
    # Check if the day is between 6 and 10
    elif 6 <= day <= 10:
        return 2
    # Check if the day is between 11 and 15
    elif 11 <= day <= 15:
        return 3
    # Check if the day is between 16 and 20
    elif 16 <= day <= 20:
        return 4
    # Check if the day is between 21 and 25
    elif 21 <= day <= 25:
        return 5
    # If the day is between 26 and 31, it falls into the last category
    else:
        return 6


def summary_metrics(feature_df, model, x_tr, y_tr, x_te, y_te):
    """
        Generates a summary of model metrics including feature weights, intercept, and accuracies.

        Parameters:
        - feature_df (pd.DataFrame): DataFrame containing feature names.
        - model (sklearn.linear_model.LogisticRegression): Trained logistic regression model.
        - x_tr (pd.DataFrame or np.array): Training feature set.
        - y_tr (pd.Series or np.array): Training target set.
        - x_te (pd.DataFrame or np.array): Test feature set.
        - y_te (pd.Series or np.array): Test target set.

        Returns:
        - pd.DataFrame: DataFrame containing feature weights, odds ratios, intercept, and accuracies.
        """
    # Extract feature names and weights from the model:
    weight_names = feature_df.columns.values.tolist()
    weight_values = model.coef_[0].tolist()
    intercept = model.intercept_[0]

    # Create rows for intercept and accuracy and append them to the DataFrame:
    intercept_row = pd.DataFrame({'Weights & Metrics': ['Intercept'], 'Values': [intercept]})
    train_accuracy = model.score(x_tr, y_tr)
    train_accuracy_row = pd.DataFrame({
        'Weights & Metrics': ['Train Accuracy'], 'Values': [train_accuracy]})
    test_accuracy = model.score(x_te, y_te)
    test_accuracy_row = pd.DataFrame({
        'Weights & Metrics': ['Test Accuracy'], 'Values': [test_accuracy]})

    # Create DataFrame with feature weights:
    df = pd.DataFrame({'Weights & Metrics': weight_names, 'Values': weight_values})

    # Calculate the odds ratios:
    odds_ratios = np.exp(weight_values)
    df['Odds Ratio'] = odds_ratios

    # Concatenate DataFrames:
    df = (pd.concat([
        df, intercept_row, train_accuracy_row, test_accuracy_row])
          .set_index(keys='Weights & Metrics'))
    # Sort the DataFrame by the "Odds Ratio" column in descending order:
    df = df.sort_values(by='Odds Ratio', ascending=False)
    return df


def summary_metrics_on_new_data(new_data_df, predictions, pred_probabilities):
    """
        Adds model predictions and probabilities to the new data DataFrame.

        Parameters:
        - new_data_df (pd.DataFrame): DataFrame containing the new, unseen data.
        - predictions (np.array): Array of predictions made by the model.
        - pred_probabilities (np.array): Array of prediction probabilities from the model.

        Returns:
        - pd.DataFrame: Updated DataFrame with predictions and probabilities added.
        """
    # Add the model predictions and probabilities to the new data DataFrame
    new_data_df['Predictions'] = predictions
    new_data_df['Not Extended Absenteeism Probability'] = pred_probabilities[:, 0]
    new_data_df['Extended Absenteeism Probability'] = pred_probabilities[:, 1]
    return new_data_df
