import typing as tt


def create_table(results: tt.Dict, label_analysis: tt.Dict = None):
    """Create summary table for all the results.

    Example:
    Results:
    ```
    | Model                | Accuracy             | Balanced Accuracy    | Time Taken           |
    | -------------------: | -------------------: | -------------------: | -------------------: |
    | MultinomialNB        | 0.641908620301598    | 0.62653122841884     | 0.03511333465576172  |
    ```

    Label Analysis

    Args:
        results: Dictonary of all the results
        label_analysis: Analysis of the labels
    """
    result_format = "| {:<30}| {:<20}| {:<20}| {:<20}| {:<20}| {:<20}|"
    label_format = "| {:<20}| {:<20} |"

    # Class weights
    if label_analysis:
        print("\n Label Analysis")
        print(label_format.format("Classes", "Weights"))
        print("|--------------------:|---------------------:|")
        for name, weight in label_analysis.items():
            print(label_format.format(name, weight))

    # Result
    print("\n Result Analysis")

    print(
        result_format.format(
            "Model",
            "Accuracy",
            "Balanced Accuracy",
            "F1 Score",
            "Custom Metric Score",
            "Time Taken",
        )
    )
    print(
        result_format.format(
            "----------------------------:",
            "-------------------:",
            "-------------------:",
            "-------------------:",
            "-------------------:",
            "-------------------:",
        )
    )
    for result in results:
        temp = []
        for key, value in result.items():
            temp.append(value)
        print(
            result_format.format(temp[0], temp[1], temp[2], temp[3], temp[4], temp[5])
        )
    print("\n")
