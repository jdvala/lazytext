import typing as tt

from rich.console import Console
from rich.table import Table


def create_table(results: tt.Dict, label_analysis: tt.Dict = None):
    """Create summary table for all the results.

    Example:
    Results:
    ```
               Label Analysis
    ┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
    ┃ Classes       ┃ Weights            ┃
    ┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
    │ entertainment │ 0.8725490196078431 │
    │ sport         │ 1.1528497409326426 │
    │ business      │ 1.0671462829736211 │
    │ tech          │ 0.8708414872798435 │
    │ politics      │ 1.1097256857855362 │
    └───────────────┴────────────────────┘
                                                      Result Analysis
    ┏━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
    ┃ Model ┃ Accuracy            ┃ Balanced Accuracy  ┃ F1 Score            ┃ Custom Metric Score ┃ Time Taken        ┃
    ┡━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
    │ SVC   │ 0.23502994011976047 │ 0.2184410069760388 │ 0.10772097568020063 │ NA                  │ 6.808304071426392 │
    └───────┴─────────────────────┴────────────────────┴─────────────────────┴─────────────────────┴───────────────────┘
    ```

    Args:
        results: Dictonary of all the results
        label_analysis: Analysis of the labels
    """
    console = Console()
    if label_analysis:
        label_table = Table(title="Label Analysis")
        label_table.add_column("Classes", justify="left", style="cyan", no_wrap=True)
        label_table.add_column("Weights", justify="left", style="magenta", no_wrap=True)

        for name, weight in label_analysis.items():
            label_table.add_row(str(name), str(weight))

        console.print(label_table)

    result_table = Table(title="Result Analysis")
    result_table.add_column("Model", justify="left", style="cyan", no_wrap=True)
    result_table.add_column("Accuracy", justify="left", style="magenta", no_wrap=True)
    result_table.add_column(
        "Balanced Accuracy", justify="left", style="green", no_wrap=True
    )
    result_table.add_column("F1 Score", justify="left", style="red", no_wrap=True)
    result_table.add_column("Custom Metric Score", justify="left", style="yellow")
    result_table.add_column("Time Taken", justify="left", style="white")

    for result in results:
        temp = []
        for key, value in result.items():
            temp.append(value)

        result_table.add_row(
            str(temp[0]),
            str(temp[1]),
            str(temp[2]),
            str(temp[3]),
            str(temp[4]),
            str(temp[5]),
        )

    console.print(result_table)
