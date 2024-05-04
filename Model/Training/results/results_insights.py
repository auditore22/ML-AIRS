import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway


def results_insights(results):
    if not results:
        print("No results to display.")
        return

    try:
        # Assuming 'results' is a list of lists, where each sublist contains evaluation metrics for a fold
        results_df = pd.DataFrame(results, columns=['loss', 'accuracy'])
    except Exception as e:
        print(f"Error processing results data: {e}")
        return

    # Compute summary statistics and additional stats
    summary_stats = results_df.describe()
    variance = results_df.var()
    print("Summary Statistics for Each Metric Across Folds:")
    print(summary_stats)
    print("\nVariance for Each Metric Across Folds:")
    print(variance)

    # Plotting the results
    plt.figure(figsize=(14, 10))

    # Boxplot for each metric
    colors = ['blue', 'green', 'red', 'purple']
    for i, col in enumerate(results_df.columns):
        plt.subplot(2, 2, i+1)
        results_df.boxplot(column=col, grid=False, color=colors[i % len(colors)])
        plt.title(f'Distribution of {col}')
        plt.ylabel(col)
    plt.tight_layout()
    plt.show()

    # Histograms of evaluation metrics
    plt.figure(figsize=(14, 10))
    for i, col in enumerate(results_df.columns):
        plt.subplot(2, 2, i + 1)
        results_df[col].hist(bins=10, alpha=0.75, edgecolor='black')
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # Additional Statistical Analysis (e.g., ANOVA for significant differences)
    # Assuming that each column can be treated as an independent sample for ANOVA
    f_value, p_value = f_oneway(results_df['loss'], results_df['accuracy'])
    print(f"ANOVA results: F-value = {f_value}, p-value = {p_value}")

    if p_value < 0.05:
        print("There are statistically significant differences between the metrics.")
    else:
        print("No significant differences found between the metrics.")
