import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


def main():
    file_path = "data/CLR_Imputed_merged_chem.csv"

    df_clr = pd.read_csv(file_path)

    new_elems = [col for col in df_clr.columns if "Imputed New" in col and "CLR" not in col]
    old_elems = [col for col in df_clr.columns if "Imputed Old" in col and "CLR" not in col]

    new_element_names = [elem.replace("Imputed New ", "") for elem in new_elems]
    old_element_names = [elem.replace("Imputed Old ", "") for elem in old_elems]

    common_elements = sorted(set(new_element_names) & set(old_element_names))

    print(f"Common elements between old and new analyses: {common_elements}")
    print(f"Number of common elements: {len(common_elements)}")

    os.makedirs("outputs/figures", exist_ok=True)

    if len(common_elements) == 0:
        print("No common elements found.")
        print(f"New elements found: {new_element_names}")
        print(f"Old elements found: {old_element_names}")
        return

    results = []

    for element in common_elements:
        new_col = f"Imputed New {element}"
        old_col = f"Imputed Old {element}"

        if new_col not in df_clr.columns or old_col not in df_clr.columns:
            print(f"Warning: {new_col} or {old_col} not found.")
            continue

        clean_df = df_clr[[old_col, new_col]].dropna()

        if clean_df.empty:
            print(f"Skipping {element}: no valid data.")
            continue

        slope, intercept, r_value, p_value, std_err = linregress(
            clean_df[old_col],
            clean_df[new_col]
        )

        x_range = np.linspace(clean_df[old_col].min(), clean_df[old_col].max(), 100)
        y_pred = slope * x_range + intercept

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Comparison of Old vs New Analysis: {element}", fontsize=14)

        axes[0].scatter(clean_df[old_col], clean_df[new_col], alpha=0.6, s=20)
        axes[0].plot(
            x_range,
            y_pred,
            linewidth=2,
            label=f"y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value**2:.3f}"
        )
        axes[0].set_xlabel(f"Old {element}")
        axes[0].set_ylabel(f"New {element}")
        axes[0].set_title("Linear Scale")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        positive_df = clean_df[(clean_df[old_col] > 0) & (clean_df[new_col] > 0)]

        if not positive_df.empty:
            axes[1].scatter(positive_df[old_col], positive_df[new_col], alpha=0.6, s=20)
            axes[1].set_xscale("log")
            axes[1].set_yscale("log")
            axes[1].set_xlabel(f"Old {element} (log)")
            axes[1].set_ylabel(f"New {element} (log)")
            axes[1].set_title("Log-Log Scale")
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, "Log scale unavailable\n(non-positive values)", ha="center")
            axes[1].set_title("Log-Log Scale")

        plt.tight_layout()
        plt.savefig(f"outputs/figures/{element}_comparison.png", dpi=300)
        plt.close()

        results.append({
            "Element": element,
            "Slope": slope,
            "Intercept": intercept,
            "R_squared": r_value**2,
            "P_value": p_value,
            "Std_error": std_err
        })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No regression results generated.")
        return

    results_df.to_csv("outputs/regression_results.csv", index=False)

    print("\nLinear Regression Results:")
    print(results_df.round(4))

    print("\nSummary Statistics:")
    print(f"Average R-squared: {results_df['R_squared'].mean():.4f}")
    print(f"Median R-squared: {results_df['R_squared'].median():.4f}")
    print(f"Number of elements with R² > 0.9: {(results_df['R_squared'] > 0.9).sum()}")
    print(f"Number of elements with R² > 0.8: {(results_df['R_squared'] > 0.8).sum()}")

    plt.figure(figsize=(12, 6))
    plt.scatter(results_df["Element"], results_df["R_squared"], alpha=0.7)
    plt.axhline(y=0.9, linestyle="--", label="Excellent: R² > 0.9")
    plt.axhline(y=0.8, linestyle="--", label="Good: R² > 0.8")
    plt.xlabel("Element")
    plt.ylabel("R-squared")
    plt.title("R-squared Values for Old vs New Analysis Comparison")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/figures/r_squared_summary.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(results_df["Slope"], results_df["R_squared"], alpha=0.7)

    for _, row in results_df.iterrows():
        plt.annotate(row["Element"], (row["Slope"], row["R_squared"]), fontsize=8)

    plt.axhline(y=0.9, linestyle="--", alpha=0.5, label="R² = 0.9")
    plt.axhline(y=0.8, linestyle="--", alpha=0.5, label="R² = 0.8")
    plt.axvline(x=1.0, linestyle="--", alpha=0.5, label="Slope = 1.0")
    plt.xlabel("Slope")
    plt.ylabel("R-squared")
    plt.title("Relationship Between Slope and R-squared")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/figures/slope_vs_r_squared.png", dpi=300)
    plt.close()

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
