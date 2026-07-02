"""Builds notebooks/01_eda.ipynb from Python (guarantees valid notebook JSON).

Run:  python notebooks/_build_eda.py
This is a one-off generator; the resulting .ipynb is the deliverable.
"""
import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

md = new_markdown_cell
code = new_code_cell

cells = [
    md(
        "# Insurance Cost — Exploratory Data Analysis\n\n"
        "**Goal:** understand the drivers of individual medical `charges` before modeling.\n\n"
        "Dataset: Kaggle *Medical Cost Personal* — 7 columns, ~1,338 rows.\n"
        "Features: `age`, `sex`, `bmi`, `children`, `smoker`, `region`; target: `charges`."
    ),
    code(
        "import sys\n"
        "from pathlib import Path\n\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import seaborn as sns\n\n"
        "# Make the src package importable and reuse project config.\n"
        "PROJECT_ROOT = Path.cwd().parent\n"
        "sys.path.append(str(PROJECT_ROOT))\n"
        "from src import config  # noqa: E402\n\n"
        "sns.set_theme(style='whitegrid')\n"
        "FIG = config.FIGURES_DIR\n"
        "FIG.mkdir(parents=True, exist_ok=True)\n\n"
        "df = pd.read_csv(config.DATA_FILE)\n"
        "df.shape"
    ),
    code("df.head()"),
    md("## 1. Structure, types, and data quality"),
    code(
        "df.info()\n"
        "print('\\nMissing values per column:')\n"
        "print(df.isna().sum())\n"
        "print(f'\\nDuplicate rows: {df.duplicated().sum()}')"
    ),
    code("df.describe(include='all').T"),
    md(
        "**Takeaways:** no missing values; one exact duplicate row (dropped during "
        "modeling). `charges` is strongly right-skewed, which we confirm next."
    ),
    md("## 2. Target distribution (raw vs. log)"),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "sns.histplot(df['charges'], kde=True, ax=axes[0], color='steelblue')\n"
        "axes[0].set_title('charges (raw)')\n"
        "sns.histplot(np.log1p(df['charges']), kde=True, ax=axes[1], color='seagreen')\n"
        "axes[1].set_title('charges (log1p)')\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG / 'eda_target_distribution.png', dpi=120)\n"
        "plt.show()"
    ),
    md(
        "The raw target is heavily right-skewed; a log transform makes it far more "
        "symmetric — useful for linear models (tree/boosting models are scale-invariant "
        "to the target)."
    ),
    md("## 3. The dominant driver: smoking"),
    code(
        "fig, ax = plt.subplots(figsize=(7, 5))\n"
        "sns.boxplot(data=df, x='smoker', y='charges', hue='smoker', palette='Set2', "
        "ax=ax, legend=False)\n"
        "ax.set_title('Charges by smoking status')\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG / 'eda_charges_by_smoker.png', dpi=120)\n"
        "plt.show()\n\n"
        "df.groupby('smoker')['charges'].agg(['mean', 'median', 'count'])"
    ),
    md(
        "Smokers pay several times more on average — this is by far the strongest single "
        "predictor."
    ),
    md("## 4. Age and BMI, colored by smoking"),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n"
        "sns.scatterplot(data=df, x='age', y='charges', hue='smoker', alpha=0.6, ax=axes[0])\n"
        "axes[0].set_title('Charges vs. age')\n"
        "sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker', alpha=0.6, ax=axes[1])\n"
        "axes[1].set_title('Charges vs. BMI')\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG / 'eda_age_bmi_scatter.png', dpi=120)\n"
        "plt.show()"
    ),
    md(
        "Charges rise with age in roughly parallel bands. For BMI, the effect is "
        "concentrated in **smokers**: above BMI ~30 (obese) smoker charges jump sharply — "
        "an interaction effect."
    ),
    md("## 5. BMI ≥ 30 × smoker interaction"),
    code(
        "tmp = df.assign(obese=np.where(df['bmi'] >= 30, 'bmi>=30', 'bmi<30'))\n"
        "fig, ax = plt.subplots(figsize=(7, 5))\n"
        "sns.barplot(data=tmp, x='smoker', y='charges', hue='obese', errorbar=None, ax=ax)\n"
        "ax.set_title('Mean charges: smoker x obesity')\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG / 'eda_smoker_bmi_interaction.png', dpi=120)\n"
        "plt.show()\n\n"
        "tmp.groupby(['smoker', 'obese'])['charges'].mean().unstack()"
    ),
    md("## 6. Region, sex, and children"),
    code(
        "fig, axes = plt.subplots(1, 3, figsize=(16, 4))\n"
        "sns.boxplot(data=df, x='region', y='charges', hue='region', palette='pastel', "
        "ax=axes[0], legend=False)\n"
        "axes[0].set_title('by region'); axes[0].tick_params(axis='x', rotation=20)\n"
        "sns.boxplot(data=df, x='sex', y='charges', hue='sex', palette='pastel', "
        "ax=axes[1], legend=False)\n"
        "axes[1].set_title('by sex')\n"
        "sns.boxplot(data=df, x='children', y='charges', hue='children', palette='pastel', "
        "ax=axes[2], legend=False)\n"
        "axes[2].set_title('by number of children')\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG / 'eda_region_sex_children.png', dpi=120)\n"
        "plt.show()"
    ),
    md(
        "Region and sex show only mild differences; number of children has a weak effect. "
        "None rival smoking, age, or BMI."
    ),
    md("## 7. Correlations (numeric + encoded smoker)"),
    code(
        "num = df.copy()\n"
        "num['smoker_yes'] = (num['smoker'] == 'yes').astype(int)\n"
        "corr = num[['age', 'bmi', 'children', 'smoker_yes', 'charges']].corr()\n"
        "fig, ax = plt.subplots(figsize=(6, 5))\n"
        "sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)\n"
        "ax.set_title('Correlation with charges')\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG / 'eda_correlation.png', dpi=120)\n"
        "plt.show()"
    ),
    md(
        "## Summary\n\n"
        "- **Smoking** is the dominant cost driver; **age** and **BMI** follow.\n"
        "- A clear **smoker × BMI≥30** interaction — boosting models capture this "
        "automatically.\n"
        "- `region`, `sex`, `children` are weak predictors.\n"
        "- `charges` is right-skewed; helpful to log-transform for linear models.\n\n"
        "Next: `python -m src.train` fits and compares models and saves the best pipeline."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.13"}

with open("notebooks/01_eda.ipynb", "w", encoding="utf-8") as fh:
    nbf.write(nb, fh)
print("Wrote notebooks/01_eda.ipynb with", len(cells), "cells")
