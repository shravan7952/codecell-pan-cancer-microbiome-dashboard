
---

````markdown
# Codecell.ai — Pan-Cancer Microbiome Dashboard

This repository contains a Streamlit application for exploring microbiome–immune interactions across cancers.  
The dashboard integrates fungal abundance data from the Knight Lab Pan-Cancer Mycobiome study with immune and clinical features from the TCGA Pan-Cancer Atlas (via UCSC Xena).

---

## Features

- Overview of cancer sample distributions
- Bacterial and fungal abundance analysis
- Immune features: scores, checkpoints, cytokines, and cell types
- Clinical survival analysis using Kaplan–Meier stratification
- Dimensionality reduction with PCA and UMAP
- Interactive heatmap explorer with scaling and clustering
- Machine learning playground for classification and feature importance
- Outlier detection using Isolation Forest and Z-score methods

---

## Data

The dataset used in this dashboard is `merged_data_counts.csv`, created by combining:

- **Knight Lab Pan-Cancer Mycobiome dataset**  
  Narunsky-Haziza et al., *Cell* 2022  
  DOI: [10.1016/j.cell.2022.09.005](https://doi.org/10.1016/j.cell.2022.09.005)

- **TCGA Pan-Cancer Atlas (via UCSC Xena)**  
  Goldman et al., *Nucleic Acids Research* 2020

This merged dataset links fungal abundances with immune profiles and clinical survival data across cancers.

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/shravan7952/codecell-pan-cancer-microbiome-dashboard.git
cd codecell-pan-cancer-microbiome-dashboard
pip install -r requirements.txt
````

It is recommended to use a virtual environment or Conda environment.

---

## Usage

Run the application locally:

```bash
streamlit run main.py
```

The app will open in your default browser at `http://localhost:8501`.

---

## Deployment

The app can be deployed on **Streamlit Community Cloud** or any cloud platform supporting Python.

For Streamlit Cloud:

1. Push this repository to GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud), create a new app, and point it to `app.py`.
3. The app will be live with a public link that can be shared.

---

## Citation

If you use this dashboard or dataset, please cite:

* Narunsky-Haziza et al., *Cell*, 2022. Pan-cancer analyses reveal cancer-type-specific fungal ecologies and bacteriome interactions.
* Goldman et al., *Nucleic Acids Research*, 2020. The UCSC Xena platform for cancer genomics data visualization and interpretation.

---

## License

This project is released under the MIT License. You are free to use, modify, and distribute with attribution.

---

© 2025 Codecell.ai — AI-driven cancer research tools.

```

---

```
