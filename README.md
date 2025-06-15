# Thesis Repository

This repository belongs to Sara García De Fuentes and serves as a companion to the project presented in the Business Analytics and Management master’s thesis. It provides an overview of the methodology, organized according to the main stages described in the thesis document.

## Repository Structure
The repository contains separate scripts organized by stage of the analytical workflow:

1. Double Clustering.py
Performs behavioral segmentation using K‑means and K‑means++.

2. Gradient Boosting Potential Identification.py
Implements demand prediction via Quantile GBM, calculates 90% confidence intervals, and flags clients for growth opportunities.

3. Recommendation Algorithms.py
Generates product-level recommendations based on residuals and cluster logic, with interpretable explanations.


## Repository Scope Disclaimer
Methods included: Core workflows for data processing, demand forecasting, and recommendation generation that reproduce the thesis’s main results.

Methods excluded: Advanced techniques—such as hyperparameter grid searches, extended validation protocols, or alternate algorithms—are described in the thesis but not in these scripts, which prioritize clarity and usability.

Data not included: Input datasets (2023_DATOS_CLUSTERED.xlsx and 2024_DATOS_CLEAN.xlsx) are not provided due to privacy constraints. Please prepare and supply these files according to the formats used in the scripts.
