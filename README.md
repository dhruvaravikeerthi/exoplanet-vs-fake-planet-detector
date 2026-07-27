#  Exoplanet vs. Fake Planet Detector
App Link - https://dhruvaravikeerthi-exoplanet-vs-fake-planet-detector-app-paaque.streamlit.app

An AI-powered web app built with **Streamlit**, **XGBoost**, and **SHAP** to help astronomers automatically vet transit signals from the **NASA Kepler Space Telescope** and filter out false positives.

---

##  What It Does

When space telescopes look for exoplanets, they measure tiny drops in a star's brightness (light curves). However, many of these dips aren't caused by planets—they are caused by eclipsing binary stars, background noise, or stellar activity. 

Manually checking every single candidate takes a massive amount of time. This project solves that problem by using a machine learning pipeline to instantly classify Kepler light curve signals into **Confirmed Exoplanet**, **Candidate**, or **False Positive**. 

Instead of treating the AI as a black box, the app uses **SHAP (SHapley Additive exPlanations)** and fundamental orbital mechanics equations to show *why* the model made its prediction.

---

##  Key Features

- **Automated Vetting:** Analyzes observational data against thousands of Kepler transit signals.
- **Explainable AI:** Breaks down exact feature contributions using SHAP force plots so you can see which metrics triggered the verdict.
- **Physics Explanations:** Runs physics calculations (like transit depth ratios and planetary radius limits) to verify if a candidate exceeds physical planet boundaries.
- **Interactive UI:** A simple Streamlit dashboard where you can tweak transit parameters in real time and see how predictions shift.

---

##  Built With

- **Frontend:** Streamlit
- **Machine Learning:** XGBoost, Scikit-learn
- **Model Explainability:** SHAP, Matplotlib
- **Data & Math:** Pandas, NumPy
- **Model Storage:** Joblib

---

##  Key Transit Parameters Tracked

| Parameter | Code | Unit | What it measures |
| :--- | :--- | :--- | :--- |
| **Planetary Radius** | `koi_prad` | $R_\oplus$ (Earth Radii) | Estimated size of the planet |
| **Transit Depth** | `koi_depth` | ppm | How much light the object blocks during transit |
| **Orbital Period** | `koi_period` | Days | Days taken to complete one orbit |
| **Transit Duration** | `koi_duration` | Hours | How long the transit dip lasts |
| **Impact Parameter** | `koi_impact` | - | Trajectory across the star's disk |
| **Equilibrium Temp** | `koi_teq` | Kelvin ($K$) | Surface temperature estimate |

---

