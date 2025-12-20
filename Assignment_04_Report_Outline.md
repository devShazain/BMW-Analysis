# Assignment 4: Deployment Report

**Student Name:** [Your Name]
**Course:** Data Science (BSCS-F22)  
**Instructor:** Ghulam Ali
**Learning Outcome:** CLO-03, GA-03

---

## I. Introduction
**Purpose of the Assignment:**
The primary objective of this assignment was to transition from model development to a real-world deployment. The project mimics a production lifecycle where a Random Forest Regressor (trained to predict BMW car prices) is deployed as an interactive web application. This allows end-users to estimate car values based on features like Mileage, Year, and Model.

**Summary of Deployed System:**
- **Model:** Random Forest Regressor (Optimized for Price Prediction)
- **Interface:** Web-based Dashboard (Streamlit)
- **Key Features:** Real-time prediction, feedback collection for model retraining, and outlier handling.

## II. Deployment Tool Comparison
To ensure a robust deployment, three frameworks were evaluated. **Streamlit** was selected as the optimal choice.

| Feature | **Streamlit** (Selected) | **Flask** | **Gradio** |
| :--- | :--- | :--- | :--- |
| **Why Chosen?** | Native Python support, built-in interactive widgets, and instant deployment capability. | Too much boilerplate code needed for simple UI. | Testing interface is good, but harder to customize full dashboards. |
| **Ease of Use** | High (Pure Python) | Medium (Requires HTML/CSS) | High |
| **Customization** | High (Layouts, Sidebar) | Unlimited (Full Web Dev) | Limited |

## III. Deployment Process
The deployment followed a 3-step pipeline:

### Step 1: Model Persistence (Serialization)
- **Action:** The trained model (Random Forest) and all categorical encoders (for 'Model', 'Fuel', etc.) were serialized using `joblib`.
- **Artifact:** `bmw_price_model.pkl` (Contains Model + Encoders + Feature List).
- **Reasoning:** Saving the model ensures we do not need to retrain it for every single user request, drastically improving response time.

### Step 2: Interface Development (`app.py`)
- **Tools:** Python 3.13, Streamlit, Pandas.
- **Workflow:**
    1.  App loads `bmw_price_model.pkl` on startup.
    2.  User inputs data via Dropdowns (Models) and Sliders (Mileage).
    3.  `Predict` button triggers the inference engine.
    4.  Result is displayed instantly with error handling for realistic inputs.

### Step 3: Execution
To run the system locally, the following command is used:
```bash
streamlit run app.py
```
**Link to Application:** [Insert Localhost Link or Cloud Link if deployed]

### Screenshots
*(Please insert screenshots of your running App here)*
1.  **Main Interface:** Showing the form inputs.
2.  **Prediction Result:** Showing the calculated price.

## IV. Feedback Collection & Analysis
A feedback loop was integrated directly into the app to gather insights from 15+ users.

- **Storage:** `suggestions_dataset.csv`
- **Collected Metrics:** Usability Rating (1-5), Accuracy Perception, Qualitative Comments.

### Analysis of Feedback
| Recurring Theme | User Suggestion |
| :--- | :--- |
| **Usability** | "Add a currency converter (USD to PKR)." |
| **Accuracy** | "Predictions for very old cars (2000s) seem inaccurate." |
| **Feature Request** | "Include more car brands beyond BMW." |

## V. Improvement Plan (v2.0)
Based on the feedback, the roadmap for **Version 2.0** is defined below:

### 1. Prioritized Improvements
-   **Add "Trim/Edition" Feature:** Users noted that specific M-Sport packages affect price significantly. We will scramble more data to include this.
-   **Dark Mode Toggle:** To improve accessibility and aesthetics.

### 2. Version 2.0 Definition
-   **Release Date:** Jan 2026
-   **Scope:** Expanded dataset (2000-2024), Deployment to Streamlit Cloud, and REST API endpoints for mobile integration.

## VI. Limitations & Assumptions
### Assumptions
1.  **Market Stability:** We assume the 2010-2024 pricing trends remain valid despite inflation.
2.  **Dataset Integrity:** We assume the "clean" dataset is free of fraudulent listings.

### Limitations
1.  **Generalizability:** The model is trained *only* on BMW data. It cannot predict prices for Audi or Mercedes (poor generalizability outside domain).
2.  **Bias in Data:** The dataset is heavily skewed towards "Diesel" engines (common in Europe). The model might underestimate the price of rare "Hybrid" or "Electric" models due to lack of representation (Bias).
3.  **Deployment:** Currently local-only; scaling to thousands of users would require a cloud backend (AWS/Azure).

## VII. Conclusion
This project bridged the gap between theoretical Machine Learning and practical application. By deploying the Random Forest model with Streamlit, we successfully created a tool that provides value to used car buyers. The feedback loop highlights that while the model is accurate for common cars, edge cases (rare models) require more data, setting the stage for future improvements.

## VIII. Appendix
-   **Code:** `app.py` (Deployment), `Bmw.py` (Pipeline).
-   **Dataset:** `bmw_sales_cleaned.csv`
-   **Feedback:** `suggestions_dataset.csv`
