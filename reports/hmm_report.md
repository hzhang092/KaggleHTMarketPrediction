# Module: Hidden Markov Model (HMM) for Market Regime Detection

## 1. Objective
The primary objective of this module was to implement a probabilistic, time-aware approach to market regime detection using **Gaussian Hidden Markov Models (HMM)**. Unlike K-Means clustering, which treats each observation independently, HMMs explicitly model the temporal dependencies between market states, capturing both the persistence of regimes and the probability of transitioning between them.

## 2. Motivation: Why HMM Over K-Means?
*   **Temporal Continuity:** Financial markets exhibit strong autocorrelation—a volatile period is likely to be followed by continued volatility. HMMs model this through a **transition matrix** that captures the probability of moving from one regime to another.
*   **Soft Assignments:** Instead of assigning a single "regime label," HMMs can output the **probability distribution** over all regimes for each time point. This provides richer context for downstream models (e.g., "80% in High Volatility, 20% in Neutral").
*   **Causality Preservation:** The forward-backward algorithm used by HMMs respects the sequential nature of time-series data, making it inherently more suitable for financial forecasting.

## 3. Data Preprocessing & Feature Engineering
*   **Feature Detrending:** Applied the same stationarity transformations as in the clustering module:
    *   **Differencing** for price/economic features (P*, E*)
    *   **Rolling Z-score** for volatility and momentum indicators
*   **Imputation:** Used median imputation to handle missing values before feeding data into the HMM.
*   **Window Parameter:** Controlled the lookback period for rolling statistics (default: 50 days).

## 4. Model Architecture & Implementation
*   **Model Type:** `GaussianHMM` from the `hmmlearn` library with diagonal covariance structure (assumes feature independence within each regime).
*   **Key Hyperparameters:**
    *   `n_components`: Number of hidden states (regimes). Tested values: 12, 16, 24.
    *   `covariance_type`: Set to `"diag"` for computational efficiency and to prevent overfitting.
    *   `n_iter`: Maximum number of EM iterations (set to 100).
*   **Robustness Enhancement:** Implemented **multiple random initializations** (`n_inits=3`) to mitigate the sensitivity of the EM algorithm to initial conditions. The best-performing model (based on log-likelihood) was retained.

## 5. Model Selection & Optimization
*   **Criterion:** Used **Bayesian Information Criterion (BIC)** to select the optimal number of regimes. BIC penalizes model complexity, preventing overfitting to training data.
*   **Search Range:** Evaluated models with 2 to `max_regimes` (default: 16) hidden states.
*   **Prediction Modes:**
    *   `predict_type='labels'`: Hard regime assignment (most likely state)
    *   `predict_type='proba'`: Soft assignment (probability vector for all states)

## 6. Experimental Validation
*   **Framework:** Integrated the HMM regime feature into the same **Rolling Forward Analysis** pipeline used for clustering experiments.
*   **Visualization:** Generated time-series plots overlaying regime assignments with actual market returns to qualitatively assess regime persistence and alignment with market behavior.

## 7. Key Results
*   **Performance Improvement:** The HMM-augmented CatBoost model achieved the **best overall performance** across all tested configurations:
    *   **16-State Configuration:** Average daily Sharpe Ratio improved from ~0.024 (baseline) to **~0.044** (83% increase).
    *   **R² Improvement:** Information Coefficient increased from 0.0006 to **0.0053** (8.4x improvement).
    *   **Hit Rate:** Directional accuracy improved from 50.1% to **51.0%**, indicating better signal quality.
*   **Regime Persistence:** Visual analysis confirmed that regimes were temporally stable (i.e., markets stayed in the same regime for multiple consecutive days), validating the HMM's ability to capture true market states rather than noise.
*   **Comparison with K-Means:** HMM consistently outperformed K-Means clustering across all validation folds, demonstrating the value of explicitly modeling temporal transitions.

## 8. Conclusion
The HMM approach successfully identified recurring, interpretable market regimes with strong temporal coherence. By providing probabilistic state information as a feature to the downstream regressor, the model gained the ability to dynamically adjust its predictions based on the current market environment. This **regime-aware forecasting** strategy proved to be the most effective approach tested in this project, significantly outperforming both the baseline model and alternative clustering methods.
