Market Regime Detection Through Clustering Analysis
Overview
This analysis explored the hypothesis that financial markets exhibit distinct, identifiable behavioral regimes that can be leveraged to improve return predictions. The work progressed through two main approaches: initial K-Means clustering and a more sophisticated Hidden Markov Model (HMM) implementation, ultimately demonstrating that regime-based features can enhance predictive model performance.

Methodology
Data Preprocessing
The analysis began with 90 features from the Hull Tactical market prediction dataset, categorized into eight groups: Market Dynamics (M), Macro Economic (E), Interest Rate (I), Price Valuation (P), Volatility (V), Sentiment (S), Momentum (MOM), and Dummy Binary (D) features. Missing values exceeding 50% threshold were removed, and remaining gaps were filled using zero imputation.

Initial K-Means Clustering Approach
The first clustering attempt employed K-Means algorithm with standardized features across all categories. The optimal cluster count was determined using silhouette score analysis, which identified 18 clusters as optimal (silhouette score: 0.2284). The clusters successfully differentiated between distinct market conditions:

High-risk regimes (Clusters 7, 17, 11): Strongly negative returns with elevated volatility
Growth regimes (Clusters 4, 12, 15): Positive returns with favorable risk-adjusted performance (Sharpe ratios: 0.09-0.13)
Stable regimes (Clusters 2, 13, 16): Low volatility with consistent positive returns
However, temporal analysis revealed a critical limitation: clusters appeared sequentially without recurrence, indicating the algorithm had identified historical eras rather than repeating market states. This "era clustering" occurred because non-stationary features with long-term trends caused temporally adjacent observations to group together.

Feature Detrending and Stationarity
To address the non-stationarity issue, a comprehensive detrending strategy was implemented:

Price and economic series (P*, E* features): Converted to percentage changes using .diff() to capture relative movements
Technical indicators (V*, MOM*, S* features): Transformed using rolling Z-scores (20-period window) to normalize around zero mean and unit variance
Binary features: Retained without transformation
This transformation ensured that clusters would represent market states (high volatility, momentum shifts, etc.) rather than historical time periods.

Hidden Markov Model Implementation
Recognizing that market regimes exhibit temporal dependencies and smooth transitions, I transitioned from K-Means to a Hidden Markov Model framework. The HMM implementation included:

Multiple initializations (n_inits=3): To mitigate local optima issues inherent to EM algorithm convergence
BIC-based model selection: Automated selection of optimal regime count by minimizing Bayesian Information Criterion
Regime probability outputs: Rather than hard assignments, the model generated probability distributions across regimes for each observation, capturing regime uncertainty
Diagonal covariance structure: Balanced model flexibility with computational efficiency
The HMM approach provided several advantages over K-Means: capturing temporal persistence of regimes, allowing soft regime assignments, and modeling transition probabilities between states.

Model Evaluation
Both the baseline model (CatBoost regressor with 75 top features) and the enhanced model (baseline + HMM regime features) were evaluated using a rigorous rolling-forward validation framework:

Initial training period: 50% of data (4,495 samples)
Test windows: 10% intervals (899 samples each)
Progressive retraining: Models were retrained at each forward step to simulate realistic deployment
Additionally, a final holdout test was conducted on the last 180 trading days to assess out-of-sample generalization.

Results
The HMM regime-enhanced model demonstrated measurable improvements over the baseline:

Rolling-forward validation metrics:

Average daily Sharpe ratio increased from 0.0276 to 0.0285 (+3.3%)
R² score improved from 0.0012 to 0.0017 (+41.7%)
180-day holdout test:

Sharpe ratio improvement: Baseline achieved 0.0123, enhanced model achieved 0.0134
R² improvement: +27.6%
Directional accuracy (hit rate): Marginally improved prediction of return direction
Equity curve analysis showed that the regime-enhanced strategy achieved superior cumulative returns with smoother drawdown profiles, particularly during volatile market periods where regime identification proved most valuable.

Key Insights
Regime persistence matters: The HMM's ability to model temporal dependencies captured the reality that markets tend to remain in regimes for extended periods rather than switching randomly.

Soft assignments reduce noise: Probability-based regime assignments prevented the model from making overconfident decisions during regime transitions.

Feature stationarity is critical: The dramatic improvement from non-stationary to stationary features underscores that clustering should identify recurring behavioral patterns, not historical eras.

Incremental but consistent gains: While improvements were modest in magnitude, they were consistent across multiple validation windows, suggesting robust generalization rather than overfitting.

Conclusion
This analysis successfully demonstrated that market regime detection through HMM can provide valuable contextual information to enhance return prediction models. The transition from naive K-Means clustering to detrended HMM features represents a principled approach to incorporating market state information while maintaining model generalizability to future, unseen market conditions.