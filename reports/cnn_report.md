Module: Deep Learning Signal Generation via Temporal Convolutional Networks (TCN)
1. Objective
The objective of this module was to explore the efficacy of deep learning architectures—specifically Temporal Convolutional Networks (TCN)—in capturing long-range temporal dependencies and complex non-linear patterns in market data that traditional tree-based models might overlook.

2. Target Engineering & Transformation

Concept: Instead of predicting raw returns directly, the model was trained to predict an "Ideal Asset Allocation" (ranging from 0.0 to 2.0, representing 0% to 200% exposure).
Mapping Strategies: We experimented with two distinct functions to map forward returns to this target allocation:
Sigmoid Mapping: A direct non-linear mapping where positive returns asymptotically approach 2.0 and negative returns approach 0.0.
Volatility-Scaled Mapping: A risk-adjusted approach where returns were first normalized by their realized volatility (using a tanh function) to penalize high-risk periods.
Observation: The simpler Sigmoid Mapping proved to be more robust, leading to more stable model convergence and better alignment with the loss function compared to the volatility-scaled approach.
3. Model Architecture

Architecture: Implemented a Temporal Convolutional Network (TCN) using PyTorch.
Key Components:
Dilated Causal Convolutions: Used to exponentially expand the model's receptive field (history) without violating causality (i.e., no look-ahead bias).
Residual Blocks: Incorporated residual connections and weight normalization to facilitate gradient flow and training stability.
Training Configuration: Utilized SmoothL1Loss (Huber Loss) to minimize sensitivity to market outliers and employed the AdamW optimizer with a OneCycleLR scheduler for efficient convergence.
4. Experimental Evaluation

Framework: Applied the same rigorous Step-Forward Validation (Walk-Forward) methodology used in the clustering analysis to ensure comparable results.
Input Structure: Data was processed into 3D tensors (Batch 
×
× Features 
×
× Time Window) to preserve sequential information.
5. Key Results

Performance vs. Baseline: While the TCN was able to learn directional signals and achieved a positive Sharpe Ratio in several validation folds, it exhibited higher variance in performance compared to the tree-based models.
Comparison with Cluster-CatBoost: The TCN approach ultimately underperformed the Cluster-Augmented CatBoost model. The
deep learning model struggled to generalize as effectively on this tabular dataset, suggesting that for this specific problem, the regime-aware ensemble tree approach provides a superior signal-to-noise ratio.