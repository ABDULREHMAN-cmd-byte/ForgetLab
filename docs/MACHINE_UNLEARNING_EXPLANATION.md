# Machine Unlearning Explanation

Machine unlearning is the process of removing the influence of selected training examples from a model without necessarily retraining from scratch.

## Exact retraining baseline

This is the most reliable reference method. The model is trained on the full dataset, selected rows are removed, and the model is trained again on the retained dataset. This approach is correct and scientifically defensible, but it is costly for large production systems.

## Approximate unlearning experiment

The project also includes an experimental SGD-based approximation. The purpose is to compare an incremental update against exact retraining, not to claim a privacy guarantee. The output should be interpreted as a research experiment with limitations.

## Important caveat

No approximate method should be described as a guaranteed removal of learned information without strong empirical evidence and model-specific validation. In this project, the approximate experiment is labeled as experimental and used only for comparison.
