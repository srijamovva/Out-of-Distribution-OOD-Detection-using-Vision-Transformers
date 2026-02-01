# Out-of-Distribution (OOD) Detection using Vision Transformers
We analyze out-of-distribution detection in Vision Transformers by modeling class-wise latent embeddings and evaluating Mahalanobis distance, softmax confidence, and entropy-based metrics on CIFAR-10, CIFAR-100, SVHN, and Food-101k to assess robustness under distribution shifts.

# Out-of-Distribution (OOD) Detection using Vision Transformers
Overview

Deep learning models typically achieve strong performance when test data follows the same distribution as the training data. However, their reliability degrades significantly when exposed to out-of-distribution (OOD) samples. This limitation poses serious risks in safety-critical applications such as autonomous driving, surveillance systems, and medical image analysis.

Although Transformer-based models, particularly Vision Transformers (ViTs), have demonstrated impressive performance across vision tasks, their robustness to OOD data remains insufficient for real-world deployment. This project investigates the effectiveness of multiple OOD detection metrics applied to Vision Transformer latent representations for image classification tasks.

# Problem Statement

The goal of this work is to distinguish in-distribution (ID) samples from OOD samples by analyzing distributional shifts in the latent feature space learned by a fine-tuned Vision Transformer.

We assume that OOD samples exhibit statistically significant deviations from ID samples in the model’s learned feature representations.

# OOD Detection Metrics

We evaluate three complementary metrics to measure distributional shift:

Mahalanobis Distance in the latent embedding space

Maximum Softmax Confidence Score

Entropy of Softmax Probabilities

These metrics are computed using latent representations extracted from a fine-tuned Vision Transformer.

# Methodology
Feature Modeling

A Vision Transformer is fine-tuned on the CIFAR-10 training set.

Latent feature representations are extracted from the model.

For each class, the mean vector and covariance matrix of embeddings are computed using training samples.

Inference for a Test Sample

For a given test input:

Extract latent features using the Vision Transformer.

Compute the Mahalanobis distance to each class distribution.

Compute softmax probabilities and their entropy.

Classify the sample as OOD if any of the following conditions hold:

Minimum Mahalanobis distance > threshold

Maximum softmax probability < threshold

Entropy > threshold

Otherwise, classify the sample as In-Distribution (ID).

# Datasets

The approach is evaluated using the following datasets:

CIFAR-10 (In-distribution training data)

CIFAR-100

SVHN

Food-101k

# Experimental Setup

Vision Transformer fine-tuned on CIFAR-10 for 30 epochs

Thresholds determined using validation on CIFAR-100

Two latent feature representations evaluated:

CLS token embedding

Mean of all token embeddings

Thresholds

Mahalanobis distance threshold: 2000

Maximum softmax probability threshold: 0.5

Entropy threshold: 1.0

# Results
Experiment 1: Distance + Softmax Confidence

Model 1: CLS token embeddings

Accuracy on CIFAR-10 test set: 96.53%

Model 2: Mean of all token embeddings

Accuracy on CIFAR-10 test set: 96.71%

Experiment 2: Distance + Softmax Confidence + Entropy

Adding entropy as an additional metric did not improve OOD detection performance for either representation.

# Key Findings

Mahalanobis distance and softmax confidence are effective indicators of OOD samples.

Entropy of softmax probabilities is not a reliable OOD metric in this setting.

OOD samples often exhibit softmax probability distributions similar to ID samples, causing entropy-based thresholding to incorrectly classify OOD samples as in-distribution.

Conclusion

This study demonstrates that while Vision Transformers learn meaningful latent representations, entropy-based OOD detection is ineffective due to its dependence on softmax probabilities. In contrast, distance-based metrics in latent space, particularly Mahalanobis distance, provide a more reliable signal for identifying OOD samples. These findings highlight important limitations of commonly used confidence-based uncertainty measures in transformer-based vision models.
