# Reference Papers

This page collects research papers that are closely related to the anomaly-detection workflow in this repository.

## How To Use This List

- Start with the surveys if you want the big-picture view.
- Read the algorithm-specific papers next if you want to understand the model stack used in the code.
- Use the notes to match each paper to the part of the project it supports.

## Surveys And Overviews

- [Deep Learning for Medical Anomaly Detection -- A Survey](https://arxiv.org/abs/2012.02364)  
  Good background for the healthcare setting and for understanding why reconstruction, one-class, and transformer-style anomaly detectors are used together.

- [Deep Learning for Anomaly Detection: A Survey](https://arxiv.org/abs/1901.03407)  
  A broad deep-learning survey that organizes anomaly detection methods by assumption and training style.

- [A Unifying Review of Deep and Shallow Anomaly Detection](https://arxiv.org/abs/2009.11732)  
  Useful for connecting classical methods with deep one-class and reconstruction-based methods.

- [PyOD: A Python Toolbox for Scalable Outlier Detection](https://arxiv.org/abs/1901.01588)  
  Helpful background for the classical anomaly-detection families used in tabular pipelines.

## Classical Detectors

- [Anomaly Detection Based on Isolation Mechanisms: A Survey](https://arxiv.org/abs/2403.10802)  
  Supports the Isolation Forest style of isolation-based scoring used in the ensemble.

- [Extended Isolation Forest](https://arxiv.org/abs/1811.02141)  
  A good companion paper for understanding the limitations and score geometry of isolation-based methods.

- [A One-Class Support Vector Machine Calibration Method for Time Series Change Point Detection](https://arxiv.org/abs/1902.06361)  
  Useful for understanding how one-class SVMs are tuned when labeled anomalies are scarce.

- [Automatic Hyperparameter Tuning Method for Local Outlier Factor, with Applications to Anomaly Detection](https://arxiv.org/abs/1902.00567)  
  Useful for understanding LOF behavior and sensitivity to neighborhood settings.

- [Deep Isolation Forest for Anomaly Detection](https://arxiv.org/abs/2206.06602)  
  A useful modern companion for the isolation-based family used in the ensemble.

## Reconstruction-Based Detectors

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)  
  The core VAE paper behind probabilistic latent-variable autoencoders.

- [An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691)  
  A cleaner explanation of VAE training, reparameterization, and latent regularization.

- [GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training](https://arxiv.org/abs/1805.06725)  
  The main reference for encoder-decoder-encoder anomaly scoring.

- [Skip-GANomaly: Skip Connected and Adversarially Trained Encoder-Decoder Anomaly Detection](https://arxiv.org/abs/1901.08954)  
  A closely related follow-up that is useful for understanding latent-consistency scoring.

- [A Real-time Anomaly Detection Using Convolutional Autoencoder with Dynamic Threshold](https://arxiv.org/abs/2404.04311)  
  Useful background for thresholded convolutional reconstruction methods.

- [DFR: Deep Feature Reconstruction for Unsupervised Anomaly Segmentation](https://arxiv.org/abs/2012.07122)  
  Shows how convolutional reconstruction can be used for anomaly detection.

## Transformer And Attention-Based Methods

- [Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy](https://arxiv.org/abs/2110.02642)  
  The main reference behind attention-discrepancy style anomaly scoring.

- [Attention Is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need)  
  Foundational transformer paper that explains the attention mechanism used by transformer-style models.

## Deep One-Class And Hypersphere Methods

- [Deep Semi-Supervised Anomaly Detection](https://arxiv.org/abs/1906.02694)  
  Useful for understanding deep one-class learning and hypersphere-style normality modeling.

- [DROCC: Deep Robust One-Class Classification](https://arxiv.org/abs/2002.12718)  
  Helpful for understanding modern deep one-class methods and the limitations of collapse-prone training.

- [Explainable Deep One-Class Classification](https://arxiv.org/abs/2007.01760)  
  A strong companion paper for deep one-class representation learning and explanation ideas.

## Recommended Reading Path

1. Read the three survey papers first.
2. Read the algorithm-specific paper for the detector you are modifying.
3. Return to the model details documentation in [`model_details.md`](model_details.md) to map the theory back to the code.

## Notes

- The repository implements lightweight, tabular-friendly versions of several methods, so the papers explain the algorithmic family even when the code simplifies the original architecture.
- If you want, this page can be expanded into a bibtex-style bibliography next.
