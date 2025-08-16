# Essential Gene Prediction using GCNN-SFM

A deep learning approach for identifying essential genes using Graph Convolutional Neural Networks with Sequence Feature Maps (GCNN-SFM).

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Project Overview

This repository contains the implementation of a deep learning model for predicting essential genes across multiple species. The model combines:

- **Gapped k-mer encoding** for sequence feature extraction
- **Graph Convolutional Neural Networks** for learning gene representations
- **Cross-species evaluation** to assess generalizability

The model achieves state-of-the-art performance with an average accuracy of 94.53% across four species.

## Key Features

- 🧬 Sequence feature extraction using gapped k-mer encoding
- 🕸️ Graph-based representation of gene sequences
- 🚀 Multi-layer GCNN architecture
- 📊 Comprehensive evaluation metrics (SN, SP, ACC, AUC)
- 🔄 Cross-species validation
- ⚙️ Hyperparameter tuning framework