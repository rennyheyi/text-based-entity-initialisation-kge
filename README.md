# Text-Based Entity Initialisation for Knowledge Graph Link Prediction

This repository contains the code used for the bachelor thesis
"Text-Based Entity Initialisation for Knowledge Graph Link Prediction:
Effects on Ranking and Predictive Confidence."

## Experimental Setup

- Datasets: FB15k-237, WN18RR, and CoDEx-M
- Models: TransE and DistMult
- Conditions: random, correct text, and shuffled text
- Text encoder: sentence-transformers/all-MiniLM-L6-v2
- Seeds: five paired seeds

## Repository Structure

- `src/`: main implementation
- `scripts/`: experiment scripts
- `configs/`: experiment configurations
- `results/`: final aggregated results
- `data/README.md`: data preparation instructions

## Installation

```bash
pip install -r requirements.txt
