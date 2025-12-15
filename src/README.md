# SRC

Brief overview

- Implements core data handling, feature transformation, training orchestration, model registry, and a minimal API for serving models for the credit-risk probability project.

Contents and responsibilities

- data_manager.py

  - Data loading from csv to a data frame, handles loading both clean data and raw data
  - Data saving to csv

- data_pipeline.py

  - End-to-end data preparation pipeline. Applies sequence of cleanings, encodings, and transformations to produce model-ready features. Orchestrates calls to transformers and the data manager.

- woe_transformer.py

  - Weight-of-Evidence (WoE) transformer implementation and related encoding utilities. Fit/transform API that computes WoE per bin/category and can be persisted for inference.

- api/

  - main.py — Minimal FastAPI server entrypoint (serves predictions endpoint).
  - model_loader.py — Model artifact loader and helper to prepare production models
  - pydantic_models.py — Request/response schemas (input validation and typed outputs) used by the API.

- registry/

  - model_registry.py — Simple model registry abstraction: register, list, load model artifacts and metadata; may track versions/paths.

- training/
  - train.py — Single-experiment training script: loads data, configures model, fits, evaluates, and stores artifacts.
  - experiment_runner.py — Higher-level orchestration for experiments
