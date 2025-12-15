# Tests

## Overview

Contains unit and integration tests for the credit-risk-probability project. Use pytest to run tests and verify changes.

## Requirements

- dev dependencies: pytest, pytest-cov (install via requirements-dev.txt or pip)

## Setup

1. Create and activate a virtual environment:
   - python -m venv .venv
   - source .venv/bin/activate (macOS/Linux) or .venv\Scripts\activate (Windows)
2. Install dev dependencies:
   - pip install -r ../requirements-dev.txt

## Run tests

- Run all tests:
  - pytest -v

## Test layout (files under tests/)

- test_data_processing.py — tests for csv loading and saving from and to csv
- test_data_processing.py — tests for feature engineering by using sample df, transforming WOE and IV

## CI

Repository runs pytest on push/PR (GitHub Actions)
