# Quant Lab Strategy Pipeline

## Overview
This repository contains the `StrategyPipeline` core logic and backtesting engine for the Quant Lab. It is designed to be portable and reproducible.

## Setup
To set up the environment on a new machine (Windows):

1.  **Run Setup Script**: 
    Double-click `setup_project.bat` or run it from the terminal:
    ```powershell
    .\setup_project.bat
    ```
    This will:
    *   Check for Python.
    *   Create a virtual environment (`.venv`).
    *   Install dependencies from `requirements.txt`.
    *   Create necessary data folders (`data/`, `logs/`, etc.).

2.  **Configuration**:
    *   Rename `.env.example` to `.env`.
    *   Add your API keys (e.g., Polygon.io) to `.env`.

## Usage
Activate the environment:
```powershell
.venv\Scripts\activate
```

Run the pipeline:
```powershell
python run_experiments.py
```

## Directory Structure
*   `StrategyPipeline/`: Core source code.
*   `data/`: Local data storage (ignored by git).
*   `logs/`: Execution logs (ignored by git).
*   `outputs/`: Backtest artifacts (ignored by git).
