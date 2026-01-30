# AutoDeal End-to-End Workflow Guide

This guide explains how to analyze a specific car model from *2ememain.be*, from finding ads to identifying good deals and updating the dashboard.

## Prerequisites

Ensure you are in the `AutoDeal` project directory and the virtual environment (if any) is active.
Dependencies: Python 3, Node.js, Playwright.

## Step-by-Step Pipeline

We will use the example of analyzing a **Saab 9-3**.
You should choose a unique "run name" for your analysis, e.g., `runs/saab_93`.

### 1. SEARCH: Collect Ad URLs

Find the search results page on 2ememain.be with your filters (model, year, etc.). Copy the full URL.

**Command:**
```powershell
# Syntax: python deuxiememain_pipeline.py search --dir <RUN_DIR> --url "<URL>"
python deuxiememain_pipeline.py search --dir runs/saab_93 --url "https://www.2ememain.be/l/autos/saab/f/saab-9-3/1163/#f:10882"
```
*Output:* Creates `runs/saab_93/urls.txt`.

### 2. SCRAPE: Extract Car Details

Visit each URL to extract technical details (mileage, power, options, etc.).

**Command:**
```powershell
# Syntax: python deuxiememain_pipeline.py scrape --dir <RUN_DIR> --headless
python deuxiememain_pipeline.py scrape --dir runs/saab_93 --headless
```
*   `--headless`: Runs without opening the browser window (faster).
*Output:* Creates `runs/saab_93/vehicles.jsonl` (and `.csv`).

### 3. TRAIN: Build Price Model

Train a machine learning model specific to this dataset to estimate "fair prices".

**Command:**
```powershell
# Syntax: python deuxiememain_pipeline.py train --dir <RUN_DIR>
python deuxiememain_pipeline.py train --dir runs/saab_93
```
*Output:* Saves model artifacts in `runs/saab_93/model_artifacts/`.

### 4. SCORE: Identify Deals

Apply the model to the ads to calculate "Deal Score" (Underpriced vs Overpriced).

**Command:**
```powershell
# Syntax: python deuxiememain_pipeline.py score --dir <RUN_DIR>
python deuxiememain_pipeline.py score --dir runs/saab_93
```
*Output:* Creates `runs/saab_93/scored.csv`.

---

## Dashboard Update (Manifest)

To make your new analysis visible in the web dashboard/visualizer, you must update the `manifest.json`.

### 5. GENERATE MANIFEST

This script scans the `runs/` folder for any subfolder containing a `scored.csv` and registers it in `manifest.json`.

**Command:**
```powershell
node generate_manifest.mjs
```

*Output:* Updates `manifest.json`.
You can now open `index.html` (or refresh the dashboard) to see "saab_93" in the source list.
