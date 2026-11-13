# Industrial Machine Failure Prediction

An MCA-level predictive-maintenance project that trains machine-learning classifiers on the **UCI AI4I 2020 Predictive Maintenance Dataset** and provides a Streamlit web interface for interactive predictions.

## Features

- Trains and compares four classification algorithms:
  - Logistic Regression
  - Random Forest
  - Extra Trees
  - Histogram Gradient Boosting
- Uses a preprocessing pipeline for numeric and categorical features.
- Uses a stratified train/test split.
- Excludes the component failure flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) from the inputs to avoid target leakage.
- Reports accuracy, precision, recall, F1-score, ROC-AUC and average precision.
- Saves the best model and metadata.
- Provides a Streamlit app for entering machine sensor values and predicting failure probability.

## Project structure

```text
industrial-machine-failure-prediction/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── .gitkeep
└── models/
    └── .gitkeep
```

Generated model files and the dataset are excluded from Git by default. Run training after cloning to generate them locally.

## Requirements

- Python 3.10 or newer
- VS Code (recommended)
- Internet connection for the first training run, if you do not already have the dataset

## Run in VS Code

### 1. Download or clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/industrial-machine-failure-prediction.git
cd industrial-machine-failure-prediction
```

Or open the extracted project folder directly in VS Code.

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Train the models

```bash
python train.py
```

On the first run, the script downloads the UCI AI4I 2020 dataset into `data/` using the `certifi` certificate bundle, trains the classifiers, selects the model with the highest average precision, and saves:

- `models/best_machine_failure_model.joblib`
- `models/model_metadata.json`

You can also provide a CSV file yourself:

```bash
python train.py --csv path/to/ai4i2020.csv
```

The CSV must contain a `Machine failure` target column and the expected AI4I sensor columns.

### 5. Start the Streamlit application

```bash
streamlit run app.py
```

Open the local URL printed in the terminal (usually `http://localhost:8501`).

## Push the project to GitHub

Create an empty GitHub repository, then run these commands from the project directory:

```bash
git init
git add .
git commit -m "Initial commit: machine failure prediction project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/industrial-machine-failure-prediction.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username and use your actual repository URL.

**Important:** The `.gitignore` intentionally excludes the dataset and generated model files. Anyone cloning the repository can recreate them by running `python train.py`.

## Dataset

UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset:

https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

The training script downloads the dataset automatically when `data/ai4i2020.csv` is not present.

## Notes

- The failure class is relatively rare in this dataset. Accuracy alone can be misleading; review recall, F1-score, ROC-AUC and average precision.
- The model is an educational prototype. Do not use it as the sole basis for safety-critical maintenance decisions.


### If dataset download reports an SSL certificate error on Windows

1. Check Windows **Settings → Time & language → Date & time** and enable automatic time synchronization.
2. In the activated virtual environment, run `python -m pip install --upgrade certifi`.
3. Retry `python train.py`.
4. If the UCI website's certificate is still expired or a network proxy intercepts HTTPS, download the dataset ZIP using your browser from the UCI dataset page linked above, extract the CSV, and run `python train.py --csv "C:\path\to\ai4i2020.csv"`.

The application does not disable SSL certificate verification.
