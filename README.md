# PredicRNA CLI Tool

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A command-line tool for predicting preeclampsia (PE) risk using cell-free RNA (cfRNA) expression data. This tool uses machine learning models trained on the GSE192902 dataset.

## 📋 Features

- **Binary Classification**: Predict whether a sample is Normal or Preeclampsia (Step 4.4 Model)
- **Severity Assessment**: Evaluate PE severity (Mild/Moderate/Severe) (Step 4.5 Model)
- **Personalized Risk Assessment**: Individual risk scoring using deep learning (Step 9 Model)
- **Risk Scoring**: Calculate overall risk score with clinical recommendations
- **Batch Processing**: Process multiple samples from CSV files

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yangfangs/predicrna-cli.git
cd predicrna-cli
```

### 2. Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using conda
conda create -n predicrna python=3.9
conda activate predicrna
pip install -r requirements.txt
```

### 3. Run Demo

```bash
python predicrna_cli.py demo
```

### 4. Run Your Own Prediction

```bash
# Basic prediction (includes all models: Classification, Severity, Personalized Risk)
python predicrna_cli.py predict --input your_data.csv --output results.csv
```

## 📖 Usage

### Available Commands

| Command | Description |
|---------|-------------|
| `predict` | Run full prediction pipeline (Classification, Severity, Personalized) |
| `demo` | Run demo with sample data |
| `info` | Show system information and required features |
| `validate` | Validate CSV format |

### Predict Command

```bash
# Basic usage
python predicrna_cli.py predict --input data.csv
```

**Output includes:**
- Prediction (Normal/Preeclampsia)
- PE Probability
- Confidence level
- Severity assessment
- Personalized Risk Score (from Step 9 Model)
- Overall Risk score (0-100)
- Clinical recommendation

### Example Output

```
Loading models...
  ✓ Loaded classification (Gaussian Naive Bayes)
  ✓ Loaded severity (Random Forest Classifier)
  ✓ Loaded personalized (Personalized Risk Model)

Loaded 5 samples from data.csv
Prepared 5 samples with 44 features

Processing samples...
--------------------------------------------------
Sample: Control60
  Prediction: Normal
  Probability: 23.7% (Low Confidence)
  Personalized Risk Score: 18.4
  Overall Risk Score: 23.7 (Low Risk)
  Recommendation: Continue routine prenatal care
--------------------------------------------------
Sample: preeclampsia100
  Prediction: Preeclampsia
  Probability: 99.8% (High Confidence)
  Severity: Severe (Score: 0.92)
  Personalized Risk Score: 88.5
  Overall Risk Score: 100.0 (Critical Risk)
  Recommendation: Immediate specialist consultation recommended
--------------------------------------------------
```

## 📁 Input Data Format

Your input CSV file should contain:

1. **Optional**: A `patient_id` or `sample_id` column (will be used as sample identifier)
2. **Required**: 44 cfRNA expression features (Z-score normalized values)

### Required Features (44 markers)

```
CCDC85B, CDC27, CDK6, DDX46, DDX6, GDI2, GP9, GSE1, HIPK1, IRAK3,
ISCA1, LMNA, LPIN2, MAP3K2, MTRNR2L8, Metazoa_SRP.161, NDUFA3, OAZ1,
OSBPL8, PABPN1, PPP1R12A, PSME4, RIOK3, RN7SL128P, RN7SL151P, RN7SL396P,
RN7SL4P, RN7SL5P, RN7SL674P, RN7SL752P, RN7SL767P, RNA5-8SP6, RNA5SP202,
RNA5SP352, RPGR, RPPH1, STAG2, STRADB, TAB2, TBL1XR1, U1.14, UBE2B,
USP34, YOD1
```

### Example CSV Format

```csv
patient_id,CCDC85B,CDC27,CDK6,DDX46,...,YOD1
Sample_001,-0.187,0.912,-0.651,0.314,...,-0.341
Sample_002,-1.345,0.402,-0.929,-0.964,...,-3.454
```

See `examples/sample_data.csv` for a complete example.

## 🧬 Models

| Model | Type | Task | Accuracy |
|-------|------|------|----------|
| Classification | Gaussian Naive Bayes | PE vs Normal | ~85% |
| Severity | Random Forest | Mild/Moderate/Severe | ~80% |
| Temporal | LSTM with Attention | Multi-week prediction | ~84% |

All models were trained on the **GSE192902** dataset.

## 📂 Project Structure

```
cfrna-cli/
├── cfrna_cli.py              # Main CLI script
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation (English)
├── README_CN.md              # Documentation (Chinese)
├── LICENSE                   # MIT License
├── .gitignore               # Git ignore rules
├── setup.py                 # Package setup
├── MANIFEST.in              # Package manifest
├── models/                  # Pre-trained models
│   ├── best_model_gaussian_nb.pkl         # Classification model
│   ├── best_model_random_forest.pkl       # Severity model
│   └── temporal_prediction_model_fixed.h5 # Temporal LSTM model
├── examples/                # Example data
│   ├── sample_data.csv      # Mixed sample data
│   ├── preeclampsia_case.csv # PE case examples
│   ├── normal_case.csv      # Normal case examples
│   └── *_results.csv        # Demo output results
└── data/                    # Additional data
```

## 🔬 Data Preprocessing

Before using this tool, your cfRNA expression data should be:

1. **Log2 transformed** (if raw counts)
2. **Z-score normalized** (subtract mean, divide by std)
3. **Feature selected** to the 44 required markers

Example preprocessing:

```python
import pandas as pd
import numpy as np
from scipy import stats

# Load raw expression data
df = pd.read_csv('raw_expression.csv', index_col=0)

# Log2 transform
df_log = np.log2(df + 1)

# Z-score normalize
df_zscore = df_log.apply(stats.zscore, axis=0)

# Select required features (use 'info' command to see full list)
df_final = df_zscore[required_features]

# Save for prediction
df_final.to_csv('preprocessed_data.csv')
```

## 📊 Risk Score Interpretation

| Risk Score | Risk Level | Recommendation |
|------------|------------|----------------|
| 75-100 | 🔴 Critical | Immediate specialist consultation |
| 50-74 | 🟠 High | Enhanced monitoring and prevention |
| 25-49 | 🟡 Medium | Regular indicator monitoring |
| 0-24 | 🟢 Low | Continue routine prenatal care |

## ⚠️ Disclaimer

This tool is for **research purposes only** and should not be used for clinical diagnosis. Always consult with healthcare professionals for medical decisions.

## 📚 Citation

If you use this tool in your research, please cite:

```bibtex
@article{cfrna_pe_prediction,
  title={cfRNA-based Preeclampsia Prediction using Machine Learning},
  journal={...},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

Made with ❤️ for advancing maternal health research
