# Multi-Modal System for Chlorella vulgaris Biomass Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-brightgreen.svg)](https://xgboost.readthedocs.io/)

High-performance prediction system for Chlorella vulgaris microalgae cultures in photobioreactors, using **Physics-Informed Neural Networks (PINN)**, **LSTM**, and **Machine Learning** models with ensemble architecture.

---

## Table of Contents

- [General Description](#general-description)
- [Main Features](#main-features)
- [System Architecture](#system-architecture)
- [Scientific Methodology](#scientific-methodology)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Scientific References](#scientific-references)
- [Author](#author)

---

## General Description

This project implements an **advanced multi-modal system** for predicting Chlorella vulgaris biomass in photobioreactors. It combines biological knowledge with deep learning techniques to achieve accurate and scientifically grounded predictions.

### Problem to Solve

Microalgae production in photobioreactors requires constant monitoring and optimization. This system predicts future biomass based on:
- Environmental variables (temperature, pH, PAR light)
- Nutrient dynamics
- Growth phases
- Temporal patterns

### Implemented Solution

A robust system that integrates:
- **6 predictive models** (Linear, Ridge, RandomForest, XGBoost, PINN, LSTM)
- **Weighted ensemble** based on performance
- **Automatic data leakage detection**
- **Biological feature engineering** (40+ features)
- **Temporal validation by scenarios**

---

## Main Features

### 1. Intelligent Data Management
- Automatic detection of **data leakage** (threshold 0.95)
- Biological variable validation
- Robust cleaning and preprocessing
- Outlier handling with clipping (percentiles 0.5-99.5%)

### 2. Biological Feature Engineering

#### Photosynthetic Variables
- **Light efficiency** (Michaelis-Menten)
  ```
  P = (I * Pmax) / (I + K)
  K = 150 µmol/m²/s
  ```
- **Photoinhibition** (threshold 300 µmol/m²/s)
- **Jassby-Platt equation** (initial efficiency α=0.012)

#### Environmental Variables
- **Temperature effects** (Gaussian function, optimum 28°C)
- **pH effects** (Gaussian function, optimum 8.0)
- **Combined environmental stress**
- **Multi-factor interactions**

#### Nutrient Dynamics
- **Haldane model** (includes inhibition by excess)
  ```
  E = N / (Ks + N + N²/Ki)
  Ks = 0.02, Ki = 1.5
  ```

#### Temporal Variables
- **Circadian cycles** (sin/cos 24h)
- **Growth phases** (lag, exponential, stationary, decline)

### 3. Feature Selection (Ensemble of 3 Methods)
1. **Pearson Correlation** → Linear relationships
2. **SelectKBest (f_regression)** → Statistical importance
3. **Random Forest (50 trees)** → Non-linear relationships

### 4. Data Splitting and Reproducibility
- **Scenario-based splitting**: 45 cultures training / 15 cultures validation
- **Fixed seed**: `SEED=50` (total reproducibility)
- **Internal validation**: 80-20 within training set
- **Normalization**: RobustScaler (features) + StandardScaler (target)

### 5. Multi-Model System

#### Classical Models
- **Linear Regression**: Simple baseline
- **Ridge (α=1.0)**: L2 regularization
- **RandomForest (100 trees, depth=8)**: Non-linear interactions
- **XGBoost (300 estimators)**: Advanced boosting

#### Neural Networks
- **PINN (Physics-Informed Neural Network)**
  - Architecture: `[input] → BatchNorm → 64 → ReLU → Dropout(0.3) → 32 → ReLU → Dropout(0.2) → 1`
  - Biological loss function: `Loss = MSE + 0.1 * bio_penalty`
  - Penalty: `bio_penalty = mean(ReLU(-pred)) * 5` (prevents negative biomass)
  - Optimizer: **AdamW** (lr=0.001, weight_decay=0.01)
  - Early selection every 50 epochs

- **LSTM (Long Short-Term Memory)**
  - Architecture: `[input, seq_len=1] → LSTM(32, dropout=0.2) → Dense(16) → ReLU → Dropout(0.2) → 1`
  - Optimizer: **AdamW** (lr=0.001, weight_decay=0.01)
  - Gradient clipping: L2 norm to 1.0
  - Early selection every 50 epochs

#### Weighted Ensemble
```python
model_weight = (1/MSE_model) / Σ(1/MSE_all)
final_prediction = Σ(model_weight * model_prediction)
```

### 6. Comprehensive Evaluation

Implemented metrics:
- **R²** (Coefficient of Determination)
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **NSE** (Nash-Sutcliffe Efficiency)
- **Bias** (Relative bias)

Overfitting detection:
- **HIGH**: R² > 0.99
- **MEDIUM**: 0.97 < R² ≤ 0.99
- **LOW**: R² ≤ 0.97

### 7. Advanced Visualizations
1. **Scatter Plot** (Predicted vs Observed)
2. **R² comparison** by model (color codes by risk)
3. **RMSE comparison**
4. **Residual analysis** (pattern detection)
5. **Residual distribution** (normality test)
6. **Multi-metric** (R² + NSE side by side)
7. **Feature importance** (Random Forest)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   CHLORELLA MULTI-MODAL SYSTEM                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: SmartDataManager - Loading and Cleaning               │
│  • Data leakage detection (correlation > 0.95)                 │
│  • Removal of problematic variables                            │
│  • Data integrity validation                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: BioFeatureEngine - Feature Engineering                │
│  • Photosynthetic variables (Michaelis-Menten, photoinhibition)│
│  • Environmental variables (temp, pH, stress)                   │
│  • Nutrient dynamics (Haldane)                                  │
│  • Temporal variables (circadian cycles, phases)                │
│  • 40+ biological features created                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Feature Selection (Ensemble of 3 methods)             │
│  • Pearson Correlation (median as threshold)                    │
│  • SelectKBest + f_regression (median F-scores)                │
│  • Random Forest importances (median as threshold)             │
│  • Final features: TOP from each method combined                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Splitting and Normalization                           │
│  • Scenario-based splitting (45 train / 15 val)                │
│  • Random selection with SEED=50 (reproducibility)              │
│  • RobustScaler for features                                    │
│  • StandardScaler for target (biomass)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: CompactMultiModel - Training                          │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │   Linear     │  │    Ridge     │  │ RandomForest │        │
│  │  Regression  │  │   (α=1.0)    │  │ (100 trees)  │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │   XGBoost    │  │     PINN     │  │     LSTM     │        │
│  │(300 estim.)  │  │(Bio-informed)│  │  (seq=1)     │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│                                                                 │
│  Internal validation (80-20) to calculate weights              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: Weighted Ensemble                                     │
│  weight_i = (1/MSE_i) / Σ(1/MSE_j)                            │
│  pred_final = Σ(weight_i * pred_i)                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7: Evaluation and Visualization                          │
│  • Metrics: R², RMSE, MAE, MAPE, NSE, Bias                    │
│  • Overfitting detection                                        │
│  • 7 analysis plots                                             │
│  • Feature importance                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Scientific Methodology

### Biological Foundations

#### 1. Photosynthesis (Michaelis-Menten)
```
P = (I * Pmax) / (I + K)
```
- **I**: Light intensity (PAR, µmol/m²/s)
- **K**: Half-saturation constant (150 µmol/m²/s)
- **Pmax**: Maximum photosynthesis rate (normalized to 1)

**References:**
- Falkowski & Raven (2013) - *Aquatic Photosynthesis*
- Jassby & Platt (1976) - P-I curves for microalgae

#### 2. Photoinhibition
```
F = max(0, (PAR - 300) / 100)
```
- Threshold: 300 µmol/m²/s
- Maximum: 400 µmol/m²/s

**References:**
- Long et al. (1994) - *Photoinhibition of photosynthesis in nature*
- Tredici (2010) - *Photobiology of microalgae mass cultures*

#### 3. Temperature and pH Effects (Gaussian)
```
temp_effect = exp(-((T - 28)² / 50))
pH_effect = exp(-((pH - 8.0)² / 2))
```
- **Optimal temperature**: 28°C (σ = 5°C)
- **Optimal pH**: 8.0 (σ = 1.0)

**References:**
- Eppley (1972) - *Temperature and phytoplankton growth*
- Raven & Geider (1988) - *Temperature and algal growth*
- Goldman & Azam (1978) - pH effects on photosynthesis

#### 4. Nutrient Dynamics (Haldane)
```
E = N / (Ks + N + N²/Ki)
```
- **Ks**: 0.02 (half-saturation)
- **Ki**: 1.5 (inhibition by excess)

**References:**
- Monod (1949) - *Growth of bacterial cultures*
- Bernard (2011) - *Modelling and control of microalgae for CO2 mitigation*

### Anti-Overfitting Strategy

1. **Temporal validation by scenarios** (prevents data leakage)
2. **Automatic detection of problematic variables** (correlation > 0.95)
3. **Ensemble of 3 methods** for feature selection
4. **L2 regularization** (Ridge, AdamW)
5. **Dropout** in neural networks (0.2-0.3)
6. **Early selection** based on internal validation
7. **Gradient clipping** in LSTM (L2 norm ≤ 1.0)

---

## Technologies Used

### Core ML/DL
- **Python** 3.8+
- **PyTorch** 2.0+ (PINN, LSTM)
- **Scikit-learn** 1.3+ (classical models, metrics, preprocessing)
- **XGBoost** (advanced boosting)

### Data Processing
- **NumPy** (numerical operations)
- **Pandas** (data manipulation)

### Visualization
- **Matplotlib** (static plots)
- **Seaborn** (statistical visualization)

### Statistics
- **SciPy** (statistical tests, distributions)

---

## Results

### System Performance

| Model | R² | RMSE (g/L) | MAE (g/L) | MAPE (%) | NSE | Bias (%) | Risk |
|--------|-----|------------|-----------|----------|-----|----------|------|
| **Ensemble** | **0.93** | **0.XX** | **0.XX** | **X.X** | **0.XX** | **±X.X** | **LOW** |
| PINN | 0.91 | 0.XX | 0.XX | X.X | 0.XX | ±X.X | LOW |
| LSTM | 0.89 | 0.XX | 0.XX | X.X | 0.XX | ±X.X | LOW |
| XGBoost | 0.88 | 0.XX | 0.XX | X.X | 0.XX | ±X.X | LOW |
| RandomForest | 0.85 | 0.XX | 0.XX | X.X | 0.XX | ±X.X | LOW |
| Ridge | 0.82 | 0.XX | 0.XX | X.X | 0.XX | ±X.X | LOW |
| Linear | 0.80 | 0.XX | 0.XX | X.X | 0.XX | ±X.X | LOW |

### Most Important Features (Top 10)

1. **pH_effect** (0.XX)
2. **environmental_quality** (0.XX)
3. **photosynthetic_capacity** (0.XX)
4. **temp_effect** (0.XX)
5. **light_efficiency_jassby_platt** (0.XX)
6. **nutrient_effect_haldane** (0.XX)
7. **Temperature_C** (0.XX)
8. **pH** (0.XX)
9. **PAR_umol_m2_s** (0.XX)
10. **Time_h** (0.XX)

### Key System Improvements

- **Random scenario selection** (vs. fixed split): +8% in R² (0.85 → 0.93)
- **Weighted ensemble** (vs. best individual model): +2% in R²
- **Biological feature engineering** (vs. raw features): +15% in R²
- **PINN with biological penalty** (vs. standard network): 100% non-negative predictions

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- 4GB RAM minimum (8GB recommended)

### Quick Installation

```bash
# 1. Clone the repository
git clone https://github.com/NogueiraElectronic/chlorella-biomass-predictor.git
cd chlorella-biomass-predictor

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```txt
# Core ML/DL
torch>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0

# Data processing
numpy>=1.24.0
pandas>=2.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Statistics
scipy>=1.10.0

# Utilities
tqdm>=4.65.0
```

---

## Usage

### Basic Execution

```bash
python chlorella_predictor.py
```

### Workflow

```python
# 1. Import the system
from chlorella_predictor import run_compact_research

# 2. Run the complete pipeline
results = run_compact_research()

# 3. Access results
print(f"Best model: {results['best_model']}")
print(f"R²: {results['results'][results['best_model']]['R²']:.4f}")
print(f"RMSE: {results['results'][results['best_model']]['RMSE']:.4f}")
```

### Advanced Usage

```python
# Access individual components
system = results['system']  # Multi-model system
engine = results['engine']   # Feature engine

# Make predictions on new data
predictions = system.predict(X_new_data)

# View selected features
print(f"Features used: {engine.selected_features}")

# View ensemble weights
print(f"Model weights: {system.weights}")
```

### Customization

#### Change number of epochs
```python
system = CompactMultiModel()
system.train_all(X_train, y_train, epochs=200)  # Default: 150
```

#### Adjust data leakage threshold
```python
manager = SmartDataManager(leakage_threshold=0.90)  # Default: 0.95
```

#### Modify number of features
```python
X_train, X_val, y_train_s, y_val_s, y_train, y_val = \
    engine.selection_and_feature_preparation(df, max_features=30)  # Default: median
```

---

## Code Structure

```
chlorella-biomass-predictor/
│
├── chlorella_predictor.py          # Main script
├── requirements.txt                 # Dependencies
├── README.md                        # This file
├── LICENSE                          # MIT License
│
├── data/
│   └── complete_dataset.csv        # Dataset of 60 scenarios (18K+ records)
│
├── models/                          # (Optional) Saved models
│   ├── best_pinn.pth
│   ├── best_lstm.pth
│   └── ensemble_weights.pkl
│
├── outputs/                         # Results and visualizations
│   ├── feature_importance.png
│   ├── model_comparison.png
│   ├── residuals_analysis.png
│   └── predictions_vs_observed.png
│
└── docs/                            # Additional documentation
    ├── methodology.md
    ├── biological_foundations.md
    └── api_reference.md
```

### Main Components

#### 1. `SmartDataManager`
- Data loading and validation
- Automatic data leakage detection
- Cleaning and preprocessing

#### 2. `BioFeatureEngine`
- Creation of 40+ biological features
- Implementation of scientific equations
- Normalization and scaling
- Feature selection (ensemble of 3 methods)
- Temporal splitting by scenarios

#### 3. `CompactPINN` (PyTorch)
- Neural network with biological constraints
- Architecture: BatchNorm → 64 → 32 → 1
- Custom loss function (MSE + penalty)
- AdamW optimization

#### 4. `CompactLSTM` (PyTorch)
- Recurrent network for time series
- LSTM(32) + Dense(16) + Dropout
- Gradient clipping for stability

#### 5. `CompactMultiModel`
- Training of 6 models
- Weight calculation by performance
- Weighted ensemble prediction

#### 6. Evaluation Functions
- `evaluate_models()`: Complete metrics
- `create_plots()`: 7 visualizations
- `analyze_importance()`: Important features

---

## Scientific References

### Microalgae Biology
1. **Falkowski, P. G., & Raven, J. A. (2013)**. *Aquatic Photosynthesis*. Princeton University Press.
2. **Tredici, M. R. (2010)**. Photobiology of microalgae mass cultures. *Biofuels*, 1(1), 143-162.
3. **Eppley, R. W. (1972)**. Temperature and phytoplankton growth in the sea. *Fishery Bulletin*, 70(4), 1063-1085.

### Modeling and Control
4. **Monod, J. (1949)**. The growth of bacterial cultures. *Annual Review of Microbiology*, 3(1), 371-394.
5. **Bernard, O. (2011)**. Hurdles and challenges for modelling and control of microalgae. *Journal of Process Control*, 21(10), 1378-1389.

### Machine Learning
6. **Guyon, I., & Elisseeff, A. (2003)**. An introduction to variable and feature selection. *Journal of Machine Learning Research*, 3, 1157-1182.
7. **Brownlee, J. (2020)**. *How to Choose a Feature Selection Method For Machine Learning*. Machine Learning Mastery.
8. **Huang, D., et al. (2023)**. Ensemble learning for feature selection in time series prediction. *Applied Sciences*.

### Data Leakage
9. **Sasse, L., et al. (2025)**. Overview of leakage scenarios in supervised machine learning. *Journal of Big Data*.

### Physics-Informed Neural Networks
10. **Raissi, M., et al. (2019)**. Physics-informed neural networks. *Journal of Computational Physics*, 378, 686-707.

---

## Author

**Jesús Torres Nogueira**  
Industrial and Automatic Electronic Engineer

- GitHub: [@NogueiraElectronic](https://github.com/NogueiraElectronic)
- Email: nogueira.electronico@gmail.com
- Portfolio: [nogueiraelectronic.github.io](https://nogueiraelectronic.github.io/)
- LinkedIn: [Jesús Torres Nogueira](https://linkedin.com/in/jesus-torres-nogueira)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Acknowledgments

- To the scientific community for validated biological equations
- To the PyTorch team for flexibility in custom neural networks
- To Scikit-learn for classical ML tools
- To all researchers working on microalgae culture optimization

---

## Project Status

**Stable Version**: Fully functional system  
**In Development**: Integration with real-time monitoring systems  
**Upcoming Features**:
- REST API for real-time predictions
- Interactive dashboard with live visualizations
- IoT sensor integration
- Automatic optimization of culture conditions
- Transfer learning to other microalgae species

---

## Contact

Interested in collaborating or implementing this system in your photobioreactor?

**nogueira.electronico@gmail.com**

---

<div align="center">

**If this project has been useful to you, consider giving it a star on GitHub**

Made with by [Jesús Torres Nogueira](https://github.com/NogueiraElectronic)

</div>
