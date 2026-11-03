# 🧠 Mental Health Score Prediction

A machine learning regression project that explores the relationship between students' social media usage, daily habits, academic activities, and their mental health scores. The notebook performs exploratory data analysis (EDA), cleans and preprocesses the dataset, builds regression pipelines, compares model performance, and saves the trained Extra Trees model.

> **Important:** This project predicts the numerical `Mental_Health_Score` field in the dataset. It is an educational machine learning project, not a clinical assessment or mental health diagnostic tool.

## 📌 Project Overview

The objective is to estimate a student's mental health score using demographic information, social media usage patterns, and lifestyle-related features.

The project follows a complete regression workflow:

1. Load and inspect the dataset.
2. Perform exploratory data analysis.
3. Identify and handle duplicate records and invalid physical-activity values.
4. Engineer a grouped country feature.
5. Split the data into training and testing sets.
6. Build preprocessing pipelines for different feature types.
7. Train and evaluate regression models.
8. Save the selected model using Joblib.

## 📂 Dataset

The notebook reads the dataset from:

```text
Student Social Media And Mental Health Impact.csv
```

### Dataset summary

| Property | Value |
|---|---:|
| Initial records | 5,000 |
| Initial columns | 13 |
| Missing values | 0 |
| Duplicate records identified | 2 |
| Records after duplicate removal | 4,998 |
| Target variable | `Mental_Health_Score` |

### Features

| Feature | Description |
|---|---|
| `Age` | Student's age |
| `Gender` | Gender category |
| `Country` | Country |
| `Academic_Level` | Academic level |
| `Most_Used_Platform` | Most-used social media platform |
| `Purpose_Of_Use` | Main purpose of social media use |
| `Avg_Daily_Usage_Hours` | Average daily social media usage |
| `Daily_Unlocks` | Number of daily device unlocks |
| `Study_Hours` | Study hours |
| `Physical_Activity_Hours` | Physical activity hours |
| `Sleep_Hours_Per_Night` | Sleep hours per night |
| `Stress_Level` | Categorical stress level |

**Target:** `Mental_Health_Score` — a numerical score predicted by the regression models.

## 🔍 Exploratory Data Analysis

The notebook uses Pandas, Matplotlib, and Seaborn to investigate the dataset.

EDA includes:

- Dataset structure, data types, and descriptive statistics.
- Missing-value and duplicate checks.
- Distribution of `Mental_Health_Score`.
- Correlation heatmap for numerical features.
- Relationship between stress level and mental health score.
- Scatter plots comparing mental health score with:
  - Average daily social media usage
  - Sleep hours
  - Physical activity hours
  - Study hours
  - Daily device unlocks
- Frequency distribution of the most-used social media platforms.
- IQR-based outlier inspection and skewness analysis.

## 🧹 Data Cleaning & Feature Engineering

The notebook applies the following transformations:

- Removes the **2 duplicate records**, leaving 4,998 records.
- Clips negative values in `Physical_Activity_Hours` to zero.
- Identifies the 10 most frequent countries and groups all remaining countries into `Other`.
- Creates the `Grouped_country` feature for model input.

No missing values were found in the initial dataset.

## ⚙️ Preprocessing Pipeline

The features are handled according to their data types:

| Feature type | Features | Transformation |
|---|---|---|
| Skewed numerical | `Study_Hours` | `log1p` transformation followed by `StandardScaler` |
| Other numerical | Age, usage hours, unlocks, physical activity, sleep hours | `StandardScaler` |
| Ordinal categorical | `Stress_Level` | `OrdinalEncoder` with Low, Medium, High, Very High order |
| Nominal categorical | Gender, academic level, platform, purpose, grouped country | `OneHotEncoder(handle_unknown="ignore")` |

A `ColumnTransformer` combines these preprocessing pipelines. Each regression model is wrapped in a Scikit-learn `Pipeline`, so preprocessing is fitted on the training data and applied consistently during prediction.

### Train-test split

The notebook uses:

```python
train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)
```

| Subset | Share |
|---|---:|
| Training | 70% |
| Testing | 30% |

## 🤖 Machine Learning Models

Three regression algorithms are trained and compared:

1. **Linear Regression**
2. **Random Forest Regressor**
3. **Extra Trees Regressor**

## 📈 Model Evaluation & Results

The notebook evaluates the models using:

- **R² score:** Measures the proportion of target variance explained by the model. Higher is better.
- **MAE (Mean Absolute Error):** Measures average absolute prediction error. Lower is better.
- **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors more strongly. Lower is better.

### Test-set comparison

The following values are taken from the notebook's consolidated results table:

| Model | Training R² | Test R² | MAE | RMSE |
|---|---:|---:|---:|---:|
| Linear Regression | 0.7237 | 0.7398 | 0.5362 | 0.6760 |
| Random Forest Regressor | 0.9808 | 0.8776 | 0.3472 | 0.4637 |
| **Extra Trees Regressor** | **0.99999** | **0.9194** | **0.2659** | **0.3761** |

### Best-performing model

<badge color="success">Best test R²</badge>

**Extra Trees Regressor** achieves the strongest test-set performance in the notebook's consolidated comparison:

- **Test R²:** 0.9194
- **MAE:** 0.2659
- **RMSE:** 0.3761

The training R² is almost 1.0, while the test R² is lower. This gap suggests possible overfitting, so validation on additional data and further generalization checks are recommended.

## 💾 Saved Model

The notebook saves the complete Extra Trees preprocessing and regression pipeline with Joblib:

```python
import joblib

joblib.dump(extra_pipeline, "Mental_Health_Model.pkl")
```

The saved pipeline includes both the fitted preprocessing steps and the trained Extra Trees Regressor. This makes it possible to pass input data with the expected original feature columns to the pipeline without manually repeating the transformations.

**Note:** The notebook creates `Mental_Health_Model.pkl` when the final save cell is executed. The CSV dataset is a separate file and is not embedded in the saved model.

## 🛠️ Technologies & Libraries

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/mental-health-score-prediction.git
cd mental-health-score-prediction
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
```

### 4. Add the dataset

Place `Student Social Media And Mental Health Impact.csv` in the project root directory, or update the CSV path in the notebook.

### 5. Run the notebook

```bash
jupyter notebook
```

Open `Mental_Health(1).ipynb` and run the cells in order.

## 📁 Suggested Repository Structure

```text
mental-health-score-prediction/
│
├── Mental_Health.ipynb
├── Student Social Media And Mental Health Impact.csv
├── Mental_Health_Model.pkl
├── README.md
└── requirements.txt
```

Only include the `.pkl` model and dataset in the repository if you intend to distribute them and their size and licensing permit it. Consider using Git LFS or a release asset for large model files.

To create a dependency file from your active environment:

```bash
pip freeze > requirements.txt
```

## 🔮 Future Improvements

- Use cross-validation to obtain a more reliable estimate of model performance.
- Tune Extra Trees and Random Forest hyperparameters.
- Investigate the training-test performance gap and evaluate overfitting.
- Add residual plots and predicted-versus-actual visualizations.
- Compare feature importance or permutation importance to understand model behavior.
- Evaluate performance on an independent dataset from a different population.
- Add input validation and a prediction interface if deploying the model.
- Document the dataset source, collection methodology, score definition, and usage limitations.

## ⚠️ Disclaimer

This project is for learning and research purposes only. The predicted score reflects patterns in the supplied dataset and should not be interpreted as a diagnosis, clinical measurement, or substitute for professional mental health support. Correlations and model predictions do not establish causation.

## 👤 Author

**Your Name**

- GitHub: [your-username](https://github.com/your-username)
- LinkedIn: [your-linkedin](https://www.linkedin.com/)

---

If you find this project useful, consider giving the repository a ⭐.
