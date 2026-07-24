# Employee Attrition Prediction using Decision Tree and Random Forest Classification

## 📋 Objective
Develop and compare Decision Tree and Random Forest classification models to predict employee attrition based on demographic, professional, and work-related attributes. Identify key factors driving employee turnover to enable proactive retention strategies.

## 📊 Dataset
**IBM HR Analytics Employee Attrition & Performance Dataset**
- **Source**: [Kaggle - IBM HR Analytics Employee Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Size**: 1,470 employees × 35 features
- **Target Variable**: `Attrition` (Yes/No) - 16.1% attrition rate
- **License**: Open Database License (ODbL) - Dataset not uploaded to this repository per Kaggle terms

### Features
| Category | Features |
|----------|----------|
| **Demographic** | Age, Gender, MaritalStatus, DistanceFromHome |
| **Professional** | Education, EducationField, JobRole, JobLevel, Department, TotalWorkingYears, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager, NumCompaniesWorked |
| **Work-Related** | BusinessTravel, OverTime, DailyRate, HourlyRate, MonthlyIncome, MonthlyRate, PercentSalaryHike, PerformanceRating, StockOptionLevel, TrainingTimesLastYear, EnvironmentSatisfaction, JobSatisfaction, RelationshipSatisfaction, WorkLifeBalance, JobInvolvement |
| **Administrative** | EmployeeNumber, EmployeeCount, StandardHours, Over18 |

## 🛠️ Libraries Used
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **matplotlib** - Static visualizations
- **seaborn** - Statistical visualizations
- **scikit-learn** - Machine learning algorithms and metrics:
  - `DecisionTreeClassifier`
  - `RandomForestClassifier`
  - `train_test_split`
  - `LabelEncoder`
  - `accuracy_score`, `precision_score`, `recall_score`, `f1_score`
  - `confusion_matrix`, `classification_report`
  - `export_text` (decision tree visualization)

## 🔬 Methodology

### Task 1: Data Understanding
- Loaded dataset and displayed first 5 records
- Identified 26 numerical features, 8 categorical features, and 1 target variable (`Attrition`)
- Analyzed dataset info, summary statistics, and target distribution (83.9% No, 16.1% Yes)
- Visualized attrition distribution

### Task 2: Data Preprocessing
- **Missing Values**: Checked - no missing values found
- **Removed Unnecessary Columns**: Dropped `EmployeeCount`, `Over18`, `StandardHours` (single unique value)
- **Encoded Categorical Variables**: Applied Label Encoding to 8 categorical features and target variable
- **Train-Test Split**: 80% training (1,176 samples), 20% testing (294 samples), stratified by target

### Task 3: Model Development
| Model | Configuration |
|-------|--------------|
| **Decision Tree** | `criterion='gini'`, `max_depth=5`, `min_samples_split=5`, `min_samples_leaf=2`, `random_state=42` |
| **Random Forest** | `n_estimators=100`, `max_depth=10`, `min_samples_split=5`, `min_samples_leaf=2`, `random_state=42`, `n_jobs=-1` |

Both models trained on identical training data.

### Task 4: Model Evaluation and Comparison
Evaluated both models using:
- **Accuracy Score**
- **Precision** (weighted average)
- **Recall** (weighted average)
- **F1-Score** (weighted average)
- **Confusion Matrix**
- **Feature Importance** (Random Forest only)

### Task 5: Conclusion
Comparative analysis with actionable insights for HR retention strategies.

## 📈 Results

### Model Performance Comparison

| Metric | Decision Tree | Random Forest | Winner |
|--------|--------------|---------------|--------|
| **Accuracy** | 84.01% | 82.99% | Decision Tree |
| **Precision** | 80.44% | 77.06% | Decision Tree |
| **Recall** | 84.01% | 82.99% | Decision Tree |
| **F1-Score** | 80.89% | 78.29% | Decision Tree |

### Confusion Matrices

**Decision Tree**
```
[[238   9]
 [ 38   9]]
```
- True Negatives: 238 | False Positives: 9
- False Negatives: 38 | True Positives: 9

**Random Forest**
```
[[240   7]
 [ 43   4]]
```
- True Negatives: 240 | False Positives: 7
- False Negatives: 43 | True Positives: 4

### Feature Importance (Random Forest - Top 15)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | MonthlyIncome | 0.0789 |
| 2 | Age | 0.0712 |
| 3 | TotalWorkingYears | 0.0592 |
| 4 | OverTime | 0.0510 |
| 5 | DailyRate | 0.0507 |
| 6 | YearsAtCompany | 0.0468 |
| 7 | MonthlyRate | 0.0463 |
| 8 | EmployeeNumber | 0.0462 |
| 9 | DistanceFromHome | 0.0461 |
| 10 | YearsWithCurrManager | 0.0391 |
| 11 | HourlyRate | 0.0390 |
| 12 | JobRole | 0.0341 |
| 13 | NumCompaniesWorked | 0.0341 |
| 14 | EnvironmentSatisfaction | 0.0307 |
| 15 | PercentSalaryHike | 0.0292 |

## 🔄 Model Comparison

### Observations
1. **Decision Tree slightly outperforms Random Forest** across all metrics (accuracy: 84.01% vs 82.99%), likely due to the relatively small dataset and constrained tree depth preventing overfitting.

2. **Random Forest achieves higher True Negative Rate** (97.2% vs 96.4%) but lower True Positive Rate (19.1% vs 8.5%), indicating it's more conservative in predicting attrition.

3. **Feature Importance reveals key attrition drivers**: MonthlyIncome, Age, TotalWorkingYears, and OverTime are top predictors, providing actionable insights for HR.

4. **Class Imbalance Impact**: With 16% attrition rate, both models struggle with minority class (Yes) recall. Decision Tree achieves 19.1% recall for attrition class vs Random Forest's 8.5%.

## 📝 Conclusion

The Decision Tree classifier marginally outperforms Random Forest on this dataset across all evaluation metrics, achieving 84.01% accuracy compared to 82.99%. This unexpected result likely stems from the constrained tree depth (max_depth=5) preventing overfitting, while Random Forest's ensemble of 100 deeper trees (max_depth=10) may overfit the relatively small dataset of 1,470 samples.

Random Forest typically outperforms Decision Trees through ensemble learning—building multiple trees on bootstrapped samples and averaging predictions via bagging. This reduces variance and overfitting inherent in single decision trees, which tend to memorize training data noise. However, with limited data and appropriate regularization, a well-tuned single tree can match or exceed ensemble performance.

**Decision Tree Limitation**: Prone to overfitting with deep trees, creating complex decision boundaries that don't generalize. High variance makes them unstable—small data changes yield vastly different trees.

**Random Forest Limitation**: Reduced interpretability—hundreds of trees create a "black box" versus a single tree's transparent rules. Higher computational cost and memory requirements for training and inference.

**Recommendation**: For this specific dataset, the Decision Tree offers better performance with superior interpretability. Key attrition drivers (OverTime, MonthlyIncome, TotalWorkingYears, Age) should inform targeted retention programs: overtime management, competitive compensation for tenured employees, and career development for younger workforce segments.

## 📁 Repository Structure
```
Assignment-5/
├── Assignment-5.py          # Complete analysis script
├── README.md                # This file
├── .gitignore               # Git ignore rules
├── attrition_distribution.png
├── confusion_matrices.png
├── feature_importance.png
├── model_comparison.png
└── conclusion.txt           # Detailed conclusion
```

## 🚀 How to Run
```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the analysis
python Assignment-5.py
```

## 📥 Dataset Download
Download the dataset from Kaggle:
```bash
# Option 1: Kaggle CLI
kaggle datasets download -d pavansubhasht/ibm-hr-analytics-attrition-dataset

# Option 2: Direct download from
# https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
```
Place `WA_Fn-UseC_-HR-Employee-Attrition.csv` in the project root and rename to `emp_attrition.csv` or update the filename in `Assignment-5.py`.

## 📄 Submission Details
- **Assignment**: Employee Attrition Prediction
- **Deadline**: 27 July 2026, 11:59 PM IST
- **Total Marks**: 10 Marks
- **Tasks**: 5 Tasks covering Data Understanding, Preprocessing, Model Development, Evaluation, and Conclusion