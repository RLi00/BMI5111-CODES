import pandas as pd
import numpy as np
from scipy.stats import chi2
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv(r"C:\Users\lenovo\Desktop\Liver data.csv", low_memory=False)

# Convert categorical variables to numeric
df['GENDER'] = df['GENDER'].map({'M': 0, 'F': 1})
df['ABO'] = df['ABO'].map({'A': 1, 'B': 2, 'AB': 3, 'O': 4})
df = df.drop(columns=['COD_WL'])

# Retain columns ARGININE_DON and HEPATIC_ART_THROM，and delete all rows containing missing values in these two columns
df = df.dropna(subset=['ARGININE_DON', 'HEPATIC_ART_THROM'])

# Drop non-numeric variables
df = df.select_dtypes(include=[int, float])

# Step 1: Remove features with more than 70% missing values
feature_threshold = 0.7
df = df.dropna(thresh=int((1 - feature_threshold) * len(df)), axis=1)

# Step 2: Remove samples with more than 40% missing values, 但保留已经处理的列
sample_threshold = 0.4
df = df.dropna(thresh=int((1 - sample_threshold) * df.shape[1]), axis=0)

# Step 3: Handle remaining missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Select features and target variable
X = df_imputed.drop(columns=['HEPATIC_ART_THROM'])
y = df_imputed['HEPATIC_ART_THROM']

# Feature scaling (optional for XGBoost, but included here for completeness)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lasso feature selection
from sklearn.linear_model import LassoCV

lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y)
selected_features = X.columns[(lasso.coef_ != 0)]

print(f"Selected features by Lasso: {selected_features.tolist()}")

# Re-create feature matrix with selected features
X_selected = df_imputed[selected_features]

# Perform Propensity Score Matching (PSM)
logit = LogisticRegression()
logit.fit(X_selected, df_imputed['ARGININE_DON'])

# Compute propensity scores
df_imputed['propensity_score'] = logit.predict_proba(X_selected)[:, 1]

# Create treatment and control groups
treated = df_imputed[df_imputed['ARGININE_DON'] == 1]
control = df_imputed[df_imputed['ARGININE_DON'] == 0]

# Nearest neighbor matching
nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[['propensity_score']])
distances, indices = nn.kneighbors(treated[['propensity_score']])
control_matched = control.iloc[indices.flatten()]

# Combine matched dataset
matched_df = pd.concat([treated, control_matched], axis=0)

# Re-create matched feature matrix and target
X_matched = matched_df[selected_features]
y_matched = matched_df['HEPATIC_ART_THROM']

# Balance the matched dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_matched, y_matched)

# Split matched dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Hyperparameter tuning with RandomizedSearchCV
param_dist = {
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 3],
    'gamma': [0, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200]
}

random_search = RandomizedSearchCV(
    estimator=xgb.XGBClassifier(eval_metric='logloss', random_state=42),
    param_distributions=param_dist,
    scoring='accuracy',
    n_iter=10,  # interation times
    cv=3,  # folds
    verbose=1
)

random_search.fit(X_train, y_train)
print("Best parameters found: ", random_search.best_params_)

# Train XGBoost model with best parameters
best_params = random_search.best_params_
xgb_model = xgb.XGBClassifier(
    eval_metric='logloss',
    random_state=42,
    **best_params
)

xgb_model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = xgb_model.predict(X_test)
y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Adjust classification threshold
threshold = 0.3
y_pred_adjusted = (y_pred_prob > threshold).astype(int)

print(f"\nClassification Report after adjusting threshold to {threshold}:")
print(classification_report(y_test, y_pred_adjusted))
print("Adjusted Model Accuracy:", accuracy_score(y_test, y_pred_adjusted))


# Hosmer-Lemeshow Test
def hosmer_lemeshow_test(y_true, y_pred_prob, g=10):
    data = pd.DataFrame({'y_true': y_true, 'y_pred_prob': y_pred_prob})
    data['cut'] = pd.qcut(data['y_pred_prob'], g, duplicates='drop')
    obs = data.groupby('cut')['y_true'].sum()
    exp = data.groupby('cut')['y_pred_prob'].sum()
    n = data.groupby('cut').size()
    hl_stat = np.sum((obs - exp) ** 2 / (exp * (1 - exp / n)))
    p_value = chi2.sf(hl_stat, g - 2)

    # Plotting observed vs. expected
    plt.figure(figsize=(10, 6))
    plt.scatter(exp, obs, color='blue', label='Observed')
    plt.plot([0, max(exp)], [0, max(obs)], color='red', linestyle='--', label='Expected')
    plt.xlabel('Expected Events')
    plt.ylabel('Observed Events')
    plt.title('Hosmer-Lemeshow Test Observed vs. Expected')
    plt.title(f'Hosmer-Lemeshow Test: Observed vs. Expected (p-value = {p_value:.4f})')
    plt.legend()
    plt.show()

    return hl_stat, p_value


hl_stat, hl_p_value = hosmer_lemeshow_test(y_test, y_pred_prob)

print(f'Hosmer-Lemeshow Statistic: {hl_stat:.4f}')
print(f'Hosmer-Lemeshow p-value: {hl_p_value:.4f}')

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
