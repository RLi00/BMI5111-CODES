import pandas as pd
import numpy as np
from scipy.stats import chi2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from statsmodels.stats.outliers_influence import variance_inflation_factor
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv(r"C:\Users\lenovo\Desktop\Liver data.csv", low_memory=False)

# Convert categorical variables to numeric
df['GENDER'] = df['GENDER'].map({'M': 0, 'F': 1})
df['ABO'] = df['ABO'].map({'A': 1, 'B': 2, 'AB': 3, 'O': 4})
df = df.drop(columns=['COD_WL', 'BILIARY', 'DIFFUSE_CHOLANG', 'GRF_FAIL_CAUSE_OSTXT', 'HEP_DENOVO', 'HEP_RECUR',
                      'HEPATIC_OUT_OBS', 'INFECT', 'OTHER_VASC_THROMB', 'PORTAL_VEIN_THROM', 'PRI_GRF_FAIL',
                      'PRI_NON_FUNC', 'RECUR_DISEASE', 'REJ_ACUTE', 'REJ_CHRONIC', 'VASC_THROMB', 'TRTREJ1Y'])

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

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lasso feature selection
lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y)
selected_features = X.columns[(lasso.coef_ != 0)]

# Re-create feature matrix with selected features
X_selected = df_imputed[selected_features]

# VIF calculation and removing features with high VIF
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# Iteratively remove features with VIF > 10
max_vif = 10
while True:
    vif_data = calculate_vif(X_selected)
    max_vif_value = vif_data['VIF'].max()
    if max_vif_value > max_vif:
        remove_feature = vif_data.loc[vif_data['VIF'].idxmax(), 'Feature']
        X_selected = X_selected.drop(columns=[remove_feature])
    else:
        break

# 输出最终保留的特征数量和名称
print(f"Final number of features retained: {X_selected.shape[1]}")
print(f"Final features: {X_selected.columns.tolist()}")

# Recalculate propensity scores using reduced features
logit_model = LogisticRegression(max_iter=10000)
logit_model.fit(X_selected, y)
df_imputed['propensity_score'] = logit_model.predict_proba(X_selected)[:, 1]

# 1:n matching function
def match_propensity_scores(treated, control, score_col, n_matches):
    nn = NearestNeighbors(n_neighbors=n_matches)
    nn.fit(control[[score_col]])
    distances, indices = nn.kneighbors(treated[[score_col]])
    matched_indices = indices.flatten()
    matched_control = control.iloc[matched_indices].copy()
    matched_control['match_id'] = np.repeat(np.arange(len(treated)), n_matches)
    treated['match_id'] = np.arange(len(treated))
    matched = pd.concat([treated, matched_control], axis=0)
    return matched

# Separate treated and control groups
treated = df_imputed[df_imputed['ARGININE_DON'] == 1].copy()
control = df_imputed[df_imputed['ARGININE_DON'] == 0].copy()

# Determine n_matches for matching
n_matches = max(len(control) // len(treated), 1)

# Perform propensity score matching
matched_df = match_propensity_scores(treated, control, score_col='propensity_score', n_matches=n_matches)

# Create matched feature matrix and target using all Lasso-selected features
X_matched = matched_df[selected_features]
y_matched = matched_df['HEPATIC_ART_THROM']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_matched, y_matched, test_size=0.3, random_state=42)

# Balance the training set using SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Build logistic regression model with pipeline
logit_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logit', LogisticRegression(max_iter=50000, solver='lbfgs'))
])

logit_pipeline.fit(X_train_smote, y_train_smote)

# Predict and evaluate the model
y_pred = logit_pipeline.predict(X_test)
y_pred_prob = logit_pipeline.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Hosmer-Lemeshow Test with Plotting
def hosmer_lemeshow_test(y_true, y_pred_prob, g=10):
    data = pd.DataFrame({'y_true': y_true, 'y_pred_prob': y_pred_prob})
    data['cut'] = pd.qcut(data['y_pred_prob'], g, duplicates='drop')
    obs = data.groupby('cut', observed=False)['y_true'].sum()
    exp = data.groupby('cut', observed=False)['y_pred_prob'].sum()
    n = data.groupby('cut', observed=False).size()
    hl_stat = np.sum((obs - exp) ** 2 / (exp * (1 - exp / n)))
    p_value = chi2.sf(hl_stat, g - 2)

    # Plotting observed vs. expected
    plt.figure(figsize=(10, 6))
    plt.scatter(exp, obs, color='blue', label='Observed')
    plt.plot([0, max(exp)], [0, max(obs)], color='red', linestyle='--', label='Expected')
    plt.xlabel('Expected Events')
    plt.ylabel('Observed Events')
    plt.title(f'Hosmer-Lemeshow Test: Observed vs. Expected (p-value = {p_value:.4f})')
    plt.legend()
    plt.show(block=False)

    return hl_stat, p_value


hl_stat, hl_p_value = hosmer_lemeshow_test(y_test, y_pred_prob)

print(f'Hosmer-Lemeshow Statistic: {hl_stat:.4f}')
print(f'Hosmer-Lemeshow p-value: {hl_p_value:.100f}')


# Plot ROC curve and calculate AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc:.2f}")

# Find best threshold using Youden's J statistic
J = tpr - fpr
best_threshold_index = np.argmax(J)
best_threshold = thresholds[best_threshold_index]
print(f"Best threshold by Youden's J: {best_threshold:.2f}")

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show(block=False)

# Predict by the best threshold
y_pred_best_threshold = (y_pred_prob > best_threshold).astype(int)

# Output the classification report as the best threshold used
from sklearn.metrics import classification_report, accuracy_score

print(f"\nClassification Report with Best Threshold ({best_threshold:.2f}):")
print(classification_report(y_test, y_pred_best_threshold))
print("Accuracy with Best Threshold:", accuracy_score(y_test, y_pred_best_threshold))
