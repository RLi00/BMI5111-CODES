import pandas as pd
import numpy as np
from scipy.stats import chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load data
df = pd.read_csv(r"C:\Users\lenovo\Desktop\Liver data.csv", low_memory=False)

# Exclude post-exposure variables
post_exposure_vars = [
                  'ABO', 'ABO_MAT', 'ACADEMIC_LEVEL_TCR', 'ACADEMIC_LEVEL_TRR', 'ACADEMIC_PRG_TCR', 'ACADEMIC_PRG_TRR',
                  'ACUTE_REJ_EPI', 'ADMISSION_DATE', 'AGE', 'AGE_GROUP', 'ALBUMIN_TX', 'AMIS', 'ARTIFICIAL_LI_TCR',
                  'ARTIFICIAL_LI_TRR', 'ASCITES_TCR', 'ASCITES_TRR_OLD', 'ASCITES_TX', 'BACT_PERIT_TCR', 'BILIARY',
                  'BMI_CALC', 'BMI_TCR', 'BW4', 'BW6', 'C1', 'C2', 'CITIZEN_COUNTRY', 'CITIZENSHIP', 'CITIZENSHIP_DON',
                  'CMV_IGG', 'CMV_IGM', 'CMV_STATUS', 'COD', 'COD_OSTXT', 'COD_OSTXT_WL', 'COD_WL', 'COD2', 'COD2_OSTXT',
                  'COD3', 'COD3_OSTXT', 'COMPOSITE_DEATH_DATE', 'CREAT_TX', 'CTR_CODE', 'DATA_TRANSPLANT', 'DATA_WAITLIST',
                  'DAYSWAIT_CHRON', 'DEATH_DATE', 'DGN_OSTXT_TCR', 'DGN_TCR', 'DGN2_OSTXT_TCR', 'DGN2_TCR', 'DIAB', 'DIAG',
                  'DIAG_OSTXT', 'DIAL_TX', 'DIFFUSE_CHOLANG', 'DIS_ALKPHOS', 'DIS_SGOT', 'DISCHARGE_DATE', 'DISTANCE',
                  'DON_TY', 'DONATION_DON', 'DQ1', 'DQ2', 'DR51', 'DR51_2', 'DR52', 'DR52_2', 'DR53', 'DR53_2',
                  'EBV_SEROSTATUS', 'ECMO', 'EDUCATION', 'ENCEPH_TCR', 'ENCEPH_TRR_OLD', 'ENCEPH_TX', 'END_BMI_CALC',
                  'END_DATE', 'END_OPO_CTR_CODE', 'END_STAT', 'ETHCAT', 'ETHNICITY', 'DEAD', 'EXC_OTHER_DIAG', 'EXC_CASE',
                  'FINAL_ALBUMIN', 'FINAL_ASCITES', 'FINAL_BILIRUBIN', 'FINAL_CTP_SCORE', 'FINAL_DIALYSIS_PRIOR_WEEK',
                  'FINAL_ENCEPH', 'FINAL_INR', 'FINAL_MELD_OR_PELD', 'FINAL_MELD_PELD_LAB_SCORE', 'FINAL_SERUM_CREAT',
                  'FINAL_SERUM_SODIUM', 'FREE_DON', 'FUNC_STAT_TCR', 'FUNC_STAT_TRF', 'FUNC_STAT_TRR', 'GENDER',
                  'GRF_FAIL_CAUSE_OSTXT', 'GRF_FAIL_DATE', 'GRF_STAT', 'GSTATUS', 'GTIME', 'HBEAB_OLD', 'HBV_CORE',
                  'HBV_NAT', 'HBV_SUR_ANTIGEN', 'HBV_SURF_TOTAL', 'HCC_DIAG', 'HCC_DIAGNOSIS_TCR', 'HCC_EVER_APPR',
                  'HCV_NAT', 'HCV_SEROSTATUS', 'HEP_DENOVO', 'HEP_RECUR', 'HEPATIC_ART_THROM', 'HEPATIC_OUT_OBS', 'HEPD_OLD',
                  'HGT_CM_CALC', 'HGT_CM_TCR', 'HIV_NAT', 'HIV_SEROSTATUS', 'HMO_PPO_DON', 'IABP', 'INACT_REASON_CD', 'INFECT',
                  'INIT_ALBUMIN', 'INIT_ASCITES', 'INIT_BILIRUBIN', 'INIT_BMI_CALC', 'INIT_CTP_SCORE', 'INIT_DATE',
                  'INIT_DIALYSIS_PRIOR_WEEK', 'INIT_ENCEPH', 'INIT_HGT_CM', 'INIT_INR', 'INIT_MELD_OR_PELD', 'INIT_MELD_PELD_LAB_SCORE',
                  'INIT_OPO_CTR_CODE', 'INIT_SERUM_CREAT', 'INIT_SERUM_SODIUM', 'INIT_STAT', 'INIT_WGT_KG', 'INOTROPES', 'INR_TX',
                  'LIFE_SUP_TCR', 'LIFE_SUP_TRR', 'LISTING_CTR_CODE', 'LISTYR', 'LITYP', 'LIV_DON_TY', 'LIV_DON_TY_OSTXT',
                  'LOS', 'MALIG_OSTXT_TRR', 'MALIG_TCR', 'MALIG_TRR', 'MALIG_TY_TRR', 'MED_COND_TRR', 'MEDICAID_DON',
                  'MEDICARE_DON', 'MELD_PELD_LAB_SCORE', 'MRCREATG_OLD', 'MULTIORG', 'MUSCLE_WAST_TCR', 'NEOADJUVANT_THERAPY_TCR',
                  'NUM_PREV_TX', 'ON_VENT_TRR', 'OPO_CTR_CODE', 'OTH_LIFE_SUP_OSTXT_TCR', 'OTH_LIFE_SUP_TCR', 'OTH_LIFE_SUP_TRR',
                  'OTHER_VASC_THROMB', 'PERM_STATE_TRR', 'PGE', 'PORTAL_VEIN_TCR', 'PORTAL_VEIN_THROM', 'PORTAL_VEIN_TRR',
                  'PREV_AB_SURG_TCR', 'PREV_AB_SURG_TRR', 'PREV_PI_TX_TCR_ARCHIVE', 'PREV_TX', 'PREV_TX_ANY', 'PREV_TX_DATE',
                  'PRI_GRF_FAIL', 'PRI_NON_FUNC', 'PRI_PAYMENT_CTRY_DON', 'PRI_PAYMENT_CTRY_TCR', 'PRI_PAYMENT_CTRY_TRR',
                  'PRI_PAYMENT_DON', 'PRI_PAYMENT_TCR', 'PRI_PAYMENT_TRR', 'PRIV_INS_DON', 'PRVTXDIF', 'PSTATUS', 'PT_CODE',
                  'PT_OTH_DON', 'PTIME', 'PX_NON_COMPL', 'PX_STAT', 'PX_STAT_DATE', 'RA1', 'RA2', 'RB1', 'RB2', 'RDR1',
                  'RDR2', 'RECUR_DISEASE', 'REFERRAL_DATE', 'REGION', 'REJ_ACUTE', 'REJ_CHRONIC', 'REM_CD', 'RETXDATE',
                  'SELF_DON', 'SHARE_TY', 'STATUS_DDR', 'STATUS_LDR', 'STATUS_TCR', 'STATUS_TRR', 'TBILI_TX', 'TIPSS_TCR',
                  'TIPSS_TRR', 'TOT_SERUM_ALBUM', 'TRR_ID_CODE', 'TRTREJ1Y', 'TRTREJ6M', 'TX_DATE', 'TX_MELD', 'TX_PROCEDUR_TY',
                  'TX_YEAR', 'TXHRT', 'TXINT', 'TXKID', 'TXLIV', 'TXLNG', 'TXPAN', 'TXVCA', 'VAD_TAH', 'VAL_DT_DDR', 'VAL_DT_LDR',
                  'VAL_DT_TCR', 'VAL_DT_TRR', 'VASC_THROMB', 'VENTILATOR_TCR', 'WGT_KG_CALC', 'WGT_KG_TCR', 'WL_ID_CODE',
                  'WLHL', 'WLHR', 'WLIN', 'WLKI', 'WLKP', 'WLLI', 'WLLU', 'WLPA', 'WLPI', 'WLVC', 'WORK_INCOME_TCR',
                  'WORK_INCOME_TRR', 'YR_ENTRY_US_TCR', 'RECOV_OUT_US', 'RECOV_COUNTRY', 'RECOVERY_DATE_DON',
                  'COLD_ISCH', 'LI_BIOPSY']

df = df.drop(columns=post_exposure_vars)
df = df.dropna(subset=['ARGININE_DON'])

# Remove variables with more than 50% missing values
feature_threshold = 0.5
df = df.dropna(thresh=int((1 - feature_threshold) * len(df)), axis=1)

# Define numeric and categorical variables
original_num_variables = [
    'WGT_KG_TCR', 'HGT_CM_TCR', 'BMI_TCR', 'INIT_WGT_KG', 'INIT_HGT_CM', 'DAYSWAIT_CHRON', 'INIT_AGE',
    'INIT_BMI_CALC', 'END_BMI_CALC', 'INIT_ALBUMIN', 'INIT_BILIRUBIN', 'INIT_INR', 'INIT_MELD_PELD_LAB_SCORE',
    'INIT_SERUM_CREAT', 'INIT_SERUM_SODIUM', 'INIT_CTP_SCORE', 'FINAL_ALBUMIN', 'FINAL_BILIRUBIN',
    'FINAL_INR', 'FINAL_MELD_PELD_LAB_SCORE', 'FINAL_SERUM_CREAT', 'FINAL_SERUM_SODIUM', 'FINAL_CTP_SCORE',
    'PTIME', 'DA1', 'DA2', 'DB1', 'DB2', 'DDR1', 'DDR2', 'RA1', 'RA2', 'RB1', 'RB2', 'RDR1', 'RDR2',
    'AGE_DON', 'BUN_DON', 'CREAT_DON', 'SGOT_DON', 'SGPT_DON', 'TBILI_DON', 'CANCER_FREE_INT_DON',
    'HGT_CM_DON_CALC', 'WGT_KG_DON_CALC', 'BMI_DON_CALC', 'CREAT_TX', 'TBILI_TX', 'INR_TX', 'ALBUMIN_TX',
    'MELD_PELD_LAB_SCORE', 'LOS', 'AGE', 'DIAG', 'DISTANCE', 'COLD_ISCH', 'HGT_CM_CALC', 'WGT_KG_CALC',
    'BMI_CALC', 'DIS_SGOT', 'DIS_ALKPHOS', 'MACRO_FAT_LI_DON', 'MICRO_FAT_LI_DON', 'LV_EJECT_DON', 'PH_DON',
    'HEMATOCRIT_DON'
]

num_variables = list(set(df.columns).intersection(set(original_num_variables)))
Target = ['ARGININE_DON']
cat_variables = list(set(list(df.columns)) - set(original_num_variables) - set(Target))

# Count the different numbers of values of lists
unique_value_counts = df[cat_variables].nunique()

# 筛选出不同值数量大于等于 10 的列
columns_with_10_or_more_unique = unique_value_counts[unique_value_counts >= 10]

# 将满足条件的列名生成一个 list
columns_to_select = columns_with_10_or_more_unique.index.tolist()

# 更新 cat_variables，去掉有太多不同值的列
cat_variables = list(set(cat_variables) - set(columns_to_select))

# 选择所有变量
all_variables = cat_variables + num_variables + Target
df = df[all_variables]

# Convert categorical variables to dummy variables
df = pd.get_dummies(df, columns=cat_variables, drop_first=True)

# Fill missing values
imputer = SimpleImputer(strategy='mean')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Standardize numerical variables
scaler = StandardScaler()
df[num_variables] = scaler.fit_transform(df[num_variables])

# **模型构建和预测步骤**
# Separate independent and dependent variables
X = df.drop(columns=['ARGININE_DON'])
y = df['ARGININE_DON']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# LassoCV for feature selection
lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso.fit(X_train, y_train)

# Get the selected features from Lasso
selected_features = X_train.columns[(lasso.coef_ != 0)]
print(f"Selected features by Lasso: {selected_features}")

# Train and test using only the selected features
X_train_lasso = X_train[selected_features]
X_test_lasso = X_test[selected_features]

# 定义模型评估函数
def xgb_evaluate(max_depth, learning_rate, n_estimators, reg_lambda, reg_alpha):
    model = XGBClassifier(
        eval_metric='logloss',
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        n_estimators=int(n_estimators),
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha
    )
    cv_result = cross_val_score(model, X_train_lasso, y_train, cv=5, scoring='accuracy')
    return np.mean(cv_result)

# 定义参数空间
pbounds = {
    'max_depth': (3, 7),
    'learning_rate': (0.01, 0.3),
    'n_estimators': (100, 300),
    'reg_lambda': (0.1, 10),
    'reg_alpha': (0.1, 10)
}

# 进行贝叶斯优化
xgb_bo = BayesianOptimization(f=xgb_evaluate, pbounds=pbounds, random_state=42)
xgb_bo.maximize(init_points=10, n_iter=150)

# 从贝叶斯优化结果中获取最佳参数
best_params = xgb_bo.max['params']

# 将参数中的整数类型值进行转换
best_params['max_depth'] = int(best_params['max_depth'])
best_params['n_estimators'] = int(best_params['n_estimators'])

# 使用贝叶斯优化找到的最佳参数来训练 XGBoost 模型
best_xgb_model = XGBClassifier(
    eval_metric='logloss',
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    n_estimators=best_params['n_estimators'],
    reg_alpha=best_params['reg_alpha'],
    reg_lambda=best_params['reg_lambda']
)

# 训练模型
best_xgb_model.fit(X_train_lasso, y_train)

# 用训练好的模型进行预测
y_pred = best_xgb_model.predict(X_test_lasso)

# 输出预测结果
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# **绘制ROC曲线**
y_pred_prob = best_xgb_model.predict_proba(X_test_lasso)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

print(f"AUC: {roc_auc:.4f}")

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
    plt.show()

    return hl_stat, p_value


hl_stat, hl_p_value = hosmer_lemeshow_test(y_test, y_pred_prob)

print(f'Hosmer-Lemeshow Statistic: {hl_stat:.4f}')
print(f'Hosmer-Lemeshow p-value: {hl_p_value:.4f}')

# **PSM步骤放在模型预测之后**

# Calculate propensity scores with logistic regression
log_reg = LogisticRegression(solver='liblinear', max_iter=1000)
log_reg.fit(X, y)

# Add propensity scores to the original dataset
df['propensity_score'] = log_reg.predict_proba(X)[:, 1]

# Separate treated and control groups
treated = df[df['ARGININE_DON'] == 1]
control = df[df['ARGININE_DON'] == 0]

# Use 1:2 nearest neighbor matching
nn = NearestNeighbors(n_neighbors=2)
nn.fit(control[['propensity_score']])

# Find nearest control samples
distances, indices = nn.kneighbors(treated[['propensity_score']])

# Apply caliper
caliper = 0.05
matched_indices = indices[distances.max(axis=1) <= caliper]

# Extract matched samples
matched_control = control.iloc[matched_indices.flatten()]
matched_data = pd.concat([treated.iloc[:len(matched_indices)], matched_control])

# Print the matching results
print(f"Matched treated samples: {len(treated)}, Matched control samples: {len(matched_control)}")

# **检查匹配后数据的数值平衡性**
def standardized_mean_diff(var):
    treated_mean = matched_data[matched_data['ARGININE_DON'] == 1][var].mean()
    control_mean = matched_data[matched_data['ARGININE_DON'] == 0][var].mean()
    pooled_std = np.sqrt(0.5 * (matched_data[matched_data['ARGININE_DON'] == 1][var].var() +
                                matched_data[matched_data['ARGININE_DON'] == 0][var].var()))
    return np.abs(treated_mean - control_mean) / pooled_std

for var in num_variables:
    smd = standardized_mean_diff(var)
    print(f"{var} 's Standardized Mean Difference is': {smd}")
