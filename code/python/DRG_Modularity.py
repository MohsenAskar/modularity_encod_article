##############################################################################
# This file is to implement LR, SVC, GBM on DRG datasets  datasets           #
##############################################################################
import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics._classification import classification_report
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearnex import patch_sklearn 
patch_sklearn()
from sklearn.svm import NuSVC
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import sem
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# import all datasets
#------------------------
raw_drg = pd.read_stata(r"C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\DRG\Datasets_For_Analysis\Selected_Variables_Final_Model_DRG.dta")
R1_drg = pd.read_stata(r"C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\DRG\Datasets_For_Analysis\DRG_R1_M24.dta")
##############################
#   1. Raw_DRG dataset       #
#############################
#####################################
# Dealing with imbalanced labels
#####################################
# Counting label classes 
#--------------------------
raw_drg.shape
raw_drg.groupby('readmission_90').count()
target_count = raw_drg.readmission_90.value_counts()
target_count.plot(kind='bar', title='Count (readmission_90)')
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1') # (2.94 : 1)
raw_drg.columns

# Downsampling
#------------------------
raw_drg.sort_values(by=['subject_id','hadm_id'])
raw_majority = raw_drg[raw_drg.readmission_90==0]
raw_minority = raw_drg[raw_drg.readmission_90==1]
raw_majority_downsampled = resample (raw_majority,
                                            replace=False, n_samples =32091, 
                                            random_state=123)
# Combine minority class with downsampled majority class
raw_downsampled = pd.concat([raw_majority_downsampled, raw_minority])
raw_downsampled.readmission_90.value_counts()

# Dropping id variables 
#-------------------------
raw_downsampled.drop(['subject_id','hadm_id'], axis=1, inplace=True)
raw_downsampled.columns
raw_downsampled.dtypes
# Change variable type of age_group and gender to include them in categorical set of data
#------------------------------------------------------------------------------------------
raw_downsampled['age_group'] = raw_downsampled['age_group'].astype(str)
raw_downsampled['gender'] = raw_downsampled['gender'].astype(str)

# Splitting data into X,y
#-----------------------------
features =[]
for column in raw_downsampled.columns:
    if column != 'readmission_90':
        features.append(column)
    X_raw = raw_downsampled[features]
    y_raw = raw_downsampled['readmission_90']
X_raw.columns
y_raw.shape
X_raw.shape# (64182, 10)

# Splitting X into numerical and categorical feature
#--------------------------------------------------------
X_raw_numerical =X_raw[['los', 'los_ED', 'no_ED_admissions']].copy()
X_raw_categorical = X_raw.select_dtypes(['object', 'int16', 'int8'])

# encoding 
#------------------
X_raw_categorical.columns
encoded_X_raw_categorical= pd.get_dummies(data=X_raw_categorical, columns=['admission_location', 'discharge_location','insurance','marital_status',
                                                                   'drg_code', 'gender','age_group'])

encoded_X_raw_categorical.shape
X_raw_numerical.reset_index(drop=True, inplace=True)
encoded_X_raw_categorical.reset_index(drop=True, inplace=True)
X2_raw= pd.concat([X_raw_numerical, encoded_X_raw_categorical], axis = 1)
X2_raw.shape #(64182, 1482)

# X split and standerdizing X_train, X_test 
#------------------------------------------------
X2_raw.dtypes
X_raw_train, X_raw_test, y_raw_train, y_raw_test = train_test_split(X2_raw,y_raw, test_size = 0.3, random_state = 0, stratify=y_raw)
X_raw_train.columns
X_raw_numerical_train = X_raw_train[['los', 'los_ED', 'no_ED_admissions']].copy() 
X_raw_numerical_train.shape
X_raw_categorical_train = X_raw_train.select_dtypes(['uint8','object'])
X_raw_categorical_train.shape
X_raw_numerical_test = X_raw_test[['los', 'los_ED', 'no_ED_admissions']].copy() 
X_raw_numerical_test.shape
X_raw_categorical_test = X_raw_test.select_dtypes(['uint8','object'])
X_raw_categorical_test.shape
for key in X_raw_numerical_train.keys():
    print (key)
for key in X_raw_categorical_train.keys():
    print (key)   
for key in X_raw_numerical_test.keys():
    print (key)  
for key in X_raw_categorical_test.keys():
    print (key)
    
scalar = StandardScaler()
scaled_X_raw_numerical_train= pd.DataFrame(scalar.fit_transform(X_raw_numerical_train), columns=X_raw_numerical_train.keys())
print(scaled_X_raw_numerical_train) # note here that we use fit_transform for X_train part
scaled_X_raw_numerical_test = pd.DataFrame(scalar.transform(X_raw_numerical_test), columns=X_raw_numerical_test.keys()) # note here that we only use the tranform (not fit_transform). That is to use the understanding from train data to test data #VERY IMPORTANT
print(scaled_X_raw_numerical_test)

# Recombining all X parts
#----------------------------------
scaled_X_raw_numerical_train.reset_index(drop=True, inplace=True)
X_raw_categorical_train.reset_index(drop=True, inplace=True)
X_raw_clean_train= pd.concat([scaled_X_raw_numerical_train, X_raw_categorical_train], axis = 1)
X_raw_clean_train.head()
X_raw_clean_train.shape 
scaled_X_raw_numerical_test.reset_index(drop=True, inplace=True)
X_raw_categorical_test.reset_index(drop=True, inplace=True)
X_raw_clean_test = pd.concat([scaled_X_raw_numerical_test, X_raw_categorical_test], axis = 1)
X_raw_clean_test.head()
X_raw_clean_test.shape 

###########################################
# Implementing Logistic regression Raw_DRG
###########################################
log_reg_raw = LogisticRegression(random_state=0, solver='liblinear')
log_reg_raw.fit(X_raw_clean_train, y_raw_train)

y_pred_log_reg_raw = log_reg_raw.predict(X_raw_clean_test)
print("Accuracy of logistic regression Raw:", metrics.accuracy_score(y_raw_test, y_pred_log_reg_raw))
print("Precision:",metrics.precision_score(y_raw_test, y_pred_log_reg_raw))
print("Recall:",metrics.recall_score(y_raw_test, y_pred_log_reg_raw))
print(classification_report(y_raw_test, y_pred_log_reg_raw))

# ROC curve
#--------------------
y_pred_proba_log_reg_raw = log_reg_raw.predict_proba(X_raw_clean_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_raw_test,  y_pred_proba_log_reg_raw)
AUC = metrics.roc_auc_score(y_raw_test, y_pred_proba_log_reg_raw)
plt.plot(fpr,tpr,label="Logistic Regression Raw, AUC="+str(AUC))
plt.legend(loc=4)
plt.show()

## Calculate confidence interval for the metrics
#----------------------------------------
y_pred_raw_log_reg_CI = np.array(y_pred_log_reg_raw)
y_raw_true = np.array(y_raw_test)
y_pred_proba_raw_log_reg_CI = np.array(y_pred_proba_log_reg_raw)

n_bootstraps = 1000
rng_seed = 42  
rng = np.random.RandomState(rng_seed)

bootstrapped_accuracies_raw = []
bootstrapped_precisions_raw = []
bootstrapped_recalls_raw = []
bootstrapped_f1s_raw = []
bootstrapped_aucs_raw = []

for i in range(n_bootstraps):
    indices_raw = rng.randint(0, len(y_pred_raw_log_reg_CI), len(y_pred_raw_log_reg_CI))
    if len(np.unique(y_raw_true[indices_raw])) < 2:
        continue

    accuracy_raw = accuracy_score(y_raw_true[indices_raw], y_pred_raw_log_reg_CI[indices_raw])
    precision_raw = precision_score(y_raw_true[indices_raw], y_pred_raw_log_reg_CI[indices_raw])
    recall_raw = recall_score(y_raw_true[indices_raw], y_pred_raw_log_reg_CI[indices_raw])
    f1_raw = f1_score(y_raw_true[indices_raw], y_pred_raw_log_reg_CI[indices_raw])
    auc_raw = roc_auc_score(y_raw_true[indices_raw], y_pred_proba_raw_log_reg_CI[indices_raw])

    bootstrapped_accuracies_raw.append(accuracy_raw)
    bootstrapped_precisions_raw.append(precision_raw)
    bootstrapped_recalls_raw.append(recall_raw)
    bootstrapped_f1s_raw.append(f1_raw)
    bootstrapped_aucs_raw.append(auc_raw)

# Computing the 95% confidence intervals for each metric
accuracy_ci_raw = (np.percentile(bootstrapped_accuracies_raw, 2.5), np.percentile(bootstrapped_accuracies_raw, 97.5))
precision_ci_raw = (np.percentile(bootstrapped_precisions_raw, 2.5), np.percentile(bootstrapped_precisions_raw, 97.5))
recall_ci_raw = (np.percentile(bootstrapped_recalls_raw, 2.5), np.percentile(bootstrapped_recalls_raw, 97.5))
f1_ci_raw = (np.percentile(bootstrapped_f1s_raw, 2.5), np.percentile(bootstrapped_f1s_raw, 97.5))
auc_ci_raw = (np.percentile(bootstrapped_aucs_raw, 2.5), np.percentile(bootstrapped_aucs_raw, 97.5))

print(f"Accuracy CI_raw: [{accuracy_ci_raw[0]:.3f} - {accuracy_ci_raw[1]:.3f}]")
print(f"Precision CI_raw: [{precision_ci_raw[0]:.3f} - {precision_ci_raw[1]:.3f}]")
print(f"Recall CI_raw: [{recall_ci_raw[0]:.3f} - {recall_ci_raw[1]:.3f}]")
print(f"F1-Score CI_raw: [{f1_ci_raw[0]:.3f} - {f1_ci_raw[1]:.3f}]")
print(f"AUC CI_raw: [{auc_ci_raw[0]:.3f} - {auc_ci_raw[1]:.3f}]")

# Classification report to calculate f1-score 
#---------------------------------------------------
cm_raw = metrics.confusion_matrix(y_raw_test, y_pred_log_reg_raw)
print('Confusion matrix:', cm_raw)

# plot the confusion materix (2*2 tabel)
#-----------------------------------------
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm_raw), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix LR_Raw', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

###########################################
# Implementing Non-linear SVC on Raw_DRG
###########################################
from sklearnex import patch_sklearn 
patch_sklearn()  # to speed up SVC training 
from sklearn.svm import NuSVC
SVM_raw = NuSVC(gamma="auto",probability=True, kernel = 'rbf', random_state=42)
SVM_raw.fit(X_raw_clean_train, y_raw_train)
y_pred_SVM_raw = SVM_raw.predict(X_raw_clean_test)
print("Accuracy of Support Vector Machine Raw:", metrics.accuracy_score(y_raw_test, y_pred_SVM_raw))
print("Precision:",metrics.precision_score(y_raw_test, y_pred_SVM_raw))
print("Recall:",metrics.recall_score(y_raw_test, y_pred_SVM_raw))
print(classification_report(y_raw_test, y_pred_SVM_raw))

# ROC curve
#--------------------
y_pred_proba_SVM_raw = SVM_raw.predict_proba(X_raw_clean_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_raw_test, y_pred_proba_SVM_raw)
AUC = metrics.roc_auc_score(y_raw_test, y_pred_proba_SVM_raw)
plt.plot(fpr,tpr,label="SVM_raw, AUC="+str(AUC))
plt.legend(loc=4)
plt.show()


## Calculate confidence interval for the metrics
#----------------------------------------
y_pred_raw_SVM_CI = np.array(y_pred_SVM_raw)
y_raw_true = np.array(y_raw_test)
y_pred_proba_raw_SVM_CI = np.array(y_pred_proba_SVM_raw)

n_bootstraps = 1000
rng_seed = 42  
rng = np.random.RandomState(rng_seed)

bootstrapped_accuracies_raw = []
bootstrapped_precisions_raw = []
bootstrapped_recalls_raw = []
bootstrapped_f1s_raw = []
bootstrapped_aucs_raw = []

for i in range(n_bootstraps):
    indices_raw = rng.randint(0, len(y_pred_raw_SVM_CI), len(y_pred_raw_SVM_CI))
    if len(np.unique(y_raw_true[indices_raw])) < 2:
        continue

    accuracy_raw = accuracy_score(y_raw_true[indices_raw], y_pred_raw_SVM_CI[indices_raw])
    precision_raw = precision_score(y_raw_true[indices_raw], y_pred_raw_SVM_CI[indices_raw])
    recall_raw = recall_score(y_raw_true[indices_raw], y_pred_raw_SVM_CI[indices_raw])
    f1_raw = f1_score(y_raw_true[indices_raw], y_pred_raw_SVM_CI[indices_raw])
    auc_raw = roc_auc_score(y_raw_true[indices_raw], y_pred_proba_raw_SVM_CI[indices_raw])

    bootstrapped_accuracies_raw.append(accuracy_raw)
    bootstrapped_precisions_raw.append(precision_raw)
    bootstrapped_recalls_raw.append(recall_raw)
    bootstrapped_f1s_raw.append(f1_raw)
    bootstrapped_aucs_raw.append(auc_raw)

# Computing the 95% confidence intervals for each metric
accuracy_ci_raw = (np.percentile(bootstrapped_accuracies_raw, 2.5), np.percentile(bootstrapped_accuracies_raw, 97.5))
precision_ci_raw = (np.percentile(bootstrapped_precisions_raw, 2.5), np.percentile(bootstrapped_precisions_raw, 97.5))
recall_ci_raw = (np.percentile(bootstrapped_recalls_raw, 2.5), np.percentile(bootstrapped_recalls_raw, 97.5))
f1_ci_raw = (np.percentile(bootstrapped_f1s_raw, 2.5), np.percentile(bootstrapped_f1s_raw, 97.5))
auc_ci_raw = (np.percentile(bootstrapped_aucs_raw, 2.5), np.percentile(bootstrapped_aucs_raw, 97.5))

print(f"Accuracy CI_raw: [{accuracy_ci_raw[0]:.3f} - {accuracy_ci_raw[1]:.3f}]")
print(f"Precision CI_raw: [{precision_ci_raw[0]:.3f} - {precision_ci_raw[1]:.3f}]")
print(f"Recall CI_raw: [{recall_ci_raw[0]:.3f} - {recall_ci_raw[1]:.3f}]")
print(f"F1-Score CI_raw: [{f1_ci_raw[0]:.3f} - {f1_ci_raw[1]:.3f}]")
print(f"AUC CI_raw: [{auc_ci_raw[0]:.3f} - {auc_ci_raw[1]:.3f}]")


# plot the confusion materix (2*2 tabel)
#-----------------------------------------
cm_raw= metrics.confusion_matrix(y_raw_test, y_pred_SVM_raw)
print('Confusion matrix:', cm_raw)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm_raw), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix SVM_Raw', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


#################################
# Implementing GBM on Raw_DRG
################################
GBM_raw = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
max_depth=1, random_state=0)
GBM_raw.fit(X_raw_clean_train, y_raw_train)
y_pred_GBM_raw = GBM_raw.predict(X_raw_clean_test)
print("Accuracy of GBM_raw:", metrics.accuracy_score(y_raw_test, y_pred_GBM_raw))
print("Precision:",metrics.precision_score(y_raw_test, y_pred_GBM_raw))
print("Recall:",metrics.recall_score(y_raw_test, y_pred_GBM_raw))
print(classification_report(y_raw_test, y_pred_GBM_raw))

# AUC
#-------------------------------
y_pred_prob_GBM_raw = GBM_raw.predict_proba(X_raw_clean_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_raw_test, y_pred_prob_GBM_raw)
AUC = metrics.roc_auc_score(y_raw_test, y_pred_prob_GBM_raw)
plt.plot(fpr,tpr,label="GBM_raw , AUC="+str(AUC))
plt.legend(loc=4)
plt.show()

## Calculate confidence interval for the metrics
#----------------------------------------
y_pred_raw_GBM_CI = np.array(y_pred_GBM_raw)
y_raw_true = np.array(y_raw_test)
y_pred_proba_raw_GBM_CI = np.array(y_pred_prob_GBM_raw)

n_bootstraps = 1000
rng_seed = 42  
rng = np.random.RandomState(rng_seed)

bootstrapped_accuracies_raw = []
bootstrapped_precisions_raw = []
bootstrapped_recalls_raw = []
bootstrapped_f1s_raw = []
bootstrapped_aucs_raw = []

for i in range(n_bootstraps):
    indices_raw = rng.randint(0, len(y_pred_raw_GBM_CI), len(y_pred_raw_GBM_CI))
    if len(np.unique(y_raw_true[indices_raw])) < 2:
        continue

    accuracy_raw = accuracy_score(y_raw_true[indices_raw], y_pred_raw_GBM_CI[indices_raw])
    precision_raw = precision_score(y_raw_true[indices_raw], y_pred_raw_GBM_CI[indices_raw])
    recall_raw = recall_score(y_raw_true[indices_raw], y_pred_raw_GBM_CI[indices_raw])
    f1_raw = f1_score(y_raw_true[indices_raw], y_pred_raw_GBM_CI[indices_raw])
    auc_raw = roc_auc_score(y_raw_true[indices_raw], y_pred_proba_raw_GBM_CI[indices_raw])

    bootstrapped_accuracies_raw.append(accuracy_raw)
    bootstrapped_precisions_raw.append(precision_raw)
    bootstrapped_recalls_raw.append(recall_raw)
    bootstrapped_f1s_raw.append(f1_raw)
    bootstrapped_aucs_raw.append(auc_raw)

# Computing the 95% confidence intervals for each metric
accuracy_ci_raw = (np.percentile(bootstrapped_accuracies_raw, 2.5), np.percentile(bootstrapped_accuracies_raw, 97.5))
precision_ci_raw = (np.percentile(bootstrapped_precisions_raw, 2.5), np.percentile(bootstrapped_precisions_raw, 97.5))
recall_ci_raw = (np.percentile(bootstrapped_recalls_raw, 2.5), np.percentile(bootstrapped_recalls_raw, 97.5))
f1_ci_raw = (np.percentile(bootstrapped_f1s_raw, 2.5), np.percentile(bootstrapped_f1s_raw, 97.5))
auc_ci_raw = (np.percentile(bootstrapped_aucs_raw, 2.5), np.percentile(bootstrapped_aucs_raw, 97.5))

print(f"Accuracy CI_raw: [{accuracy_ci_raw[0]:.3f} - {accuracy_ci_raw[1]:.3f}]")
print(f"Precision CI_raw: [{precision_ci_raw[0]:.3f} - {precision_ci_raw[1]:.3f}]")
print(f"Recall CI_raw: [{recall_ci_raw[0]:.3f} - {recall_ci_raw[1]:.3f}]")
print(f"F1-Score CI_raw: [{f1_ci_raw[0]:.3f} - {f1_ci_raw[1]:.3f}]")
print(f"AUC CI_raw: [{auc_ci_raw[0]:.3f} - {auc_ci_raw[1]:.3f}]")

#############################
#   2. R1_DRG dataset       #
#############################
#####################################
# Dealing with imbalanced labels
#####################################
# Counting label classes 
#--------------------------
R1_drg.shape
R1_drg.groupby('readmission_90').count()
target_count = R1_drg.readmission_90.value_counts()
target_count.plot(kind='bar', title='Count (readmission_90)')
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1') #(6.3:1)
R1_drg.columns

# Downsampling
#------------------------
R1_drg.sort_values(by=['subject_id','hadm_id'])
R1_majority = R1_drg[R1_drg.readmission_90==0]
R1_minority = R1_drg[R1_drg.readmission_90==1]
R1_majority_downsampled = resample (R1_majority,
                                            replace=False, n_samples =12448, 
                                            random_state=123)
# Combine minority class with downsampled majority class
R1_downsampled = pd.concat([R1_majority_downsampled, R1_minority])
R1_downsampled.readmission_90.value_counts()

# Dropping id variables 
#-------------------------
R1_downsampled.drop(['subject_id','hadm_id'], axis=1, inplace=True)
R1_downsampled.columns
R1_downsampled.dtypes
# Change variable type of age_group and gender to include them in categorical set of data
#------------------------------------------------------------------------------------------
R1_downsampled['age_group'] = R1_downsampled['age_group'].astype(str)
R1_downsampled['gender'] = R1_downsampled['gender'].astype(str)

# Splitting data into X,y
#-----------------------------
features =[]
for column in R1_downsampled.columns:
    if column != 'readmission_90':
        features.append(column)
    X_R1 = R1_downsampled[features]
    y_R1 = R1_downsampled['readmission_90']
X_R1.columns
y_R1.shape
X_R1.shape# (24896,10)

# Splitting X into numerical and categorical feature
#--------------------------------------------------------
X_R1_numerical =X_R1[['los', 'los_ED', 'no_ED_admissions']].copy()
X_R1_categorical = X_R1.select_dtypes(['object', 'int16', 'int8'])

# encoding 
#------------------
X_R1_categorical.columns
encoded_X_R1_categorical= pd.get_dummies(data=X_R1_categorical, columns=['admission_location', 'discharge_location','insurance','marital_status',
                                                                   'modularity_class', 'gender','age_group'])

encoded_X_R1_categorical.shape
X_R1_numerical.reset_index(drop=True, inplace=True)
encoded_X_R1_categorical.reset_index(drop=True, inplace=True)
X2_R1= pd.concat([X_R1_numerical, encoded_X_R1_categorical], axis = 1)
X2_R1.shape #(24896, 64)

# X split and standerdizing X_train, X_test 
#------------------------------------------------
X2_R1.dtypes
X_R1_train, X_R1_test, y_R1_train, y_R1_test = train_test_split(X2_R1,y_R1, test_size = 0.3, random_state = 0, stratify=y_R1)
X_R1_train.columns
X_R1_numerical_train = X_R1_train[['los', 'los_ED', 'no_ED_admissions']].copy() 
X_R1_numerical_train.shape
X_R1_categorical_train = X_R1_train.select_dtypes(['uint8','object'])
X_R1_categorical_train.shape
X_R1_numerical_test = X_R1_test[['los', 'los_ED', 'no_ED_admissions']].copy() 
X_R1_numerical_test.shape
X_R1_categorical_test = X_R1_test.select_dtypes(['uint8','object'])
X_R1_categorical_test.shape
for key in X_R1_numerical_train.keys():
    print (key)
for key in X_R1_categorical_train.keys():
    print (key)   
for key in X_R1_numerical_test.keys():
    print (key)  
for key in X_R1_categorical_test.keys():
    print (key)
    
scalar = StandardScaler()
scaled_X_R1_numerical_train= pd.DataFrame(scalar.fit_transform(X_R1_numerical_train), columns=X_R1_numerical_train.keys())
print(scaled_X_R1_numerical_train) # note here that we use fit_transform for X_train part
scaled_X_R1_numerical_test = pd.DataFrame(scalar.transform(X_R1_numerical_test), columns=X_R1_numerical_test.keys()) # note here that we only use the tranform (not fit_transform). That is to use the understanding from train data to test data #VERY IMPORTANT
print(scaled_X_R1_numerical_test)

# Recombining all X parts
#----------------------------------
scaled_X_R1_numerical_train.reset_index(drop=True, inplace=True)
X_R1_categorical_train.reset_index(drop=True, inplace=True)
X_R1_clean_train= pd.concat([scaled_X_R1_numerical_train, X_R1_categorical_train], axis = 1)
X_R1_clean_train.head()
X_R1_clean_train.shape 
scaled_X_R1_numerical_test.reset_index(drop=True, inplace=True)
X_R1_categorical_test.reset_index(drop=True, inplace=True)
X_R1_clean_test = pd.concat([scaled_X_R1_numerical_test, X_R1_categorical_test], axis = 1)
X_R1_clean_test.head()
X_R1_clean_test.shape #(7469,64)

##############################################
# Implementing Logistic regression R1_DRG
#############################################
log_reg_R1 = LogisticRegression(random_state=0, solver='liblinear')
log_reg_R1.fit(X_R1_clean_train, y_R1_train)

y_pred_log_reg_R1 = log_reg_R1.predict(X_R1_clean_test)
print("Accuracy of logistic regression R1:", metrics.accuracy_score(y_R1_test, y_pred_log_reg_R1))
print("Precision:",metrics.precision_score(y_R1_test, y_pred_log_reg_R1))
print("Recall:",metrics.recall_score(y_R1_test, y_pred_log_reg_R1))
print(classification_report(y_R1_test, y_pred_log_reg_R1))

# get model parameters
#----------------------------
log_reg_R1.get_params()
# ROC curve
#--------------------
y_pred_proba_log_reg_R1 = log_reg_R1.predict_proba(X_R1_clean_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_R1_test,  y_pred_proba_log_reg_R1)
AUC = metrics.roc_auc_score(y_R1_test, y_pred_proba_log_reg_R1)
plt.plot(fpr,tpr,label="Logistic Regression R1, AUC="+str(AUC))
plt.legend(loc=4)
plt.show()

## Calculate confidence interval for the metrics
#----------------------------------------
y_pred_R1_log_reg_CI = np.array(y_pred_log_reg_R1)
y_R1_true = np.array(y_R1_test)
y_pred_proba_R1_log_reg_CI = np.array(y_pred_proba_log_reg_R1)

n_bootstraps = 1000
rng_seed = 42  
rng = np.random.RandomState(rng_seed)

bootstrapped_accuracies_R1 = []
bootstrapped_precisions_R1 = []
bootstrapped_recalls_R1 = []
bootstrapped_f1s_R1 = []
bootstrapped_aucs_R1 = []

for i in range(n_bootstraps):
    indices_R1 = rng.randint(0, len(y_pred_R1_log_reg_CI), len(y_pred_R1_log_reg_CI))
    if len(np.unique(y_R1_true[indices_R1])) < 2:
        continue

    accuracy_R1 = accuracy_score(y_R1_true[indices_R1], y_pred_R1_log_reg_CI[indices_R1])
    precision_R1 = precision_score(y_R1_true[indices_R1], y_pred_R1_log_reg_CI[indices_R1])
    recall_R1 = recall_score(y_R1_true[indices_R1], y_pred_R1_log_reg_CI[indices_R1])
    f1_R1 = f1_score(y_R1_true[indices_R1], y_pred_R1_log_reg_CI[indices_R1])
    auc_R1 = roc_auc_score(y_R1_true[indices_R1], y_pred_proba_R1_log_reg_CI[indices_R1])

    bootstrapped_accuracies_R1.append(accuracy_R1)
    bootstrapped_precisions_R1.append(precision_R1)
    bootstrapped_recalls_R1.append(recall_R1)
    bootstrapped_f1s_R1.append(f1_R1)
    bootstrapped_aucs_R1.append(auc_R1)

# Computing the 95% confidence intervals for each metric
accuracy_ci_R1 = (np.percentile(bootstrapped_accuracies_R1, 2.5), np.percentile(bootstrapped_accuracies_R1, 97.5))
precision_ci_R1 = (np.percentile(bootstrapped_precisions_R1, 2.5), np.percentile(bootstrapped_precisions_R1, 97.5))
recall_ci_R1 = (np.percentile(bootstrapped_recalls_R1, 2.5), np.percentile(bootstrapped_recalls_R1, 97.5))
f1_ci_R1 = (np.percentile(bootstrapped_f1s_R1, 2.5), np.percentile(bootstrapped_f1s_R1, 97.5))
auc_ci_R1 = (np.percentile(bootstrapped_aucs_R1, 2.5), np.percentile(bootstrapped_aucs_R1, 97.5))

print(f"Accuracy CI_R1: [{accuracy_ci_R1[0]:.3f} - {accuracy_ci_R1[1]:.3f}]")
print(f"Precision CI_R1: [{precision_ci_R1[0]:.3f} - {precision_ci_R1[1]:.3f}]")
print(f"Recall CI_R1: [{recall_ci_R1[0]:.3f} - {recall_ci_R1[1]:.3f}]")
print(f"F1-Score CI_R1: [{f1_ci_R1[0]:.3f} - {f1_ci_R1[1]:.3f}]")
print(f"AUC CI_R1: [{auc_ci_R1[0]:.3f} - {auc_ci_R1[1]:.3f}]")



# Classification report to calculate f1-score 
#---------------------------------------------------
cm_R1 = metrics.confusion_matrix(y_R1_test, y_pred_log_reg_R1)
print('Confusion matrix:', cm_R1)

# plot the confusion materix (2*2 tabel)
#-----------------------------------------
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm_R1), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix LR_R1', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

###########################################
# Implementing Non-linear SVC R1_DRG
###########################################
patch_sklearn()  # to speed up SVC training 
from sklearn.svm import NuSVC
SVM_R1 = NuSVC(gamma="auto",probability=True, kernel = 'rbf', random_state=42)
SVM_R1.fit(X_R1_clean_train, y_R1_train)
y_pred_SVM_R1 = SVM_R1.predict(X_R1_clean_test)
print("Accuracy of Support Vector Machine R1_DRG:", metrics.accuracy_score(y_R1_test, y_pred_SVM_R1))
print("Precision:",metrics.precision_score(y_R1_test, y_pred_SVM_R1))
print("Recall:",metrics.recall_score(y_R1_test, y_pred_SVM_R1))
print(classification_report(y_R1_test, y_pred_SVM_R1))

# get model parameters
#--------------------------------------
SVM_R1.get_params()

# ROC curve
#--------------------
y_pred_proba_SVM_R1 = SVM_R1.predict_proba(X_R1_clean_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_R1_test, y_pred_proba_SVM_R1)
AUC = metrics.roc_auc_score(y_R1_test, y_pred_proba_SVM_R1)
plt.plot(fpr,tpr,label="SVM_R1, AUC="+str(AUC))
plt.legend(loc=4)
plt.show()

## Calculate confidence interval for the metrics
#----------------------------------------
y_pred_R1_SVM_CI = np.array(y_pred_SVM_R1)
y_R1_true = np.array(y_R1_test)
y_pred_proba_R1_SVM_CI = np.array(y_pred_proba_SVM_R1)

n_bootstraps = 1000
rng_seed = 42  
rng = np.random.RandomState(rng_seed)

bootstrapped_accuracies_R1 = []
bootstrapped_precisions_R1 = []
bootstrapped_recalls_R1 = []
bootstrapped_f1s_R1 = []
bootstrapped_aucs_R1 = []

for i in range(n_bootstraps):
    indices_R1 = rng.randint(0, len(y_pred_R1_SVM_CI), len(y_pred_R1_SVM_CI))
    if len(np.unique(y_R1_true[indices_R1])) < 2:
        continue

    accuracy_R1 = accuracy_score(y_R1_true[indices_R1], y_pred_R1_SVM_CI[indices_R1])
    precision_R1 = precision_score(y_R1_true[indices_R1], y_pred_R1_SVM_CI[indices_R1])
    recall_R1 = recall_score(y_R1_true[indices_R1], y_pred_R1_SVM_CI[indices_R1])
    f1_R1 = f1_score(y_R1_true[indices_R1], y_pred_R1_SVM_CI[indices_R1])
    auc_R1 = roc_auc_score(y_R1_true[indices_R1], y_pred_proba_R1_SVM_CI[indices_R1])

    bootstrapped_accuracies_R1.append(accuracy_R1)
    bootstrapped_precisions_R1.append(precision_R1)
    bootstrapped_recalls_R1.append(recall_R1)
    bootstrapped_f1s_R1.append(f1_R1)
    bootstrapped_aucs_R1.append(auc_R1)

# Computing the 95% confidence intervals for each metric
accuracy_ci_R1 = (np.percentile(bootstrapped_accuracies_R1, 2.5), np.percentile(bootstrapped_accuracies_R1, 97.5))
precision_ci_R1 = (np.percentile(bootstrapped_precisions_R1, 2.5), np.percentile(bootstrapped_precisions_R1, 97.5))
recall_ci_R1 = (np.percentile(bootstrapped_recalls_R1, 2.5), np.percentile(bootstrapped_recalls_R1, 97.5))
f1_ci_R1 = (np.percentile(bootstrapped_f1s_R1, 2.5), np.percentile(bootstrapped_f1s_R1, 97.5))
auc_ci_R1 = (np.percentile(bootstrapped_aucs_R1, 2.5), np.percentile(bootstrapped_aucs_R1, 97.5))

print(f"Accuracy CI_R1: [{accuracy_ci_R1[0]:.3f} - {accuracy_ci_R1[1]:.3f}]")
print(f"Precision CI_R1: [{precision_ci_R1[0]:.3f} - {precision_ci_R1[1]:.3f}]")
print(f"Recall CI_R1: [{recall_ci_R1[0]:.3f} - {recall_ci_R1[1]:.3f}]")
print(f"F1-Score CI_R1: [{f1_ci_R1[0]:.3f} - {f1_ci_R1[1]:.3f}]")
print(f"AUC CI_R1: [{auc_ci_R1[0]:.3f} - {auc_ci_R1[1]:.3f}]")

# plot the confusion materix (2*2 tabel)
#-----------------------------------------
cm_R1= metrics.confusion_matrix(y_R1_test, y_pred_SVM_R1)
print('Confusion matrix:', cm_R1)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm_R1), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix SVM_R1', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

##############################
# Implementing GBM R1_DRG
##############################
GBM_R1 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
max_depth=1, random_state=0)
GBM_R1.fit(X_R1_clean_train, y_R1_train)
y_pred_GBM_R1 = GBM_R1.predict(X_R1_clean_test)
print("Accuracy of GBM_R1:", metrics.accuracy_score(y_R1_test, y_pred_GBM_R1))
print("Precision:",metrics.precision_score(y_R1_test, y_pred_GBM_R1))
print("Recall:",metrics.recall_score(y_R1_test, y_pred_GBM_R1))
print(classification_report(y_R1_test, y_pred_GBM_R1))
# get model parametrs
#-----------------------
GBM_R1.get_params()

# AUC
#-------------------------------
y_pred_prob_GBM_R1 = GBM_R1.predict_proba(X_R1_clean_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_R1_test, y_pred_prob_GBM_R1)
AUC = metrics.roc_auc_score(y_R1_test, y_pred_prob_GBM_R1)
plt.plot(fpr,tpr,label="GBM_R1 , AUC="+str(AUC))
plt.legend(loc=4)
plt.show()

# Feature importance
#-----------------------
feature_importance = GBM_R1.feature_importances_
feature_import = pd.Series(GBM_R1.feature_importances_, index=X2_R1.columns)
feature_import.sort_values(ascending=False)
feature_import.nlargest(10).plot(kind='barh')
plt.title("Feature importance GBM for R1 dataset")
plt.show()

## Calculate confidence interval for the metrics
#----------------------------------------
y_pred_R1_GBM_CI = np.array(y_pred_GBM_R1)
y_R1_true = np.array(y_R1_test)
y_pred_proba_R1_GBM_CI = np.array(y_pred_prob_GBM_R1)

n_bootstraps = 1000
rng_seed = 42  
rng = np.random.RandomState(rng_seed)

bootstrapped_accuracies_R1 = []
bootstrapped_precisions_R1 = []
bootstrapped_recalls_R1 = []
bootstrapped_f1s_R1 = []
bootstrapped_aucs_R1 = []

for i in range(n_bootstraps):
    indices_R1 = rng.randint(0, len(y_pred_R1_GBM_CI), len(y_pred_R1_GBM_CI))
    if len(np.unique(y_R1_true[indices_R1])) < 2:
        continue

    accuracy_R1 = accuracy_score(y_R1_true[indices_R1], y_pred_R1_GBM_CI[indices_R1])
    precision_R1 = precision_score(y_R1_true[indices_R1], y_pred_R1_GBM_CI[indices_R1])
    recall_R1 = recall_score(y_R1_true[indices_R1], y_pred_R1_GBM_CI[indices_R1])
    f1_R1 = f1_score(y_R1_true[indices_R1], y_pred_R1_GBM_CI[indices_R1])
    auc_R1 = roc_auc_score(y_R1_true[indices_R1], y_pred_proba_R1_GBM_CI[indices_R1])

    bootstrapped_accuracies_R1.append(accuracy_R1)
    bootstrapped_precisions_R1.append(precision_R1)
    bootstrapped_recalls_R1.append(recall_R1)
    bootstrapped_f1s_R1.append(f1_R1)
    bootstrapped_aucs_R1.append(auc_R1)

# Computing the 95% confidence intervals for each metric
accuracy_ci_R1 = (np.percentile(bootstrapped_accuracies_R1, 2.5), np.percentile(bootstrapped_accuracies_R1, 97.5))
precision_ci_R1 = (np.percentile(bootstrapped_precisions_R1, 2.5), np.percentile(bootstrapped_precisions_R1, 97.5))
recall_ci_R1 = (np.percentile(bootstrapped_recalls_R1, 2.5), np.percentile(bootstrapped_recalls_R1, 97.5))
f1_ci_R1 = (np.percentile(bootstrapped_f1s_R1, 2.5), np.percentile(bootstrapped_f1s_R1, 97.5))
auc_ci_R1 = (np.percentile(bootstrapped_aucs_R1, 2.5), np.percentile(bootstrapped_aucs_R1, 97.5))

print(f"Accuracy CI_R1: [{accuracy_ci_R1[0]:.3f} - {accuracy_ci_R1[1]:.3f}]")
print(f"Precision CI_R1: [{precision_ci_R1[0]:.3f} - {precision_ci_R1[1]:.3f}]")
print(f"Recall CI_R1: [{recall_ci_R1[0]:.3f} - {recall_ci_R1[1]:.3f}]")
print(f"F1-Score CI_R1: [{f1_ci_R1[0]:.3f} - {f1_ci_R1[1]:.3f}]")
print(f"AUC CI_R1: [{auc_ci_R1[0]:.3f} - {auc_ci_R1[1]:.3f}]")


# Plot all R1_DRG AUCs
#----------------------
# log. reg.
fpr, tpr, thresh = metrics.roc_curve(y_R1_test, y_pred_proba_log_reg_R1)
AUC = metrics.roc_auc_score(y_R1_test, y_pred_proba_log_reg_R1)
plt.plot(fpr,tpr,label="Log.Reg., AUC="+str(AUC))

# SVM
fpr, tpr, thresh = metrics.roc_curve(y_R1_test, y_pred_proba_SVM_R1)
AUC = metrics.roc_auc_score(y_R1_test, y_pred_proba_SVM_R1)
plt.plot(fpr,tpr,label="SVM, AUC="+str(AUC))

# GBM
fpr, tpr, thresh = metrics.roc_curve(y_R1_test, y_pred_prob_GBM_R1)
AUC = metrics.roc_auc_score(y_R1_test, y_pred_prob_GBM_R1)
plt.plot(fpr,tpr,label="GBM, AUC="+str(AUC))
plt.title('AUC of Different Models R1_DRG', y=1.1)
plt.legend(loc=0)