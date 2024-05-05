########################################################################################
# This file is to implement Gradient Boosting Machine GBM on the different datasets    #
########################################################################################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics._classification import classification_report
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import sem
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# import all datasets
#------------------------
R1 = pd.read_stata(r'C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Final_Datasets_To_Implement_Models\R1_M8.dta')
R08 = pd.read_stata(r'C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Final_Datasets_To_Implement_Models\R0.8_M16.dta')
R06 = pd.read_stata(r'C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Final_Datasets_To_Implement_Models\R0.6_M32.dta')
R05 = pd.read_stata(r'C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Final_Datasets_To_Implement_Models\R0.5_M47.dta')
R01 = pd.read_stata(r'C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Final_Datasets_To_Implement_Models\R0.1_M314.dta')
R001 = pd.read_stata(r'C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Final_Datasets_To_Implement_Models\R0.01_M1078.dta')
CCS = pd.read_stata(r'C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Final_Datasets_To_Implement_Models\ccs.dta')
ICD_HIGHEST = pd.read_stata(r'C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Final_Datasets_To_Implement_Models\Data_ICD_Recoded_Highest_Hirarchy_90_days.dta')
raw = pd.read_stata(r"C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Final_Datasets_To_Implement_Models\Selected_Variables_Final_Model_90_days.dta")

##########################
#   1. Raw dataset       #
##########################
#####################################
# Dealing with imbalanced labels
#####################################
# Counting label classes 
#--------------------------
raw.shape
raw.groupby('readmission_90').count()
target_count = raw.readmission_90.value_counts()
target_count.plot(kind='bar', title='Count (readmission_90)')
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1') # (5.71 : 1)
raw.columns

# Downsampling
#------------------------
raw.sort_values(by=['subject_id','hadm_id'])
raw_majority = raw[raw.readmission_90==0]
raw_minority = raw[raw.readmission_90==1]
raw_majority_downsampled = resample (raw_majority,
                                            replace=False, n_samples =65925, 
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
X_raw.shape# ok

# Splitting X into numerical and categorical feature
#--------------------------------------------------------
X_raw_numerical =X_raw[['los', 'los_ED', 'no_ED_admissions']].copy()
X_raw_categorical = X_raw.select_dtypes(['object', 'int16', 'int8'])

# encoding 
#------------------
X_raw_categorical.columns
encoded_X_raw_categorical= pd.get_dummies(data=X_raw_categorical, columns=['admission_location', 'discharge_location','insurance','marital_status',
                                                                   'icd9_code', 'gender','age_group'])

encoded_X_raw_categorical.shape
X_raw_numerical.reset_index(drop=True, inplace=True)
encoded_X_raw_categorical.reset_index(drop=True, inplace=True)
X2_raw= pd.concat([X_raw_numerical, encoded_X_raw_categorical], axis = 1)
X2_raw.shape #(131850, 4403)

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
print(scaled_X_raw_numerical_train) # note here that we use fit_transform for X_train part and only transform for X_test
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

#############################
# Implementing GBM on Raw
############################
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

# Calculating CI 95% for the metrics 
###############################################
y_pred_raw_CI = np.array(y_pred_prob_GBM_raw)
y_raw_true = np.array(y_raw_test)

n_bootstraps = 1000
rng_seed = 42  
rng = np.random.RandomState(rng_seed)

bootstrapped_accuracies_raw = []
bootstrapped_precisions_raw = []
bootstrapped_recalls_raw = []
bootstrapped_f1s_raw = []
bootstrapped_aucs_raw = []

for i in range(n_bootstraps):
    # bootstrap by sampling with replacement on the prediction indices
    indices_raw = rng.randint(0, len(y_pred_raw_CI), len(y_pred_raw_CI))
    if len(np.unique(y_raw_true[indices_raw])) < 2:
        # We need at least one positive and one negative sample for the metrics
        # to be defined: reject the sample
        continue
    
    binary_predictions_raw = (y_pred_raw_CI[indices_raw] > 0.5).astype(int)
    
    accuracy_raw = accuracy_score(y_raw_true[indices_raw], binary_predictions_raw)
    precision_raw= precision_score(y_raw_true[indices_raw], binary_predictions_raw)
    recall_raw = recall_score(y_raw_true[indices_raw], binary_predictions_raw)
    f1_raw = f1_score(y_raw_true[indices_raw], binary_predictions_raw)
    auc_raw = roc_auc_score(y_raw_true[indices_raw], y_pred_raw_CI[indices_raw])
    
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
cm_raw = metrics.confusion_matrix(y_raw_test, y_pred_GBM_raw)
print('Confusion matrix GBM_raw:', cm_raw)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm_raw), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

##########################
#   2. R1 dataset        #
##########################
#####################################
# Dealing with imbalanced labels
#####################################
# Counting label classes 
#--------------------------
R1.shape
R1.groupby('readmission_90').count()
target_count = R1.readmission_90.value_counts()
target_count.plot(kind='bar', title='Count (readmission_90)')
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1') #(6.3:1)
R1.columns

# Downsampling
#------------------------
R1.sort_values(by=['subject_id','hadm_id'])
R1_majority = R1[R1.readmission_90==0]
R1_minority = R1[R1.readmission_90==1]
R1_majority_downsampled = resample (R1_majority,
                                            replace=False, n_samples =15070, 
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
X_R1.shape# ok

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
X2_R1.shape #(30140, 56)

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
X_R1_clean_test.shape 

#########################
# Implementing GBM on R1
#########################
GBM_R1 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
max_depth=1, random_state=0)
GBM_R1.fit(X_R1_clean_train, y_R1_train)
params = GBM_R1.get_params()
# Convert the dictionary of parameters to a DataFrame for better visualization
params_df = pd.DataFrame(list(params.items()), columns=['Parameter', 'Value'])
# Print the DataFrame in a pretty tabular format
print(params_df.to_string(index=False))
y_pred_GBM_R1 = GBM_R1.predict(X_R1_clean_test)
print("Accuracy of GBM_R1:", metrics.accuracy_score(y_R1_test, y_pred_GBM_R1))
print("Precision:",metrics.precision_score(y_R1_test, y_pred_GBM_R1))
print("Recall:",metrics.recall_score(y_R1_test, y_pred_GBM_R1))
print(classification_report(y_R1_test, y_pred_GBM_R1))

# Feature importances
#-----------------------
feature_importance = GBM_R1.feature_importances_
feature_import = pd.Series(GBM_R1.feature_importances_, index=X2_R1.columns)
feature_import.sort_values(ascending=False)
feature_import.nlargest(10).plot(kind='barh')
plt.title("Feature importance GBM for R1 dataset")
plt.show() # Modularity_class 2 was the most important among the modularity classes
# going back to see which codes are under modularity class 2 we found the majority of codes are under (I) group of ICD-10 which 
# indicates Diseases of the circulatory system. This is an example of enhancing interpretation of the model 
# we can also demonstrate a pearson correlaition materix to show the correlation between modularity classes and the outcome 

#AUC
#-------------------------------
y_pred_prob_GBM_R1 = GBM_R1.predict_proba(X_R1_clean_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_R1_test, y_pred_prob_GBM_R1)
AUC = metrics.roc_auc_score(y_R1_test, y_pred_prob_GBM_R1)
plt.plot(fpr,tpr,label="GBM_R1 , AUC="+str(AUC))
plt.legend(loc=4)
plt.show()

# Calculating CI 95% for the metrics 
###############################################
y_pred_R1_CI = np.array(y_pred_prob_GBM_R1)
y_R1_true = np.array(y_R1_test)

n_bootstraps = 1000
rng_seed = 42  
rng = np.random.RandomState(rng_seed)

bootstrapped_accuracies_R1 = []
bootstrapped_precisions_R1 = []
bootstrapped_recalls_R1 = []
bootstrapped_f1s_R1 = []
bootstrapped_aucs_R1 = []

for i in range(n_bootstraps):
    # bootstrap by sampling with replacement on the prediction indices
    indices_R1 = rng.randint(0, len(y_pred_R1_CI), len(y_pred_R1_CI))
    if len(np.unique(y_R1_true[indices_R1])) < 2:
        # We need at least one positive and one negative sample for the metrics
        # to be defined: reject the sample
        continue
    
    binary_predictions_R1 = (y_pred_R1_CI[indices_R1] > 0.5).astype(int)
    
    accuracy_R1 = accuracy_score(y_R1_true[indices_R1], binary_predictions_R1)
    precision_R1= precision_score(y_R1_true[indices_R1], binary_predictions_R1)
    recall_R1 = recall_score(y_R1_true[indices_R1], binary_predictions_R1)
    f1_R1 = f1_score(y_R1_true[indices_R1], binary_predictions_R1)
    auc_R1 = roc_auc_score(y_R1_true[indices_R1], y_pred_R1_CI[indices_R1])
    
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
cm_R1 = metrics.confusion_matrix(y_R1_test, y_pred_GBM_R1)
print('Confusion matrix GBM_R1:', cm_R1)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm_R1), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

##########################
#   3. R08 dataset       #
##########################
#####################################
# Dealing with imbalanced labels
#####################################
# Counting label classes 
#--------------------------
R08.shape
R08.groupby('readmission_90').count()
target_count = R08.readmission_90.value_counts()
target_count.plot(kind='bar', title='Count (readmission_90)')
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1') #(5.99 : 1)
R08.columns

# Downsampling
#------------------------
R08.sort_values(by=['subject_id','hadm_id'])
R08_majority = R08[R08.readmission_90==0]
R08_minority = R08[R08.readmission_90==1]
R08_majority_downsampled = resample (R08_majority,
                                            replace=False, n_samples =26195, 
                                            random_state=123)
# Combine minority class with downsampled majority class
R08_downsampled = pd.concat([R08_majority_downsampled, R08_minority])
R08_downsampled.readmission_90.value_counts()

# Dropping id variables 
#-------------------------
R08_downsampled.drop(['subject_id','hadm_id'], axis=1, inplace=True)
R08_downsampled.columns
R08_downsampled.dtypes
# Change variable type of age_group and gender to include them in categorical set of data
#------------------------------------------------------------------------------------------
R08_downsampled['age_group'] = R08_downsampled['age_group'].astype(str)
R08_downsampled['gender'] = R08_downsampled['gender'].astype(str)

# Splitting data into X,y
#-----------------------------
features =[]
for column in R08_downsampled.columns:
    if column != 'readmission_90':
        features.append(column)
    X_R08 = R08_downsampled[features]
    y_R08 = R08_downsampled['readmission_90']
X_R08.columns
y_R08.shape
X_R08.shape# ok

# Splitting X into numerical and categorical feature
#--------------------------------------------------------
X_R08_numerical =X_R08[['los', 'los_ED', 'no_ED_admissions']].copy()
X_R08_categorical = X_R08.select_dtypes(['object', 'int16', 'int8'])

# encoding 
#------------------
X_R08_categorical.columns
encoded_X_R08_categorical= pd.get_dummies(data=X_R08_categorical, columns=['admission_location', 'discharge_location','insurance','marital_status',
                                                                   'modularity_class', 'gender','age_group'])

encoded_X_R08_categorical.shape
X_R08_numerical.reset_index(drop=True, inplace=True)
encoded_X_R08_categorical.reset_index(drop=True, inplace=True)
X2_R08= pd.concat([X_R08_numerical, encoded_X_R08_categorical], axis = 1)
X2_R08.shape #(52390, 63)

# X split and standerdizing X_train, X_test 
#------------------------------------------------
X2_R08.dtypes
X_R08_train, X_R08_test, y_R08_train, y_R08_test = train_test_split(X2_R08,y_R08, test_size = 0.3, random_state = 0, stratify=y_R08)
X_R08_train.columns
X_R08_numerical_train = X_R08_train[['los', 'los_ED', 'no_ED_admissions']].copy() 
X_R08_numerical_train.shape
X_R08_categorical_train = X_R08_train.select_dtypes(['uint8','object'])
X_R08_categorical_train.shape
X_R08_numerical_test = X_R08_test[['los', 'los_ED', 'no_ED_admissions']].copy() 
X_R08_numerical_test.shape
X_R08_categorical_test = X_R08_test.select_dtypes(['uint8','object'])
X_R08_categorical_test.shape
for key in X_R08_numerical_train.keys():
    print (key)
for key in X_R08_categorical_train.keys():
    print (key)   
for key in X_R08_numerical_test.keys():
    print (key)  
for key in X_R08_categorical_test.keys():
    print (key)
    
scalar = StandardScaler()
scaled_X_R08_numerical_train= pd.DataFrame(scalar.fit_transform(X_R08_numerical_train), columns=X_R08_numerical_train.keys())
print(scaled_X_R08_numerical_train) # note here that we use fit_transform for X_train part
scaled_X_R08_numerical_test = pd.DataFrame(scalar.transform(X_R08_numerical_test), columns=X_R08_numerical_test.keys()) # note here that we only use the tranform (not fit_transform). That is to use the understanding from train data to test data #VERY IMPORTANT
print(scaled_X_R08_numerical_test)

# Recombining all X parts
#----------------------------------
scaled_X_R08_numerical_train.reset_index(drop=True, inplace=True)
X_R08_categorical_train.reset_index(drop=True, inplace=True)
X_R08_clean_train= pd.concat([scaled_X_R08_numerical_train, X_R08_categorical_train], axis = 1)
X_R08_clean_train.head()
X_R08_clean_train.shape 
scaled_X_R08_numerical_test.reset_index(drop=True, inplace=True)
X_R08_categorical_test.reset_index(drop=True, inplace=True)
X_R08_clean_test = pd.concat([scaled_X_R08_numerical_test, X_R08_categorical_test], axis = 1)
X_R08_clean_test.head()
X_R08_clean_test.shape 

############################
# Implementing GBM on R08
############################
GBM_R08 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
max_depth=1, random_state=0)
GBM_R08.fit(X_R08_clean_train, y_R08_train)
y_pred_GBM_R08 = GBM_R08.predict(X_R08_clean_test)
print("Accuracy of GBM_R08:", metrics.accuracy_score(y_R08_test, y_pred_GBM_R08))
print("Precision:",metrics.precision_score(y_R08_test, y_pred_GBM_R08))
print("Recall:",metrics.recall_score(y_R08_test, y_pred_GBM_R08))
print(classification_report(y_R08_test, y_pred_GBM_R08))

# AUC
#-------------------------------
y_pred_prob_GBM_R08 = GBM_R08.predict_proba(X_R08_clean_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_R08_test, y_pred_prob_GBM_R08)
AUC = metrics.roc_auc_score(y_R08_test, y_pred_prob_GBM_R08)
plt.plot(fpr,tpr,label="GBM_R08 , AUC="+str(AUC))
plt.legend(loc=4)
plt.show()

# Calculating CI 95% for the metrics 
###############################################
y_pred_R08_CI = np.array(y_pred_prob_GBM_R08)
y_R08_true = np.array(y_R08_test)

n_bootstraps = 1000
rng_seed = 42  
rng = np.random.RandomState(rng_seed)

bootstrapped_accuracies_R08 = []
bootstrapped_precisions_R08 = []
bootstrapped_recalls_R08 = []
bootstrapped_f1s_R08 = []
bootstrapped_aucs_R08 = []

for i in range(n_bootstraps):
    # bootstrap by sampling with replacement on the prediction indices
    indices_R08 = rng.randint(0, len(y_pred_R08_CI), len(y_pred_R08_CI))
    if len(np.unique(y_R08_true[indices_R08])) < 2:
        # We need at least one positive and one negative sample for the metrics
        # to be defined: reject the sample
        continue
    
    binary_predictions_R08 = (y_pred_R08_CI[indices_R08] > 0.5).astype(int)
    
    accuracy_R08 = accuracy_score(y_R08_true[indices_R08], binary_predictions_R08)
    precision_R08= precision_score(y_R08_true[indices_R08], binary_predictions_R08)
    recall_R08 = recall_score(y_R08_true[indices_R08], binary_predictions_R08)
    f1_R08 = f1_score(y_R08_true[indices_R08], binary_predictions_R08)
    auc_R08 = roc_auc_score(y_R08_true[indices_R08], y_pred_R08_CI[indices_R08])
    
    bootstrapped_accuracies_R08.append(accuracy_R08)
    bootstrapped_precisions_R08.append(precision_R08)
    bootstrapped_recalls_R08.append(recall_R08)
    bootstrapped_f1s_R08.append(f1_R08)
    bootstrapped_aucs_R08.append(auc_R08)

# Computing the 95% confidence intervals for each metric
accuracy_ci_R08 = (np.percentile(bootstrapped_accuracies_R08, 2.5), np.percentile(bootstrapped_accuracies_R08, 97.5))
precision_ci_R08 = (np.percentile(bootstrapped_precisions_R08, 2.5), np.percentile(bootstrapped_precisions_R08, 97.5))
recall_ci_R08 = (np.percentile(bootstrapped_recalls_R08, 2.5), np.percentile(bootstrapped_recalls_R08, 97.5))
f1_ci_R08 = (np.percentile(bootstrapped_f1s_R08, 2.5), np.percentile(bootstrapped_f1s_R08, 97.5))
auc_ci_R08 = (np.percentile(bootstrapped_aucs_R08, 2.5), np.percentile(bootstrapped_aucs_R08, 97.5))

print(f"Accuracy CI_R08: [{accuracy_ci_R08[0]:.3f} - {accuracy_ci_R08[1]:.3f}]")
print(f"Precision CI_R08: [{precision_ci_R08[0]:.3f} - {precision_ci_R08[1]:.3f}]")
print(f"Recall CI_R08: [{recall_ci_R08[0]:.3f} - {recall_ci_R08[1]:.3f}]")
print(f"F1-Score CI_R08: [{f1_ci_R08[0]:.3f} - {f1_ci_R08[1]:.3f}]")
print(f"AUC CI_R08: [{auc_ci_R08[0]:.3f} - {auc_ci_R08[1]:.3f}]")

# plot the confusion materix (2*2 tabel)
#-----------------------------------------
cm_R08 = metrics.confusion_matrix(y_R08_test, y_pred_GBM_R08)
print('Confusion matrix GBM_R08:', cm_R08)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm_R08), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

##########################
#   4. R06 dataset       #
##########################
#####################################
# Dealing with imbalanced labels
#####################################
# Counting label classes 
#--------------------------
R06.shape
R06.groupby('readmission_90').count()
target_count = R06.readmission_90.value_counts()
target_count.plot(kind='bar', title='Count (readmission_90)')
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1') # (5.82 : 1)
R06.columns

# Downsampling
#------------------------
R06.sort_values(by=['subject_id','hadm_id'])
R06_majority = R06[R06.readmission_90==0]
R06_minority = R06[R06.readmission_90==1]
R06_majority_downsampled = resample (R06_majority,
                                            replace=False, n_samples =34531, 
                                            random_state=123)
# Combine minority class with downsampled majority class
R06_downsampled = pd.concat([R06_majority_downsampled, R06_minority])
R06_downsampled.readmission_90.value_counts()

# Dropping id variables 
#-------------------------
R06_downsampled.drop(['subject_id','hadm_id'], axis=1, inplace=True)
R06_downsampled.columns
R06_downsampled.dtypes
# Change variable type of age_group and gender to include them in categorical set of data
#------------------------------------------------------------------------------------------
R06_downsampled['age_group'] = R06_downsampled['age_group'].astype(str)
R06_downsampled['gender'] = R06_downsampled['gender'].astype(str)

# Splitting data into X,y
#-----------------------------
features =[]
for column in R06_downsampled.columns:
    if column != 'readmission_90':
        features.append(column)
    X_R06 = R06_downsampled[features]
    y_R06 = R06_downsampled['readmission_90']
X_R06.columns
y_R06.shape
X_R06.shape# ok

# Splitting X into numerical and categorical feature
#--------------------------------------------------------
X_R06_numerical =X_R06[['los', 'los_ED', 'no_ED_admissions']].copy()
X_R06_categorical = X_R06.select_dtypes(['object', 'int16', 'int8'])

# encoding 
#------------------
X_R06_categorical.columns
encoded_X_R06_categorical= pd.get_dummies(data=X_R06_categorical, columns=['admission_location', 'discharge_location','insurance','marital_status',
                                                                   'modularity_class', 'gender','age_group'])

encoded_X_R06_categorical.shape
X_R06_numerical.reset_index(drop=True, inplace=True)
encoded_X_R06_categorical.reset_index(drop=True, inplace=True)
X2_R06= pd.concat([X_R06_numerical, encoded_X_R06_categorical], axis = 1)
X2_R06.shape #(69062, 80)

# X split and standerdizing X_train, X_test 
#------------------------------------------------
X2_R06.dtypes
X_R06_train, X_R06_test, y_R06_train, y_R06_test = train_test_split(X2_R06,y_R06, test_size = 0.3, random_state = 0, stratify=y_R06)
X_R06_train.columns
X_R06_numerical_train = X_R06_train[['los', 'los_ED', 'no_ED_admissions']].copy() 
X_R06_numerical_train.shape
X_R06_categorical_train = X_R06_train.select_dtypes(['uint8','object'])
X_R06_categorical_train.shape
X_R06_numerical_test = X_R06_test[['los', 'los_ED', 'no_ED_admissions']].copy() 
X_R06_numerical_test.shape
X_R06_categorical_test = X_R06_test.select_dtypes(['uint8','object'])
X_R06_categorical_test.shape
for key in X_R06_numerical_train.keys():
    print (key)
for key in X_R06_categorical_train.keys():
    print (key)   
for key in X_R06_numerical_test.keys():
    print (key)  
for key in X_R06_categorical_test.keys():
    print (key)
    
scalar = StandardScaler()
scaled_X_R06_numerical_train= pd.DataFrame(scalar.fit_transform(X_R06_numerical_train), columns=X_R06_numerical_train.keys())
print(scaled_X_R06_numerical_train) # note here that we use fit_transform for X_train part
scaled_X_R06_numerical_test = pd.DataFrame(scalar.transform(X_R06_numerical_test), columns=X_R06_numerical_test.keys()) # note here that we only use the tranform (not fit_transform). That is to use the understanding from train data to test data #VERY IMPORTANT
print(scaled_X_R06_numerical_test)

# Recombining all X parts
#----------------------------------
scaled_X_R06_numerical_train.reset_index(drop=True, inplace=True)
X_R06_categorical_train.reset_index(drop=True, inplace=True)
X_R06_clean_train= pd.concat([scaled_X_R06_numerical_train, X_R06_categorical_train], axis = 1)
X_R06_clean_train.head()
X_R06_clean_train.shape 
scaled_X_R06_numerical_test.reset_index(drop=True, inplace=True)
X_R06_categorical_test.reset_index(drop=True, inplace=True)
X_R06_clean_test = pd.concat([scaled_X_R06_numerical_test, X_R06_categorical_test], axis = 1)
X_R06_clean_test.head()
X_R06_clean_test.shape 

###########################
# Implementing GBM on R06
###########################
GBM_R06 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
max_depth=1, random_state=0)
GBM_R06.fit(X_R06_clean_train, y_R06_train)
y_pred_GBM_R06 = GBM_R06.predict(X_R06_clean_test)
print("Accuracy of GBM_R06:", metrics.accuracy_score(y_R06_test, y_pred_GBM_R06))
print("Precision:",metrics.precision_score(y_R06_test, y_pred_GBM_R06))
print("Recall:",metrics.recall_score(y_R06_test, y_pred_GBM_R06))
print(classification_report(y_R06_test, y_pred_GBM_R06))

# AUC
#-------------------------------
y_pred_prob_GBM_R06 = GBM_R06.predict_proba(X_R06_clean_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_R06_test, y_pred_prob_GBM_R06)
AUC = metrics.roc_auc_score(y_R06_test, y_pred_prob_GBM_R06)
plt.plot(fpr,tpr,label="GBM_R06 , AUC="+str(AUC))
plt.legend(loc=4)
plt.show()

# Calculating CI 95% for the metrics 
###############################################
y_pred_R06_CI = np.array(y_pred_prob_GBM_R06)
y_R06_true = np.array(y_R06_test)

n_bootstraps = 1000
rng_seed = 42  
rng = np.random.RandomState(rng_seed)

bootstrapped_accuracies_R06 = []
bootstrapped_precisions_R06 = []
bootstrapped_recalls_R06 = []
bootstrapped_f1s_R06 = []
bootstrapped_aucs_R06 = []

for i in range(n_bootstraps):
    # bootstrap by sampling with replacement on the prediction indices
    indices_R06 = rng.randint(0, len(y_pred_R06_CI), len(y_pred_R06_CI))
    if len(np.unique(y_R06_true[indices_R06])) < 2:
        # We need at least one positive and one negative sample for the metrics
        # to be defined: reject the sample
        continue
    
    binary_predictions_R06 = (y_pred_R06_CI[indices_R06] > 0.5).astype(int)
    
    accuracy_R06 = accuracy_score(y_R06_true[indices_R06], binary_predictions_R06)
    precision_R06= precision_score(y_R06_true[indices_R06], binary_predictions_R06)
    recall_R06 = recall_score(y_R06_true[indices_R06], binary_predictions_R06)
    f1_R06 = f1_score(y_R06_true[indices_R06], binary_predictions_R06)
    auc_R06 = roc_auc_score(y_R06_true[indices_R06], y_pred_R06_CI[indices_R06])
    
    bootstrapped_accuracies_R06.append(accuracy_R06)
    bootstrapped_precisions_R06.append(precision_R06)
    bootstrapped_recalls_R06.append(recall_R06)
    bootstrapped_f1s_R06.append(f1_R06)
    bootstrapped_aucs_R06.append(auc_R06)

# Computing the 95% confidence intervals for each metric
accuracy_ci_R06 = (np.percentile(bootstrapped_accuracies_R06, 2.5), np.percentile(bootstrapped_accuracies_R06, 97.5))
precision_ci_R06 = (np.percentile(bootstrapped_precisions_R06, 2.5), np.percentile(bootstrapped_precisions_R06, 97.5))
recall_ci_R06 = (np.percentile(bootstrapped_recalls_R06, 2.5), np.percentile(bootstrapped_recalls_R06, 97.5))
f1_ci_R06 = (np.percentile(bootstrapped_f1s_R06, 2.5), np.percentile(bootstrapped_f1s_R06, 97.5))
auc_ci_R06 = (np.percentile(bootstrapped_aucs_R06, 2.5), np.percentile(bootstrapped_aucs_R06, 97.5))

print(f"Accuracy CI_R06: [{accuracy_ci_R06[0]:.3f} - {accuracy_ci_R06[1]:.3f}]")
print(f"Precision CI_R06: [{precision_ci_R06[0]:.3f} - {precision_ci_R06[1]:.3f}]")
print(f"Recall CI_R06: [{recall_ci_R06[0]:.3f} - {recall_ci_R06[1]:.3f}]")
print(f"F1-Score CI_R06: [{f1_ci_R06[0]:.3f} - {f1_ci_R06[1]:.3f}]")
print(f"AUC CI_R06: [{auc_ci_R06[0]:.3f} - {auc_ci_R06[1]:.3f}]")

# plot the confusion materix (2*2 tabel)
#-----------------------------------------
cm_R06 = metrics.confusion_matrix(y_R06_test, y_pred_GBM_R06)
print('Confusion matrix GBM_R06:', cm_R06)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm_R06), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

##########################
#   5. R05 dataset       #
##########################
#####################################
# Dealing with imbalanced labels
#####################################
# Counting label classes 
#--------------------------
R05.shape
R05.groupby('readmission_90').count()
target_count = R05.readmission_90.value_counts()
target_count.plot(kind='bar', title='Count (readmission_90)')
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1') # (5.81 : 1)
R05.columns

# Downsampling
#------------------------
R05.sort_values(by=['subject_id','hadm_id'])
R05_majority = R05[R05.readmission_90==0]
R05_minority = R05[R05.readmission_90==1]
R05_majority_downsampled = resample (R05_majority,
                                            replace=False, n_samples =39247, 
                                            random_state=123)
# Combine minority class with downsampled majority class
R05_downsampled = pd.concat([R05_majority_downsampled, R05_minority])
R05_downsampled.readmission_90.value_counts()

# Dropping id variables 
#-------------------------
R05_downsampled.drop(['subject_id','hadm_id'], axis=1, inplace=True)
R05_downsampled.columns
R05_downsampled.dtypes
# Change variable type of age_group and gender to include them in categorical set of data
#------------------------------------------------------------------------------------------
R05_downsampled['age_group'] = R05_downsampled['age_group'].astype(str)
R05_downsampled['gender'] = R05_downsampled['gender'].astype(str)

# Splitting data into X,y
#-----------------------------
features =[]
for column in R05_downsampled.columns:
    if column != 'readmission_90':
        features.append(column)
    X_R05 = R05_downsampled[features]
    y_R05 = R05_downsampled['readmission_90']
X_R05.columns
y_R05.shape
X_R05.shape# ok

# Splitting X into numerical and categorical feature
#--------------------------------------------------------
X_R05_numerical =X_R05[['los', 'los_ED', 'no_ED_admissions']].copy()
X_R05_categorical = X_R05.select_dtypes(['object', 'int16', 'int8'])

# encoding 
#------------------
X_R05_categorical.columns
encoded_X_R05_categorical= pd.get_dummies(data=X_R05_categorical, columns=['admission_location', 'discharge_location','insurance','marital_status',
                                                                   'modularity_class', 'gender','age_group'])

encoded_X_R05_categorical.shape
X_R05_numerical.reset_index(drop=True, inplace=True)
encoded_X_R05_categorical.reset_index(drop=True, inplace=True)
X2_R05= pd.concat([X_R05_numerical, encoded_X_R05_categorical], axis = 1)
X2_R05.shape #(78494, 95)

# X split and standerdizing X_train, X_test 
#------------------------------------------------
X2_R05.dtypes
X_R05_train, X_R05_test, y_R05_train, y_R05_test = train_test_split(X2_R05,y_R05, test_size = 0.3, random_state = 0, stratify=y_R05)
X_R05_train.columns
X_R05_numerical_train = X_R05_train[['los', 'los_ED', 'no_ED_admissions']].copy() 
X_R05_numerical_train.shape
X_R05_categorical_train = X_R05_train.select_dtypes(['uint8','object'])
X_R05_categorical_train.shape
X_R05_numerical_test = X_R05_test[['los', 'los_ED', 'no_ED_admissions']].copy() 
X_R05_numerical_test.shape
X_R05_categorical_test = X_R05_test.select_dtypes(['uint8','object'])
X_R05_categorical_test.shape
for key in X_R05_numerical_train.keys():
    print (key)
for key in X_R05_categorical_train.keys():
    print (key)   
for key in X_R05_numerical_test.keys():
    print (key)  
for key in X_R05_categorical_test.keys():
    print (key)
    
scalar = StandardScaler()
scaled_X_R05_numerical_train= pd.DataFrame(scalar.fit_transform(X_R05_numerical_train), columns=X_R05_numerical_train.keys())
print(scaled_X_R05_numerical_train) # note here that we use fit_transform for X_train part
scaled_X_R05_numerical_test = pd.DataFrame(scalar.transform(X_R05_numerical_test), columns=X_R05_numerical_test.keys()) # note here that we only use the tranform (not fit_transform). That is to use the understanding from train data to test data #VERY IMPORTANT
print(scaled_X_R05_numerical_test)

# Recombining all X parts
#----------------------------------
scaled_X_R05_numerical_train.reset_index(drop=True, inplace=True)
X_R05_categorical_train.reset_index(drop=True, inplace=True)
X_R05_clean_train= pd.concat([scaled_X_R05_numerical_train, X_R05_categorical_train], axis = 1)
X_R05_clean_train.head()
X_R05_clean_train.shape 
scaled_X_R05_numerical_test.reset_index(drop=True, inplace=True)
X_R05_categorical_test.reset_index(drop=True, inplace=True)
X_R05_clean_test = pd.concat([scaled_X_R05_numerical_test, X_R05_categorical_test], axis = 1)
X_R05_clean_test.head()
X_R05_clean_test.shape 

###########################
# Implementing GBM on R05
###########################
GBM_R05 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
max_depth=1, random_state=0)
GBM_R05.fit(X_R05_clean_train, y_R05_train)
y_pred_GBM_R05 = GBM_R05.predict(X_R05_clean_test)
print("Accuracy of GBM_R05:", metrics.accuracy_score(y_R05_test, y_pred_GBM_R05))
print("Precision:",metrics.precision_score(y_R05_test, y_pred_GBM_R05))
print("Recall:",metrics.recall_score(y_R05_test, y_pred_GBM_R05))
print(classification_report(y_R05_test, y_pred_GBM_R05))

# AUC
#-------------------------------
y_pred_prob_GBM_R05 = GBM_R05.predict_proba(X_R05_clean_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_R05_test, y_pred_prob_GBM_R05)
AUC = metrics.roc_auc_score(y_R05_test, y_pred_prob_GBM_R05)
plt.plot(fpr,tpr,label="GBM_R05 , AUC="+str(AUC))
plt.legend(loc=4)
plt.show()


# Calculating CI 95% for the metrics 
###############################################
y_pred_R05_CI = np.array(y_pred_prob_GBM_R05)
y_R05_true = np.array(y_R05_test)

n_bootstraps = 1000
rng_seed = 42  
rng = np.random.RandomState(rng_seed)

bootstrapped_accuracies_R05 = []
bootstrapped_precisions_R05 = []
bootstrapped_recalls_R05 = []
bootstrapped_f1s_R05 = []
bootstrapped_aucs_R05 = []

for i in range(n_bootstraps):
    # bootstrap by sampling with replacement on the prediction indices
    indices_R05 = rng.randint(0, len(y_pred_R05_CI), len(y_pred_R05_CI))
    if len(np.unique(y_R05_true[indices_R05])) < 2:
        # We need at least one positive and one negative sample for the metrics
        # to be defined: reject the sample
        continue
    
    binary_predictions_R05 = (y_pred_R05_CI[indices_R05] > 0.5).astype(int)
    
    accuracy_R05 = accuracy_score(y_R05_true[indices_R05], binary_predictions_R05)
    precision_R05= precision_score(y_R05_true[indices_R05], binary_predictions_R05)
    recall_R05 = recall_score(y_R05_true[indices_R05], binary_predictions_R05)
    f1_R05 = f1_score(y_R05_true[indices_R05], binary_predictions_R05)
    auc_R05 = roc_auc_score(y_R05_true[indices_R05], y_pred_R05_CI[indices_R05])
    
    bootstrapped_accuracies_R05.append(accuracy_R05)
    bootstrapped_precisions_R05.append(precision_R05)
    bootstrapped_recalls_R05.append(recall_R05)
    bootstrapped_f1s_R05.append(f1_R05)
    bootstrapped_aucs_R05.append(auc_R05)

# Computing the 95% confidence intervals for each metric
accuracy_ci_R05 = (np.percentile(bootstrapped_accuracies_R05, 2.5), np.percentile(bootstrapped_accuracies_R05, 97.5))
precision_ci_R05 = (np.percentile(bootstrapped_precisions_R05, 2.5), np.percentile(bootstrapped_precisions_R05, 97.5))
recall_ci_R05 = (np.percentile(bootstrapped_recalls_R05, 2.5), np.percentile(bootstrapped_recalls_R05, 97.5))
f1_ci_R05 = (np.percentile(bootstrapped_f1s_R05, 2.5), np.percentile(bootstrapped_f1s_R05, 97.5))
auc_ci_R05 = (np.percentile(bootstrapped_aucs_R05, 2.5), np.percentile(bootstrapped_aucs_R05, 97.5))

print(f"Accuracy CI_R05: [{accuracy_ci_R05[0]:.3f} - {accuracy_ci_R05[1]:.3f}]")
print(f"Precision CI_R05: [{precision_ci_R05[0]:.3f} - {precision_ci_R05[1]:.3f}]")
print(f"Recall CI_R05: [{recall_ci_R05[0]:.3f} - {recall_ci_R05[1]:.3f}]")
print(f"F1-Score CI_R05: [{f1_ci_R05[0]:.3f} - {f1_ci_R05[1]:.3f}]")
print(f"AUC CI_R05: [{auc_ci_R05[0]:.3f} - {auc_ci_R05[1]:.3f}]")

# plot the confusion materix (2*2 tabel)
#-----------------------------------------
cm_R05 = metrics.confusion_matrix(y_R05_test, y_pred_GBM_R05)
print('Confusion matrix GBM_R05:', cm_R05)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm_R05), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

##########################
#   6. R01 dataset       #
##########################
#####################################
# Dealing with imbalanced labels
#####################################
# Counting label classes 
#--------------------------
R01.shape
R01.groupby('readmission_90').count()
target_count = R01.readmission_90.value_counts()
target_count.plot(kind='bar', title='Count (readmission_90)')
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1') #(5.62 : 1)
R01.columns

# Downsampling
#------------------------
R01.sort_values(by=['subject_id','hadm_id'])
R01_majority = R01[R01.readmission_90==0]
R01_minority = R01[R01.readmission_90==1]
R01_majority_downsampled = resample (R01_majority,
                                            replace=False, n_samples =56879, 
                                            random_state=123)
# Combine minority class with downsampled majority class
R01_downsampled = pd.concat([R01_majority_downsampled, R01_minority])
R01_downsampled.readmission_90.value_counts()

# Dropping id variables 
#-------------------------
R01_downsampled.drop(['subject_id','hadm_id'], axis=1, inplace=True)
R01_downsampled.columns
R01_downsampled.dtypes
# Change variable type of age_group and gender to include them in categorical set of data
#------------------------------------------------------------------------------------------
R01_downsampled['age_group'] = R01_downsampled['age_group'].astype(str)
R01_downsampled['gender'] = R01_downsampled['gender'].astype(str)

# Splitting data into X,y
#-----------------------------
features =[]
for column in R01_downsampled.columns:
    if column != 'readmission_90':
        features.append(column)
    X_R01 = R01_downsampled[features]
    y_R01 = R01_downsampled['readmission_90']
X_R01.columns
y_R01.shape
X_R01.shape# ok

# Splitting X into numerical and categorical feature
#--------------------------------------------------------
X_R01_numerical =X_R01[['los', 'los_ED', 'no_ED_admissions']].copy()
X_R01_categorical = X_R01.select_dtypes(['object', 'int16', 'int8'])

# encoding 
#------------------
X_R01_categorical.columns
encoded_X_R01_categorical= pd.get_dummies(data=X_R01_categorical, columns=['admission_location', 'discharge_location','insurance','marital_status',
                                                                   'modularity_class', 'gender','age_group'])

encoded_X_R01_categorical.shape
X_R01_numerical.reset_index(drop=True, inplace=True)
encoded_X_R01_categorical.reset_index(drop=True, inplace=True)
X2_R01= pd.concat([X_R01_numerical, encoded_X_R01_categorical], axis = 1)
X2_R01.shape #(113758, 362)

# X split and standerdizing X_train, X_test 
#------------------------------------------------
X2_R01.dtypes
X_R01_train, X_R01_test, y_R01_train, y_R01_test = train_test_split(X2_R01,y_R01, test_size = 0.3, random_state = 0, stratify=y_R01)
X_R01_train.columns
X_R01_numerical_train = X_R01_train[['los', 'los_ED', 'no_ED_admissions']].copy() 
X_R01_numerical_train.shape
X_R01_categorical_train = X_R01_train.select_dtypes(['uint8','object'])
X_R01_categorical_train.shape
X_R01_numerical_test = X_R01_test[['los', 'los_ED', 'no_ED_admissions']].copy() 
X_R01_numerical_test.shape
X_R01_categorical_test = X_R01_test.select_dtypes(['uint8','object'])
X_R01_categorical_test.shape
for key in X_R01_numerical_train.keys():
    print (key)
for key in X_R01_categorical_train.keys():
    print (key)   
for key in X_R01_numerical_test.keys():
    print (key)  
for key in X_R01_categorical_test.keys():
    print (key)
    
scalar = StandardScaler()
scaled_X_R01_numerical_train= pd.DataFrame(scalar.fit_transform(X_R01_numerical_train), columns=X_R01_numerical_train.keys())
print(scaled_X_R01_numerical_train) # note here that we use fit_transform for X_train part
scaled_X_R01_numerical_test = pd.DataFrame(scalar.transform(X_R01_numerical_test), columns=X_R01_numerical_test.keys()) # note here that we only use the tranform (not fit_transform). That is to use the understanding from train data to test data #VERY IMPORTANT
print(scaled_X_R01_numerical_test)

# Recombining all X parts
#----------------------------------
scaled_X_R01_numerical_train.reset_index(drop=True, inplace=True)
X_R01_categorical_train.reset_index(drop=True, inplace=True)
X_R01_clean_train= pd.concat([scaled_X_R01_numerical_train, X_R01_categorical_train], axis = 1)
X_R01_clean_train.head()
X_R01_clean_train.shape 
scaled_X_R01_numerical_test.reset_index(drop=True, inplace=True)
X_R01_categorical_test.reset_index(drop=True, inplace=True)
X_R01_clean_test = pd.concat([scaled_X_R01_numerical_test, X_R01_categorical_test], axis = 1)
X_R01_clean_test.head()
X_R01_clean_test.shape 

###########################
# Implementing GBM on R01
###########################
GBM_R01 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
max_depth=1, random_state=0)
GBM_R01.fit(X_R01_clean_train, y_R01_train)
y_pred_GBM_R01 = GBM_R01.predict(X_R01_clean_test)
print("Accuracy of GBM_R01:", metrics.accuracy_score(y_R01_test, y_pred_GBM_R01))
print("Precision:",metrics.precision_score(y_R01_test, y_pred_GBM_R01))
print("Recall:",metrics.recall_score(y_R01_test, y_pred_GBM_R01))
print(classification_report(y_R01_test, y_pred_GBM_R01))

# AUC
#-------------------------------
y_pred_prob_GBM_R01 = GBM_R01.predict_proba(X_R01_clean_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_R01_test, y_pred_prob_GBM_R01)
AUC = metrics.roc_auc_score(y_R01_test, y_pred_prob_GBM_R01)
plt.plot(fpr,tpr,label="GBM_R01 , AUC="+str(AUC))
plt.legend(loc=4)
plt.show()

# Calculating CI 95% for the metrics 
###############################################
y_pred_R01_CI = np.array(y_pred_prob_GBM_R01)
y_R01_true = np.array(y_R01_test)

n_bootstraps = 1000
rng_seed = 42  
rng = np.random.RandomState(rng_seed)

bootstrapped_accuracies_R01 = []
bootstrapped_precisions_R01 = []
bootstrapped_recalls_R01 = []
bootstrapped_f1s_R01 = []
bootstrapped_aucs_R01 = []

for i in range(n_bootstraps):
    # bootstrap by sampling with replacement on the prediction indices
    indices_R01 = rng.randint(0, len(y_pred_R01_CI), len(y_pred_R01_CI))
    if len(np.unique(y_R01_true[indices_R01])) < 2:
        # We need at least one positive and one negative sample for the metrics
        # to be defined: reject the sample
        continue
    
    binary_predictions_R01 = (y_pred_R01_CI[indices_R01] > 0.5).astype(int)
    
    accuracy_R01 = accuracy_score(y_R01_true[indices_R01], binary_predictions_R01)
    precision_R01= precision_score(y_R01_true[indices_R01], binary_predictions_R01)
    recall_R01 = recall_score(y_R01_true[indices_R01], binary_predictions_R01)
    f1_R01 = f1_score(y_R01_true[indices_R01], binary_predictions_R01)
    auc_R01 = roc_auc_score(y_R01_true[indices_R01], y_pred_R01_CI[indices_R01])
    
    bootstrapped_accuracies_R01.append(accuracy_R01)
    bootstrapped_precisions_R01.append(precision_R01)
    bootstrapped_recalls_R01.append(recall_R01)
    bootstrapped_f1s_R01.append(f1_R01)
    bootstrapped_aucs_R01.append(auc_R01)

# Computing the 95% confidence intervals for each metric
accuracy_ci_R01 = (np.percentile(bootstrapped_accuracies_R01, 2.5), np.percentile(bootstrapped_accuracies_R01, 97.5))
precision_ci_R01 = (np.percentile(bootstrapped_precisions_R01, 2.5), np.percentile(bootstrapped_precisions_R01, 97.5))
recall_ci_R01 = (np.percentile(bootstrapped_recalls_R01, 2.5), np.percentile(bootstrapped_recalls_R01, 97.5))
f1_ci_R01 = (np.percentile(bootstrapped_f1s_R01, 2.5), np.percentile(bootstrapped_f1s_R01, 97.5))
auc_ci_R01 = (np.percentile(bootstrapped_aucs_R01, 2.5), np.percentile(bootstrapped_aucs_R01, 97.5))

print(f"Accuracy CI_R01: [{accuracy_ci_R01[0]:.3f} - {accuracy_ci_R01[1]:.3f}]")
print(f"Precision CI_R01: [{precision_ci_R01[0]:.3f} - {precision_ci_R01[1]:.3f}]")
print(f"Recall CI_R01: [{recall_ci_R01[0]:.3f} - {recall_ci_R01[1]:.3f}]")
print(f"F1-Score CI_R01: [{f1_ci_R01[0]:.3f} - {f1_ci_R01[1]:.3f}]")
print(f"AUC CI_R01: [{auc_ci_R01[0]:.3f} - {auc_ci_R01[1]:.3f}]")

# plot the confusion materix (2*2 tabel)
#-----------------------------------------
cm_R01 = metrics.confusion_matrix(y_R01_test, y_pred_GBM_R01)
print('Confusion matrix GBM_R01:', cm_R01)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm_R01), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

###########################
#   7. R001 dataset       #
###########################
#####################################
# Dealing with imbalanced labels
#####################################
# Counting label classes 
#--------------------------
R001.shape
R001.groupby('readmission_90').count()
target_count = R001.readmission_90.value_counts()
target_count.plot(kind='bar', title='Count (readmission_90)')
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1') #(5.67 : 1)
R001.columns

# Downsampling
#------------------------
R001.sort_values(by=['subject_id','hadm_id'])
R001_majority = R001[R001.readmission_90==0]
R001_minority = R001[R001.readmission_90==1]
R001_majority_downsampled = resample (R001_majority,
                                            replace=False, n_samples =63424, 
                                            random_state=123)
# Combine minority class with downsampled majority class
R001_downsampled = pd.concat([R001_majority_downsampled, R001_minority])
R001_downsampled.readmission_90.value_counts()

# Dropping id variables 
#-------------------------
R001_downsampled.drop(['subject_id','hadm_id'], axis=1, inplace=True)
R001_downsampled.columns
R001_downsampled.dtypes
# Change variable type of age_group and gender to include them in categorical set of data
#------------------------------------------------------------------------------------------
R001_downsampled['age_group'] = R001_downsampled['age_group'].astype(str)
R001_downsampled['gender'] = R001_downsampled['gender'].astype(str)

# Splitting data into X,y
#-----------------------------
features =[]
for column in R001_downsampled.columns:
    if column != 'readmission_90':
        features.append(column)
    X_R001 = R001_downsampled[features]
    y_R001 = R001_downsampled['readmission_90']
X_R001.columns
y_R001.shape
X_R001.shape# ok

# Splitting X into numerical and categorical feature
#--------------------------------------------------------
X_R001_numerical =X_R001[['los', 'los_ED', 'no_ED_admissions']].copy()
X_R001_categorical = X_R001.select_dtypes(['object', 'int16', 'int8'])

# encoding 
#------------------
X_R001_categorical.columns
encoded_X_R001_categorical= pd.get_dummies(data=X_R001_categorical, columns=['admission_location', 'discharge_location','insurance','marital_status',
                                                                   'modularity_class', 'gender','age_group'])

encoded_X_R001_categorical.shape
X_R001_numerical.reset_index(drop=True, inplace=True)
encoded_X_R001_categorical.reset_index(drop=True, inplace=True)
X2_R001= pd.concat([X_R001_numerical, encoded_X_R001_categorical], axis = 1)
X2_R001.shape #(131850, 4403)

# X split and standerdizing X_train, X_test 
#------------------------------------------------
X2_R001.dtypes
X_R001_train, X_R001_test, y_R001_train, y_R001_test = train_test_split(X2_R001,y_R001, test_size = 0.3, random_state = 0, stratify=y_R001)
X_R001_train.columns
X_R001_numerical_train = X_R001_train[['los', 'los_ED', 'no_ED_admissions']].copy() 
X_R001_numerical_train.shape
X_R001_categorical_train = X_R001_train.select_dtypes(['uint8','object'])
X_R001_categorical_train.shape
X_R001_numerical_test = X_R001_test[['los', 'los_ED', 'no_ED_admissions']].copy() 
X_R001_numerical_test.shape
X_R001_categorical_test = X_R001_test.select_dtypes(['uint8','object'])
X_R001_categorical_test.shape
for key in X_R001_numerical_train.keys():
    print (key)
for key in X_R001_categorical_train.keys():
    print (key)   
for key in X_R001_numerical_test.keys():
    print (key)  
for key in X_R001_categorical_test.keys():
    print (key)
    
scalar = StandardScaler()
scaled_X_R001_numerical_train= pd.DataFrame(scalar.fit_transform(X_R001_numerical_train), columns=X_R001_numerical_train.keys())
print(scaled_X_R001_numerical_train) # note here that we use fit_transform for X_train part
scaled_X_R001_numerical_test = pd.DataFrame(scalar.transform(X_R001_numerical_test), columns=X_R001_numerical_test.keys()) # note here that we only use the tranform (not fit_transform). That is to use the understanding from train data to test data #VERY IMPORTANT
print(scaled_X_R001_numerical_test)

# Recombining all X parts
#----------------------------------
scaled_X_R001_numerical_train.reset_index(drop=True, inplace=True)
X_R001_categorical_train.reset_index(drop=True, inplace=True)
X_R001_clean_train= pd.concat([scaled_X_R001_numerical_train, X_R001_categorical_train], axis = 1)
X_R001_clean_train.head()
X_R001_clean_train.shape 
scaled_X_R001_numerical_test.reset_index(drop=True, inplace=True)
X_R001_categorical_test.reset_index(drop=True, inplace=True)
X_R001_clean_test = pd.concat([scaled_X_R001_numerical_test, X_R001_categorical_test], axis = 1)
X_R001_clean_test.head()
X_R001_clean_test.shape 

############################
# Implementing GBM on R001
############################
GBM_R001 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
max_depth=1, random_state=0)
GBM_R001.fit(X_R001_clean_train, y_R001_train)
y_pred_GBM_R001 = GBM_R001.predict(X_R001_clean_test)
print("Accuracy of GBM_R001:", metrics.accuracy_score(y_R001_test, y_pred_GBM_R001))
print("Precision:",metrics.precision_score(y_R001_test, y_pred_GBM_R001))
print("Recall:",metrics.recall_score(y_R001_test, y_pred_GBM_R001))
print(classification_report(y_R001_test, y_pred_GBM_R001))

# AUC
#-------------------------------
y_pred_prob_GBM_R001 = GBM_R001.predict_proba(X_R001_clean_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_R001_test, y_pred_prob_GBM_R001)
AUC = metrics.roc_auc_score(y_R001_test, y_pred_prob_GBM_R001)
plt.plot(fpr,tpr,label="GBM_R001 , AUC="+str(AUC))
plt.legend(loc=4)
plt.show()


# Calculating CI 95% for the metrics 
###############################################
y_pred_R001_CI = np.array(y_pred_prob_GBM_R001)
y_R001_true = np.array(y_R001_test)

n_bootstraps = 1000
rng_seed = 42  
rng = np.random.RandomState(rng_seed)

bootstrapped_accuracies_R001 = []
bootstrapped_precisions_R001 = []
bootstrapped_recalls_R001 = []
bootstrapped_f1s_R001 = []
bootstrapped_aucs_R001 = []

for i in range(n_bootstraps):
    # bootstrap by sampling with replacement on the prediction indices
    indices_R001 = rng.randint(0, len(y_pred_R001_CI), len(y_pred_R001_CI))
    if len(np.unique(y_R001_true[indices_R001])) < 2:
        # We need at least one positive and one negative sample for the metrics
        # to be defined: reject the sample
        continue
    
    binary_predictions_R001 = (y_pred_R001_CI[indices_R001] > 0.5).astype(int)
    
    accuracy_R001 = accuracy_score(y_R001_true[indices_R001], binary_predictions_R001)
    precision_R001= precision_score(y_R001_true[indices_R001], binary_predictions_R001)
    recall_R001 = recall_score(y_R001_true[indices_R001], binary_predictions_R001)
    f1_R001 = f1_score(y_R001_true[indices_R001], binary_predictions_R001)
    auc_R001 = roc_auc_score(y_R001_true[indices_R001], y_pred_R001_CI[indices_R001])
    
    bootstrapped_accuracies_R001.append(accuracy_R001)
    bootstrapped_precisions_R001.append(precision_R001)
    bootstrapped_recalls_R001.append(recall_R001)
    bootstrapped_f1s_R001.append(f1_R001)
    bootstrapped_aucs_R001.append(auc_R001)

# Computing the 95% confidence intervals for each metric
accuracy_ci_R001 = (np.percentile(bootstrapped_accuracies_R001, 2.5), np.percentile(bootstrapped_accuracies_R001, 97.5))
precision_ci_R001 = (np.percentile(bootstrapped_precisions_R001, 2.5), np.percentile(bootstrapped_precisions_R001, 97.5))
recall_ci_R001 = (np.percentile(bootstrapped_recalls_R001, 2.5), np.percentile(bootstrapped_recalls_R001, 97.5))
f1_ci_R001 = (np.percentile(bootstrapped_f1s_R001, 2.5), np.percentile(bootstrapped_f1s_R001, 97.5))
auc_ci_R001 = (np.percentile(bootstrapped_aucs_R001, 2.5), np.percentile(bootstrapped_aucs_R001, 97.5))

print(f"Accuracy CI_R001: [{accuracy_ci_R001[0]:.3f} - {accuracy_ci_R001[1]:.3f}]")
print(f"Precision CI_R001: [{precision_ci_R001[0]:.3f} - {precision_ci_R001[1]:.3f}]")
print(f"Recall CI_R001: [{recall_ci_R001[0]:.3f} - {recall_ci_R001[1]:.3f}]")
print(f"F1-Score CI_R001: [{f1_ci_R001[0]:.3f} - {f1_ci_R001[1]:.3f}]")
print(f"AUC CI_R001: [{auc_ci_R001[0]:.3f} - {auc_ci_R001[1]:.3f}]")

# plot the confusion materix (2*2 tabel)
#-----------------------------------------
cm_R001 = metrics.confusion_matrix(y_R001_test, y_pred_GBM_R001)
print('Confusion matrix GBM_R001:', cm_R001)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm_R001), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

##########################
#   8. CCS dataset       #
##########################
#####################################
# Dealing with imbalanced labels
#####################################
# Counting label classes 
#--------------------------
CCS.shape
CCS.groupby('readmission_90').count()
target_count = CCS.readmission_90.value_counts()
target_count.plot(kind='bar', title='Count (readmission_90)')
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1') #(5.77 : 1)
CCS.columns

# Downsampling
#------------------------
CCS.sort_values(by=['subject_id','hadm_id'])
CCS_majority = CCS[CCS.readmission_90==0]
CCS_minority = CCS[CCS.readmission_90==1]
CCS_majority_downsampled = resample (CCS_majority,
                                            replace=False, n_samples =56654, 
                                            random_state=123)
# Combine minority class with downsampled majority class
CCS_downsampled = pd.concat([CCS_majority_downsampled, CCS_minority])
CCS_downsampled.readmission_90.value_counts()

# Dropping id variables 
#-------------------------
CCS_downsampled.drop(['subject_id','hadm_id'], axis=1, inplace=True)
CCS_downsampled.columns
CCS_downsampled.dtypes
# Change variable type of age_group and gender to include them in categorical set of data
#------------------------------------------------------------------------------------------
CCS_downsampled['age_group'] = CCS_downsampled['age_group'].astype(str)
CCS_downsampled['gender'] = CCS_downsampled['gender'].astype(str)

# Splitting data into X,y
#-----------------------------
features =[]
for column in CCS_downsampled.columns:
    if column != 'readmission_90':
        features.append(column)
    X_CCS = CCS_downsampled[features]
    y_CCS = CCS_downsampled['readmission_90']
X_CCS.columns
y_CCS.shape
X_CCS.shape# ok

# Splitting X into numerical and categorical feature
#--------------------------------------------------------
X_CCS_numerical =X_CCS[['los', 'los_ED', 'no_ED_admissions']].copy()
X_CCS_categorical = X_CCS.select_dtypes(['object', 'int16', 'int8'])

# encoding 
#------------------
X_CCS_categorical.columns
encoded_X_CCS_categorical= pd.get_dummies(data=X_CCS_categorical, columns=['admission_location', 'discharge_location','insurance','marital_status',
                                                                   'ccscategory', 'gender','age_group'])

encoded_X_CCS_categorical.shape
X_CCS_numerical.reset_index(drop=True, inplace=True)
encoded_X_CCS_categorical.reset_index(drop=True, inplace=True)
X2_CCS= pd.concat([X_CCS_numerical, encoded_X_CCS_categorical], axis = 1)
X2_CCS.shape #(113308, 322)

# X split and standerdizing X_train, X_test 
#------------------------------------------------
X2_CCS.dtypes
X_CCS_train, X_CCS_test, y_CCS_train, y_CCS_test = train_test_split(X2_CCS,y_CCS, test_size = 0.3, random_state = 0, stratify=y_CCS)
X_CCS_train.columns
X_CCS_numerical_train = X_CCS_train[['los', 'los_ED', 'no_ED_admissions']].copy() 
X_CCS_numerical_train.shape
X_CCS_categorical_train = X_CCS_train.select_dtypes(['uint8','object'])
X_CCS_categorical_train.shape
X_CCS_numerical_test = X_CCS_test[['los', 'los_ED', 'no_ED_admissions']].copy() 
X_CCS_numerical_test.shape
X_CCS_categorical_test = X_CCS_test.select_dtypes(['uint8','object'])
X_CCS_categorical_test.shape
for key in X_CCS_numerical_train.keys():
    print (key)
for key in X_CCS_categorical_train.keys():
    print (key)   
for key in X_CCS_numerical_test.keys():
    print (key)  
for key in X_CCS_categorical_test.keys():
    print (key)
    
scalar = StandardScaler()
scaled_X_CCS_numerical_train= pd.DataFrame(scalar.fit_transform(X_CCS_numerical_train), columns=X_CCS_numerical_train.keys())
print(scaled_X_CCS_numerical_train) # note here that we use fit_transform for X_train part
scaled_X_CCS_numerical_test = pd.DataFrame(scalar.transform(X_CCS_numerical_test), columns=X_CCS_numerical_test.keys()) # note here that we only use the tranform (not fit_transform). That is to use the understanding from train data to test data #VERY IMPORTANT
print(scaled_X_CCS_numerical_test)

# Recombining all X parts
#----------------------------------
scaled_X_CCS_numerical_train.reset_index(drop=True, inplace=True)
X_CCS_categorical_train.reset_index(drop=True, inplace=True)
X_CCS_clean_train= pd.concat([scaled_X_CCS_numerical_train, X_CCS_categorical_train], axis = 1)
X_CCS_clean_train.head()
X_CCS_clean_train.shape 
scaled_X_CCS_numerical_test.reset_index(drop=True, inplace=True)
X_CCS_categorical_test.reset_index(drop=True, inplace=True)
X_CCS_clean_test = pd.concat([scaled_X_CCS_numerical_test, X_CCS_categorical_test], axis = 1)
X_CCS_clean_test.head()
X_CCS_clean_test.shape 

###########################
# Implementing GBM on CCS
###########################
GBM_CCS = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
max_depth=1, random_state=0)
GBM_CCS.fit(X_CCS_clean_train, y_CCS_train)
y_pred_GBM_CCS = GBM_CCS.predict(X_CCS_clean_test)
print("Accuracy of GBM_CCS:", metrics.accuracy_score(y_CCS_test, y_pred_GBM_CCS))
print("Precision:",metrics.precision_score(y_CCS_test, y_pred_GBM_CCS))
print("Recall:",metrics.recall_score(y_CCS_test, y_pred_GBM_CCS))
print(classification_report(y_CCS_test, y_pred_GBM_CCS))

# AUC
#-------------------------------
y_pred_prob_GBM_CCS = GBM_CCS.predict_proba(X_CCS_clean_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_CCS_test, y_pred_prob_GBM_CCS)
AUC = metrics.roc_auc_score(y_CCS_test, y_pred_prob_GBM_CCS)
plt.plot(fpr,tpr,label="GBM_CCS , AUC="+str(AUC))
plt.legend(loc=4)
plt.show()



# Calculating CI 95% for the metrics 
###############################################
y_pred_CCS_CI = np.array(y_pred_prob_GBM_CCS)
y_CCS_true = np.array(y_CCS_test)

n_bootstraps = 1000
rng_seed = 42  
rng = np.random.RandomState(rng_seed)

bootstrapped_accuracies_CCS = []
bootstrapped_precisions_CCS = []
bootstrapped_recalls_CCS = []
bootstrapped_f1s_CCS = []
bootstrapped_aucs_CCS = []

for i in range(n_bootstraps):
    # bootstrap by sampling with replacement on the prediction indices
    indices_CCS = rng.randint(0, len(y_pred_CCS_CI), len(y_pred_CCS_CI))
    if len(np.unique(y_CCS_true[indices_CCS])) < 2:
        # We need at least one positive and one negative sample for the metrics
        # to be defined: reject the sample
        continue
    
    binary_predictions_CCS = (y_pred_CCS_CI[indices_CCS] > 0.5).astype(int)
    
    accuracy_CCS = accuracy_score(y_CCS_true[indices_CCS], binary_predictions_CCS)
    precision_CCS= precision_score(y_CCS_true[indices_CCS], binary_predictions_CCS)
    recall_CCS = recall_score(y_CCS_true[indices_CCS], binary_predictions_CCS)
    f1_CCS = f1_score(y_CCS_true[indices_CCS], binary_predictions_CCS)
    auc_CCS = roc_auc_score(y_CCS_true[indices_CCS], y_pred_CCS_CI[indices_CCS])
    
    bootstrapped_accuracies_CCS.append(accuracy_CCS)
    bootstrapped_precisions_CCS.append(precision_CCS)
    bootstrapped_recalls_CCS.append(recall_CCS)
    bootstrapped_f1s_CCS.append(f1_CCS)
    bootstrapped_aucs_CCS.append(auc_CCS)

# Computing the 95% confidence intervals for each metric
accuracy_ci_CCS = (np.percentile(bootstrapped_accuracies_CCS, 2.5), np.percentile(bootstrapped_accuracies_CCS, 97.5))
precision_ci_CCS = (np.percentile(bootstrapped_precisions_CCS, 2.5), np.percentile(bootstrapped_precisions_CCS, 97.5))
recall_ci_CCS = (np.percentile(bootstrapped_recalls_CCS, 2.5), np.percentile(bootstrapped_recalls_CCS, 97.5))
f1_ci_CCS = (np.percentile(bootstrapped_f1s_CCS, 2.5), np.percentile(bootstrapped_f1s_CCS, 97.5))
auc_ci_CCS = (np.percentile(bootstrapped_aucs_CCS, 2.5), np.percentile(bootstrapped_aucs_CCS, 97.5))

print(f"Accuracy CI_CCS: [{accuracy_ci_CCS[0]:.3f} - {accuracy_ci_CCS[1]:.3f}]")
print(f"Precision CI_CCS: [{precision_ci_CCS[0]:.3f} - {precision_ci_CCS[1]:.3f}]")
print(f"Recall CI_CCS: [{recall_ci_CCS[0]:.3f} - {recall_ci_CCS[1]:.3f}]")
print(f"F1-Score CI_CCS: [{f1_ci_CCS[0]:.3f} - {f1_ci_CCS[1]:.3f}]")
print(f"AUC CI_CCS: [{auc_ci_CCS[0]:.3f} - {auc_ci_CCS[1]:.3f}]")


# plot the confusion materix (2*2 tabel)
#-----------------------------------------
cm_CCS = metrics.confusion_matrix(y_CCS_test, y_pred_GBM_CCS)
print('Confusion matrix GBM_CCS:', cm_CCS)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm_CCS), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

##################################
#   9. ICD_HIGHEST dataset       #
##################################
#####################################
# Dealing with imbalanced labels
#####################################
# Counting label classes 
#--------------------------
ICD_HIGHEST.shape
ICD_HIGHEST.groupby('readmission_90').count()
target_count = ICD_HIGHEST.readmission_90.value_counts()
target_count.plot(kind='bar', title='Count (readmission_90)')
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1') #(5.99 : 1)
ICD_HIGHEST.columns

# Downsampling
#------------------------
ICD_HIGHEST.sort_values(by=['subject_id','hadm_id'])
ICD_HIGHEST_majority = ICD_HIGHEST[ICD_HIGHEST.readmission_90==0]
ICD_HIGHEST_minority = ICD_HIGHEST[ICD_HIGHEST.readmission_90==1]
ICD_HIGHEST_majority_downsampled = resample (ICD_HIGHEST_majority,
                                            replace=False, n_samples =33007, 
                                            random_state=123)
# Combine minority class with downsampled majority class
ICD_HIGHEST_downsampled = pd.concat([ICD_HIGHEST_majority_downsampled, ICD_HIGHEST_minority])
ICD_HIGHEST_downsampled.readmission_90.value_counts()

# Dropping id variables 
#-------------------------
ICD_HIGHEST_downsampled.drop(['subject_id','hadm_id'], axis=1, inplace=True)
ICD_HIGHEST_downsampled.columns
ICD_HIGHEST_downsampled.dtypes
# Change variable type of age_group and gender to include them in categorical set of data
#------------------------------------------------------------------------------------------
ICD_HIGHEST_downsampled['age_group'] = ICD_HIGHEST_downsampled['age_group'].astype(str)
ICD_HIGHEST_downsampled['gender'] = ICD_HIGHEST_downsampled['gender'].astype(str)
# this only for ICD_highest group file
ICD_HIGHEST_downsampled['icd_gruppe'] = ICD_HIGHEST_downsampled['icd_gruppe'].astype(str)

# Splitting data into X,y
#-----------------------------
features =[]
for column in ICD_HIGHEST_downsampled.columns:
    if column != 'readmission_90':
        features.append(column)
    X_ICD_HIGHEST = ICD_HIGHEST_downsampled[features]
    y_ICD_HIGHEST = ICD_HIGHEST_downsampled['readmission_90']
X_ICD_HIGHEST.columns
y_ICD_HIGHEST.shape
X_ICD_HIGHEST.shape# ok

# Splitting X into numerical and categorical feature
#--------------------------------------------------------
X_ICD_HIGHEST_numerical =X_ICD_HIGHEST[['los', 'los_ED', 'no_ED_admissions']].copy()
X_ICD_HIGHEST_categorical = X_ICD_HIGHEST.select_dtypes(['object', 'int16', 'int8'])

# encoding 
#------------------
X_ICD_HIGHEST_categorical.columns
encoded_X_ICD_HIGHEST_categorical= pd.get_dummies(data=X_ICD_HIGHEST_categorical, columns=['admission_location', 'discharge_location','insurance','marital_status',
                                                                   'icd_gruppe', 'gender','age_group'])

encoded_X_ICD_HIGHEST_categorical.shape
X_ICD_HIGHEST_numerical.reset_index(drop=True, inplace=True)
encoded_X_ICD_HIGHEST_categorical.reset_index(drop=True, inplace=True)
X2_ICD_HIGHEST= pd.concat([X_ICD_HIGHEST_numerical, encoded_X_ICD_HIGHEST_categorical], axis = 1)
X2_ICD_HIGHEST.shape #(66014, 66)

# X split and standerdizing X_train, X_test 
#------------------------------------------------
X2_ICD_HIGHEST.dtypes
X_ICD_HIGHEST_train, X_ICD_HIGHEST_test, y_ICD_HIGHEST_train, y_ICD_HIGHEST_test = train_test_split(X2_ICD_HIGHEST,y_ICD_HIGHEST, test_size = 0.3, random_state = 0, stratify=y_ICD_HIGHEST)
X_ICD_HIGHEST_train.columns
X_ICD_HIGHEST_numerical_train = X_ICD_HIGHEST_train[['los', 'los_ED', 'no_ED_admissions']].copy() 
X_ICD_HIGHEST_numerical_train.shape
X_ICD_HIGHEST_categorical_train = X_ICD_HIGHEST_train.select_dtypes(['uint8','object'])
X_ICD_HIGHEST_categorical_train.shape
X_ICD_HIGHEST_numerical_test = X_ICD_HIGHEST_test[['los', 'los_ED', 'no_ED_admissions']].copy() 
X_ICD_HIGHEST_numerical_test.shape
X_ICD_HIGHEST_categorical_test = X_ICD_HIGHEST_test.select_dtypes(['uint8','object'])
X_ICD_HIGHEST_categorical_test.shape
for key in X_ICD_HIGHEST_numerical_train.keys():
    print (key)
for key in X_ICD_HIGHEST_categorical_train.keys():
    print (key)   
for key in X_ICD_HIGHEST_numerical_test.keys():
    print (key)  
for key in X_ICD_HIGHEST_categorical_test.keys():
    print (key)
    
scalar = StandardScaler()
scaled_X_ICD_HIGHEST_numerical_train= pd.DataFrame(scalar.fit_transform(X_ICD_HIGHEST_numerical_train), columns=X_ICD_HIGHEST_numerical_train.keys())
print(scaled_X_ICD_HIGHEST_numerical_train) # note here that we use fit_transform for X_train part
scaled_X_ICD_HIGHEST_numerical_test = pd.DataFrame(scalar.transform(X_ICD_HIGHEST_numerical_test), columns=X_ICD_HIGHEST_numerical_test.keys()) # note here that we only use the tranform (not fit_transform). That is to use the understanding from train data to test data #VERY IMPORTANT
print(scaled_X_ICD_HIGHEST_numerical_test)

# Recombining all X parts
#----------------------------------
scaled_X_ICD_HIGHEST_numerical_train.reset_index(drop=True, inplace=True)
X_ICD_HIGHEST_categorical_train.reset_index(drop=True, inplace=True)
X_ICD_HIGHEST_clean_train= pd.concat([scaled_X_ICD_HIGHEST_numerical_train, X_ICD_HIGHEST_categorical_train], axis = 1)
X_ICD_HIGHEST_clean_train.head()
X_ICD_HIGHEST_clean_train.shape 
scaled_X_ICD_HIGHEST_numerical_test.reset_index(drop=True, inplace=True)
X_ICD_HIGHEST_categorical_test.reset_index(drop=True, inplace=True)
X_ICD_HIGHEST_clean_test = pd.concat([scaled_X_ICD_HIGHEST_numerical_test, X_ICD_HIGHEST_categorical_test], axis = 1)
X_ICD_HIGHEST_clean_test.head()
X_ICD_HIGHEST_clean_test.shape 

###################################
# Implementing GBM on ICD-highest
###################################
GBM_ICD_HIGHEST = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
max_depth=1, random_state=0)
GBM_ICD_HIGHEST.fit(X_ICD_HIGHEST_clean_train, y_ICD_HIGHEST_train)
y_pred_GBM_ICD_HIGHEST = GBM_ICD_HIGHEST.predict(X_ICD_HIGHEST_clean_test)
print("Accuracy of GBM_ICD_HIGHEST:", metrics.accuracy_score(y_ICD_HIGHEST_test, y_pred_GBM_ICD_HIGHEST))
print("Precision:",metrics.precision_score(y_ICD_HIGHEST_test, y_pred_GBM_ICD_HIGHEST))
print("Recall:",metrics.recall_score(y_ICD_HIGHEST_test, y_pred_GBM_ICD_HIGHEST))
print(classification_report(y_ICD_HIGHEST_test, y_pred_GBM_ICD_HIGHEST))

# AUC
#-------------------------------
y_pred_prob_GBM_ICD_HIGHEST = GBM_ICD_HIGHEST.predict_proba(X_ICD_HIGHEST_clean_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_ICD_HIGHEST_test, y_pred_prob_GBM_ICD_HIGHEST)
AUC = metrics.roc_auc_score(y_ICD_HIGHEST_test, y_pred_prob_GBM_ICD_HIGHEST)
plt.plot(fpr,tpr,label="GBM_ICD_HIGHEST , AUC="+str(AUC))
plt.legend(loc=4)
plt.show()

# Calculating CI 95% for the metrics 
###############################################
y_pred_ICD_HIGHEST_CI = np.array(y_pred_prob_GBM_ICD_HIGHEST)
y_ICD_HIGHEST_true = np.array(y_ICD_HIGHEST_test)

n_bootstraps = 1000
rng_seed = 42  
rng = np.random.RandomState(rng_seed)

bootstrapped_accuracies_ICD_HIGHEST = []
bootstrapped_precisions_ICD_HIGHEST = []
bootstrapped_recalls_ICD_HIGHEST = []
bootstrapped_f1s_ICD_HIGHEST = []
bootstrapped_aucs_ICD_HIGHEST = []

for i in range(n_bootstraps):
    # bootstrap by sampling with replacement on the prediction indices
    indices_ICD_HIGHEST = rng.randint(0, len(y_pred_ICD_HIGHEST_CI), len(y_pred_ICD_HIGHEST_CI))
    if len(np.unique(y_ICD_HIGHEST_true[indices_ICD_HIGHEST])) < 2:
        # We need at least one positive and one negative sample for the metrics
        # to be defined: reject the sample
        continue
    
    binary_predictions_ICD_HIGHEST = (y_pred_ICD_HIGHEST_CI[indices_ICD_HIGHEST] > 0.5).astype(int)
    
    accuracy_ICD_HIGHEST = accuracy_score(y_ICD_HIGHEST_true[indices_ICD_HIGHEST], binary_predictions_ICD_HIGHEST)
    precision_ICD_HIGHEST= precision_score(y_ICD_HIGHEST_true[indices_ICD_HIGHEST], binary_predictions_ICD_HIGHEST)
    recall_ICD_HIGHEST = recall_score(y_ICD_HIGHEST_true[indices_ICD_HIGHEST], binary_predictions_ICD_HIGHEST)
    f1_ICD_HIGHEST = f1_score(y_ICD_HIGHEST_true[indices_ICD_HIGHEST], binary_predictions_ICD_HIGHEST)
    auc_ICD_HIGHEST = roc_auc_score(y_ICD_HIGHEST_true[indices_ICD_HIGHEST], y_pred_ICD_HIGHEST_CI[indices_ICD_HIGHEST])
    
    bootstrapped_accuracies_ICD_HIGHEST.append(accuracy_ICD_HIGHEST)
    bootstrapped_precisions_ICD_HIGHEST.append(precision_ICD_HIGHEST)
    bootstrapped_recalls_ICD_HIGHEST.append(recall_ICD_HIGHEST)
    bootstrapped_f1s_ICD_HIGHEST.append(f1_ICD_HIGHEST)
    bootstrapped_aucs_ICD_HIGHEST.append(auc_ICD_HIGHEST)

# Computing the 95% confidence intervals for each metric
accuracy_ci_ICD_HIGHEST = (np.percentile(bootstrapped_accuracies_ICD_HIGHEST, 2.5), np.percentile(bootstrapped_accuracies_ICD_HIGHEST, 97.5))
precision_ci_ICD_HIGHEST = (np.percentile(bootstrapped_precisions_ICD_HIGHEST, 2.5), np.percentile(bootstrapped_precisions_ICD_HIGHEST, 97.5))
recall_ci_ICD_HIGHEST = (np.percentile(bootstrapped_recalls_ICD_HIGHEST, 2.5), np.percentile(bootstrapped_recalls_ICD_HIGHEST, 97.5))
f1_ci_ICD_HIGHEST = (np.percentile(bootstrapped_f1s_ICD_HIGHEST, 2.5), np.percentile(bootstrapped_f1s_ICD_HIGHEST, 97.5))
auc_ci_ICD_HIGHEST = (np.percentile(bootstrapped_aucs_ICD_HIGHEST, 2.5), np.percentile(bootstrapped_aucs_ICD_HIGHEST, 97.5))

print(f"Accuracy CI_ICD_HIGHEST: [{accuracy_ci_ICD_HIGHEST[0]:.3f} - {accuracy_ci_ICD_HIGHEST[1]:.3f}]")
print(f"Precision CI_ICD_HIGHEST: [{precision_ci_ICD_HIGHEST[0]:.3f} - {precision_ci_ICD_HIGHEST[1]:.3f}]")
print(f"Recall CI_ICD_HIGHEST: [{recall_ci_ICD_HIGHEST[0]:.3f} - {recall_ci_ICD_HIGHEST[1]:.3f}]")
print(f"F1-Score CI_ICD_HIGHEST: [{f1_ci_ICD_HIGHEST[0]:.3f} - {f1_ci_ICD_HIGHEST[1]:.3f}]")
print(f"AUC CI_ICD_HIGHEST: [{auc_ci_ICD_HIGHEST[0]:.3f} - {auc_ci_ICD_HIGHEST[1]:.3f}]")

# plot the confusion materix (2*2 tabel)
#-----------------------------------------
cm_ICD_HIGHEST = metrics.confusion_matrix(y_ICD_HIGHEST_test, y_pred_GBM_ICD_HIGHEST)
print('Confusion matrix GBM_ICD_HIGHEST:', cm_ICD_HIGHEST)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm_ICD_HIGHEST), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
