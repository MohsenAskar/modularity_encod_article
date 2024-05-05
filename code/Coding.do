***********************************************************************
*                   MODULARITY ENCODING                               *
*                     Mohsen Askar                                    *
*                  Dataset preprocessing                              *
***********************************************************************
// Mimic documentaion: https://mimic.mit.edu/docs/ 
// see also: https://people.cs.pitt.edu/~jlee/note/intro_to_mimic_db.pdf 
// Stata in Jupyter 
import stata_setup
stata_setup.config("C:/Program Files/Stata17", "mp")
*** Data preprocessing
// 1. Importing, perprocessing patients.csv file 
//-------------------------------------------------
// 1.a. Creating a variable for date of birth
import delimited "D:\MIMIC_III\MIMIC_III_Extract\PATIENTS.csv", encoding(UTF-8)
codebook dob //deidentified date of birth

//  Reformatting date variable in readable dates
gen date_of_birth=date(dob,"YMD###")

// 1.b. Recoding gender variabel (1=male, 0=female)
gen gender_binary = 0
recode gender_binary (0=1) if gender =="M"
drop gender
rename gender_binary gender
// saving
save "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Tempor_Data_Files\Patients_Info.dta"

// 2. Importing, determining readmissions (outcomes)
//-----------------------------------------------------
import delimited "D:\MIMIC_III\MIMIC_III_Extract\ADMISSIONS.csv", encoding(ISO-8859-2) clear 

gen admittime2=date(admittime,"YMD###")

gen dischtime2=date(dischtime,"YMD###")
// 2.a. Generating a variable indicating Legnth of Stay (LOS)
gen los= dischtime2 - admittime2

// Creating Lag, Lead variables to calculate day difference between admissions
sort subject_id admittime dischtime admittime2 dischtime2
by subject_id : gen lead_date= admittime2[_n+1]
gen delta_days = lead_date - dischtime2
recode delta_days (.=0)
// Two obseravtions got negative values beacuse of registering error in time of discharge/admission. Changeing all negative values to positive values in delta_days
replace delta_days = delta_days *(-1) if delta_days<0
// 2.b. Generating variables for 30,90,120 and 180-day readmission
gen readmission_30 = 0
recode readmission_30 (0=1) if delta_days <=30
recode readmission_30 (1=0) if delta_days == 0   // 3,355 readmissions
// 90-day
gen readmission_90 = 0
recode readmission_90 (0=1) if delta_days <=90
recode readmission_90 (1=0) if delta_days == 0    //  5,552 readmissions
// 120-day
gen readmission_120 = 0
recode readmission_120 (0=1) if delta_days <=120
recode readmission_120 (1=0) if delta_days == 0    // 6,123 readmissions
// 180-day
gen readmission_180 = 0
recode readmission_180 (0=1) if delta_days <=180
recode readmission_180 (1=0) if delta_days == 0   //  7,020 readmissions
// All_readmissions
gen readmission_all = 0
recode readmission_all (0=1) if delta_days > 0  //  12,390 readmissions

// 2.c. Creating a variable indicating difference in time between ED admission and ED discharge (LOS_ED)
sort subject_id hadm_id
generate double edregtime2 = clock(edregtime , "YMDhms")
format edregtime2 %tc
generate double edouttime2 = clock(edouttime , "YMDhms")
format edouttime2 %tc
// As the difference here is expressed in milliseconds in Stata we divide by 3600000
gen los_ED = (edouttime2 - edregtime2)/3600000 // 13 values were found negative because of reverse registration of edregtime and edouttime, we converted these values to positive 
replace los_ED = los_ED *(-1) if los_ED <0
//recode missing values (inidcating no ED visits to 0)
recode los_ED(.=0)

// 2.d. making a variable indicating ED admission/not (1=ED admission, 0= not)
gen ED_admission = 1
recode ED_admission (1=0) if edregtime==""
// 2.e. making a variable for number of ED_admissions for each patient
sort subject_id ED_admission
by subject_id ED_admission : gen no_ED_admissions = _N 
// 2.f. replacing missing values in language, religion and marital status with "na"
replace language = "na" if (language=="")
replace marital_status = "na" if (marital_status =="")
replace religion = "na" if (religion =="")
//Saving
save "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Tempor_Data_Files\Readmissions_LOS.dta"

*** Joining files (Master file is Readmission_LOS.dta)
//----------------------------------------------------------
// Joining with Patients_Info
joinby subject_id using "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Tempor_Data_Files\Patients_Info.dta"

// Note that Patients who are older than 89 years old at any time in the database have had their date of birth shifted to obscure their age and comply with HIPAA. The shift process was as follows: the patient’s age at their first admission was determined. The date of birth was then set to exactly 300 years before their first admission. see https://mimic.mit.edu/docs/iii/tables/patients/ 

// 3. Age calculation: take difference between date of birth and first admission time (so we take first admission time as our index data for each patient in the database). 
//-------------------------------------------------------------------------------------------------------
sort subject_id hadm_id admittime
// make a lead variabel 
by subject_id : gen first_admission = 1 if _n == 1
// replace string with date format
replace first_admission = admittime2 if first_admission==1
gen age = (first_admission- date_of_birth )/365
xfill age, i(subject_id)
count if age >=300
hist age
// 4. creating age groups 
//-------------------------------
gen age_group = age
recode age_group 0/25 = 1
replace age_group = 2 if (age_group >25) & (age_group <=50)
replace age_group = 3 if (age_group >50) & (age_group <=60)
replace age_group = 4 if (age_group >60) & (age_group <=70)
replace age_group = 5 if (age_group >70) & (age_group <=80)
replace age_group = 6 if (age_group >80) & (age_group <=90)
replace age_group = 7 if (age_group >90) & (age_group <.)
tab age_group
hist age_group
// fill age_group with the same value for each patient to rehsape the data and avoid false missing values, we first install xfill 
net from https://www.sealedenvelope.com/
xfill age_group , i(subject_id)

** Joining with ICD codes 
//------------------------------------
joinby subject_id hadm_id using "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\ICD_9_MIMIC_III.dta"
// Saving 
save "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Main_File.dta"

** Exclusions
//----------------
// 1. Excluding < 18 years
// As the newest date of birth in the dataset is 24jul2201, substracting 18 years from all data will exclude popualtion who are younger that 18 years, we substract 24jul2183 from all date_of_birth
generate eighteen_years = age( date_of_birth , td(24jul2183))
// all observations got (.) indicate date of borth after 24 july 2183 (> 18 years)
recode eighteen_years (.=0)
drop if eighteen_years ==0 //8,697 observations deleted
// 2. Elective and newborn admissions
tab admission_type
drop if (admission_type =="ELECTIVE") //(74,397 observations deleted)
drop if (admission_type == "NEWBORN") //(37,209 observations deleted) // total: 111,606 observations
// 3. Drop patients who died at the hospital 
drop if (hospital_expire_flag == 1) // (78,291 obseravtions)
// 4. drop if ICD code is missing 
drop if icd9_code =="" // (8 obseravtions excluded)
// 5. drop icd codes which has a registration error
merge m:1 icd9_code using "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Main_Datasets_For_Analysis\Error_Registered_ICD_Codes_In_MIMIC_data.dta"
drop if _merge == 3 //(10,078 observations deleted)
drop if _merge == 2 // (18 observations deleted), Total obs. after exclusions 442,367 (29,247 unique patients and 37,762 unique admissions)
drop short_title long_title _merge
save "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Main_Datasets_For_Analysis\Data_After_Exclusions.dta"

** Keeping only relevant variables for the final model
//--------------------------------------------------------
use "Data_After_Exclusions.dta"
// Keeping relavant variables
keep subject_id hadm_id admission_location discharge_location insurance marital_status los readmission_90 gender icd9_code age_group los_ED ED_admission no_ED_admissions

// we dropped "has_chartevents_data" "admission_type" "language" "ethinicity" as they showed very high dominancy of one (some) category(ies) over the other, meaning they will contribute with very little information to the model 

// drop duplicates if any 
duplicates drop // 42 obseravtions dropped
save "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Main_Datasets_For_Analysis\Selected_Variables_Final_Model.dta", replace //final model data

// As we are going to implement log. reg. we should remove highly correlated variables to avoid multicolinearity
correlate los readmission_90 los_ED ED_admission no_ED_admissions gender age_group
// ED_admission and los_ED are highly correlated we will exclude ED_admission as no_ED_admissions can express the same information and it is a strong predector to the outcome
drop ED_admission
// save

** Make dataset in which we recode ICD-9 codes to 18 codes which represent the highest hirarchy in ICD-9
use "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Main_Datasets_For_Analysis\Selected_Variables_Final_Model.dta"
// extract the first 3 character of ICD9 codes
gen icd_gruppe =substr(icd9_code,1,3)
// Recode icd9-codes which starts with E or V
replace icd_gruppe = "18" if substr(icd_gruppe,1,1) == "E"
replace icd_gruppe = "18" if substr(icd_gruppe,1,1) == "V"
// convert string to numeric
generate icd_gruppe_num = real(icd_gruppe)
drop icd_gruppe
rename icd_gruppe_num icd_gruppe
sort icd_gruppe
recode icd_gruppe (18=1000)
replace icd_gruppe = 1 if ( icd_gruppe>0) & (icd_gruppe <=139)
replace icd_gruppe = 2 if ( icd_gruppe>=140) & (icd_gruppe <=239)
replace icd_gruppe = 3 if ( icd_gruppe>=240) & (icd_gruppe <=279)
replace icd_gruppe = 4 if ( icd_gruppe>=280) & (icd_gruppe <=289)
replace icd_gruppe = 5 if ( icd_gruppe>=290) & (icd_gruppe <=319)
replace icd_gruppe = 6 if ( icd_gruppe>=320) & (icd_gruppe <=389)
replace icd_gruppe = 7 if ( icd_gruppe>=390) & (icd_gruppe <=459)
replace icd_gruppe = 8 if ( icd_gruppe>=460) & (icd_gruppe <=519)
replace icd_gruppe = 9 if ( icd_gruppe>=520) & (icd_gruppe <=579)
replace icd_gruppe = 10 if ( icd_gruppe>=580) & (icd_gruppe <=629)
replace icd_gruppe = 11 if ( icd_gruppe>=630) & (icd_gruppe <=679)
replace icd_gruppe = 12 if ( icd_gruppe>=680) & (icd_gruppe <=709)
replace icd_gruppe = 13 if ( icd_gruppe>=710) & (icd_gruppe <=739)
replace icd_gruppe = 14 if ( icd_gruppe>=740) & (icd_gruppe <=759)
replace icd_gruppe = 15 if ( icd_gruppe>=760) & (icd_gruppe <=779)
replace icd_gruppe = 16 if ( icd_gruppe>=780) & (icd_gruppe <=799)
replace icd_gruppe = 17 if ( icd_gruppe>=800) & (icd_gruppe <=999)
recode icd_gruppe (1000=18)
tab icd_gruppe
// Drop ICD codes
drop icd9_code
duplicates drop 
save "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Main_Datasets_For_Analysis\Data_ICD_Recoded_Highest_Hirarchy_90_days.dta"

** Making ICD networks from MIMIC III dataset (full file without exclusions)
// 1. Network of the whole MIMIC_III (Main_Network)
import delimited "D:\MIMIC_III\MIMIC_III_Extract\DIAGNOSES_ICD.csv", encoding(ISO-8859-2)
merge m:1 icd9_code using "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Main_Datasets_For_Analysis\Error_Registered_ICD_Codes_In_MIMIC_data.dta"
drop if _merge == 3
drop if _merge == 2
keep subject_id icd9_code
duplicates drop
save "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\Patient_MIMIC_III_ICD.dta"
rename icd9_code icd9_code2
joinby subject_id using "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\Patient_MIMIC_III_ICD.dta"
bysort icd9_code icd9_code2 :egen edges=count(subject_id)
drop if icd9_code == icd9_code2
drop subject_id
duplicates drop
drop if missing(icd9_code)
drop if missing(icd9_code2)
// To create .net file (for Gephi)
nwfromedge icd9_code icd9_code2 edges, undirected keeporiginal
nwexport, type(pajek)

// Network summary from STATA

   Directed: false
   Nodes: 6840
   Edges: 952676
   Minimum value:  0
   Maximum value:  6596
   Density:  .0407311352417208

// Importing to Gephi for modularity detection and visualization
// Merging ICD codes with different grouping schemes
// Number of unique ICD9 codes in mimic iii file (14,567) in resolution files (6,840) in CSS file (15,071) 
// we perform moularity in Gephi and export files as .csv which will be imported to Stata like this for exapmle:
import delimited "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\CSV_Files_Networks\R0.01_M1078.csv", clear 
rename label icd9_code
drop timeset
replace icd9_code = subinstr(icd9_code , "_", "", .)
// same with all files 
// Merging ICD codes to their corresponding css and modularity group
merge 1:1 icd9_code using "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Grouping_Systems\Stata_Files\ICD_Diagnosis_From_MIMIC.dta"
drop if _merge == 2
drop _merge
drop row_id
save "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Grouping_Systems\Stata_Files\R1_M8.dta", replace
// and same for other files

// For css file 
import delimited "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Grouping_Systems\icd9_to_singleCCScategory.csv", clear 
rename icd9cmcode icd9_code
merge 1:1 icd9_code using "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Grouping_Systems\Stata_Files\ICD_Diagnosis_From_MIMIC.dta"
drop if _merge == 1
drop if _merge == 2
drop _merge row_id

** Recoding ICD accoriding to their referance
use "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\90_days\Selected_Variables_Final_Model.dta"
// merge with the referance file
merge m:1 icd9_code using "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Grouping_Systems\Stata_Files\R0.5_M47.dta"
drop if _merge == 1
drop if _merge == 2
drop _merge long_title short_title id icd9_code
duplicates drop
save "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\test_R0.5_M47.dta"
// same with other files

***************************
*      DRG codes          *
***************************
// Network for DRG codes
//--------------------------  
// Datapreprocessing for DRG is just the same as ICD dataset replacing ICD codes with DRG codes and holding the same features. For DRG we made only one resolution netwrok R1_M24 which we will compare to raw dataset using ML models 
import delimited "D:\MIMIC_III\MIMIC_III_Extract\DRGCODES.csv", clear 
codebook drg_code
sort subject_id hadm_id
keep subject_id drg_code
duplicates drop
save "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\DRG_MIMIC_III.dta"
rename drg_code drg_code2
joinby subject_id using "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\DRG_MIMIC_III.dta"
bysort drg_code2 drg_code :egen edges=count(subject_id)
drop if drg_code2 == drg_code
drop subject_id
duplicates drop
drop if missing( drg_code2 )
drop if missing( drg_code )
nwfromedge drg_code drg_code2 edges, undirected keeporiginal
nwexport, type(pajek)

// DRG Netwrok summary
//--------------------------
   Network name:  network
   Network id:  1
   Directed: false
   Nodes: 1661
   Edges: 41274
   Minimum value:  0
   Maximum value:  910
   Density:  .0299384171242465
   
*********************************************************************************************************
// Here, ends the main work needed for the paper. What comes after are attempts or optional things which may or may not be included in the study.


******************************
*   Quarter the dataset      *
******************************    
//Optional // 2. Network of quarter the dataset (1/4 MIMIC_III) (Network_Quarter)
//-----------------------------------------------------------------------------------
// OPTIONAL// 10. Extracting a smaller set of dataset (1/4 of the dataset), patient ID and admission time are taken into consideration (means that we will take about 3 years of comorbidity -instead of 12- according to admission time)
sort subject_id admittime admittime2
keep in 1/162762  // 1/4 the dataset
save "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Quarter_The_Dataset.dta"

// we can then exclude other variables and make an edge list from icd9_code and subject_id to generate the network (Don't forget to apply exclusions on the quarter_dataset after generating the network)
// Applying the exclusions on Quarter_The_Dataset.dta after we made the network already
// 1. Excluding < 18 years
generate eighteen_years = age( date_of_birth , td(24jul2183))
// all observations got (.) indicate date of borth after 24 july 2183 (> 18 years)
recode eighteen_years (.=0)
drop if eighteen_years ==0
// 2. Elective and newborn admissions
tab admission_type
drop if (admission_type =="ELECTIVE")
drop if (admission_type == "NEWBORN")
// 3. Drop patients who died at the hospital 
drop if (hospital_expire_flag == 1)
// 4. drop if ICD code is missing 
drop if icd9_code ==""
// 5. drop icd codes which has a registration error
merge m:1 icd9_code using "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Main_Datasets_For_Analysis\Error_Registered_ICD_Codes_In_MIMIC_data.dta"
drop if _merge == 3
drop if _merge == 2
drop short_title long_title _merge
// save 
save "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Main_Datasets_For_Analysis\Quarter_The_Dataset_Exclusions_Applied.dta", replace

use "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Main_Datasets_For_Analysis\Quarter_The_Dataset_After_Exclusions.dta", clear

sort subject_id admittime admittime2
keep subject_id icd9_code
duplicates drop
save "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\Patient_MIMIC_III_ICD.dta"
rename icd9_code icd9_code2
joinby subject_id using "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\Patient_MIMIC_III_ICD.dta"
bysort icd9_code icd9_code2 :egen edges=count(subject_id)
drop if icd9_code == icd9_code2
drop subject_id
duplicates drop
drop if missing(icd9_code)
drop if missing(icd9_code2)
nwfromedge icd9_code icd9_code2 edges, undirected keeporiginal
nwexport, type(pajek) replace
// Network summary 

   Directed: false
   Nodes: 4069
   Edges: 305612
   Minimum value:  0
   Maximum value:  1498
   Density:  .0369259574213065
   
*********************************
*          K_means              *
*********************************
//OPTIONAL// ** Making data for k-means clustering according to total cooccurancies in the dataset
//----------------------------------------------------------------------------------------------------
import delimited "D:\MIMIC_III\MIMIC_III_Extract\DIAGNOSES_ICD.csv", encoding(ISO-8859-2)clear 
keep subject_id icd9_code
duplicates drop
save "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\Patient_MIMIC_III_ICD.dta"
rename icd9_code icd9_code2
joinby subject_id using "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\Patient_MIMIC_III_ICD.dta"
bysort icd9_code icd9_code2 :egen edges=count(subject_id)
drop if icd9_code == icd9_code2
drop if missing(icd9_code)
drop if missing(icd9_code2)
drop icd9_code2
duplicates drop
rename edges cooccurancies
bysort subject_id icd9_code :egen total=count(cooccurancies)
drop cooccurancies
rename total cooccurancies
duplicates drop
codebook icd9_code
save "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Main_Datasets_For_Analysis\K-means_Dataset.dta"

*******************************************************
*      Reshaping data for frequent pattern mining FPM *
*******************************************************
// ** Reshaping the data from long to wide format (for FRP)
// Generating seq variable (seq of each patient's rows) to be as (j) in reshaping
// This form of data won't be used for ML models, but for frequent pattern mining FRP //
sort subject_id hadm_id icd9_code
by subject_id hadm_id: gen seq =_n
reshape wide icd9_code, i(hadm_id) j(seq) // seq is automatically dropped
save "C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD_Modularit_Paper\Main_Datasets_For_Analysis\Data_Short_Wide.dta"
// replace empty cells with NaN
foreach var of varlist icd9_code1-icd9_code39 {
  replace `var' = "NaN" if `var' ==""
  }

