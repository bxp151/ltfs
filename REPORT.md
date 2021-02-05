# Predicting vehicle loan defaults for LTFS India

</br>

## Objective
Financial institutions have incurred significant losses due to vehicle loans defaults. This has led to the restriction of vehicle loan underwriting and increased vehicle loan rejection rates.  A financial institution has hired you to help improve their credit scoring model by predicting the occurrence of vehicle loan defaults. 

The data was provided by LTFS India during a [machine learning hackathon](https://datahack.analyticsvidhya.com/contest/ltfs-datascience-finhack-an-online-hackathon/).

</br>


## Performance Measure
The model performance will be measured using area under the curve (AUC).  AUC tells us how well the model seperates classes (non-default vs. default).  A score of 1 would mean the model perfectly seperates classes while a score of 0.5 would mean the model has no power to seperate classes.  



</br>


## Key Findings

1. The final model achieved an AUC of 0.66075 on the test data.         

2. The most important features to predict default were

	* **ltv --**  Proportion of loan amount to the value of the asset being borrowed against
	* **perform.cns.score --** Credit bureau score 
	* **average.acct.age --** Average age of customer loans
	* **prop.account.age --** Ratio of average account age to credit history length
	* **state.id.13 --** Loans disbursed in state 13


![](./images/fig1.png)


## Model Validation
In order to validate the model, we used target shuffling which shows the probability that the model's results occured by chance. 
    
    For 200 repititions:
        
        1. Shuffle the target 
        2. Fit the best model to the shuffled target (shuffled model)
        3. Make predictions using the shuffled model and score using AUC
        4. Plot the distribution of scores 
</br>

Since the best model performed better than every target permutation model, there is a 0 in 200 probability that the model's results occured by chance

![](./images/fig2.png)


## Approach

The overall approach to building the default prediction model:

1. Initial data exploration
2. Select modeling techniques
3. Split data into Train/Test
4. Build and analyze baseline models
5. Feature engineering
6. Build and analyze final models
7. Final predictions using test set

</br>


## Potential Model Improvements

1. There was a possibility of engineering more features to achieve a better AUC score
2. More powerful algorithms such as CatBoost or LightGBM could have been used to imnprove scoring

<br>
<br>



## Original Data Definitions

The original data are 233154 observations on 40 variables. loan_default is the target variable

Variable | Description
---- | -----------  
UniqueID | Identifier for customers 
loan_default (target)| Payment default in the first EMI on due date 
disbursed_amount | Amount of Loan disbursed 
asset_cost | Cost of the Asset 
ltv | Loan to Value of the asset 
branch_id | Branch where the loan was disbursed 
supplier_id | Vehicle Dealer where the loan was disbursed 
manufacturer_id | "Vehicle manufacturer(Hero ,  Honda ,  TVS etc.)" 
Current_pincode | Current pincode of the customer 
Date.of.Birth | Date of birth of the customer 
Employment.Type | Employment Type of the customer (Salaried/Self Employed) 
DisbursalDate | Date of disbursement 
State_ID | State of disbursement 
Employee_code_ID | Employee of the organization who logged the disbursement 
MobileNo_Avl_Flag | if Mobile no. was shared by the customer then flagged as 1 
Aadhar_flag | if aadhar was shared by the customer then flagged as 1 
PAN_flag | if pan was shared by the customer then flagged as 1 
VoterID_flag | if voter  was shared by the customer then flagged as 1 
Driving_flag | if DL was shared by the customer then flagged as 1 
Passport_flag | if passport was shared by the customer then flagged as 1 
PERFORM_CNS.SCORE | Bureau Score 
PERFORM_CNS.SCORE.DESCRIPTION | Bureau score description 
PRI.NO.OF.ACCTS | count of total loans taken by the customer at the time of disbursement 
PRI.ACTIVE.ACCTS | count of active loans taken by the customer at the time of disbursement 
PRI.OVERDUE.ACCTS | count of default accounts at the time of disbursement 
PRI.CURRENT.BALANCE | total Principal outstanding amount of the active loans at the time of disbursement 
PRI.SANCTIONED.AMOUNT | total amount that was sanctioned for all the loans at the time of disbursement 
PRI.DISBURSED.AMOUNT | total amount that was disbursed for all the loans at the time of disbursement 
SEC.NO.OF.ACCTS | count of total loans taken by the customer at the time of disbursement 
SEC.ACTIVE.ACCTS | count of active loans taken by the customer at the time of disbursement 
SEC.OVERDUE.ACCTS | count of default accounts at the time of disbursement 
SEC.CURRENT.BALANCE | total Principal outstanding amount of the active loans at the time of disbursement 
SEC.SANCTIONED.AMOUNT | total amount that was sanctioned for all the loans at the time of disbursement 
SEC.DISBURSED.AMOUNT | total amount that was disbursed for all the loans at the time of disbursement 
PRIMARY.INSTAL.AMT | EMI Amount of the primary loan 
SEC.INSTAL.AMT | EMI Amount of the secondary loan 
NEW.ACCTS.IN.LAST.SIX.MONTHS | New loans taken by the customer in last 6 months before the disbursment 
DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS | Loans defaulted in the last 6 months 
AVERAGE.ACCT.AGE | Average loan tenure 
CREDIT.HISTORY.LENGTH | Time since first loan 
NO.OF_INQUIRIES | Enquries done by the customer for loans 
PRI.Prefix (not a variable) | Primary accounts are those which the customer has taken for his personal use 
SEC.Prefix (not a variable) | Secondary accounts are those which the customer act as a co-applicant or gaurantor 

