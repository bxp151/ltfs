#%% imports

import exploretransform as et 
import os
import pandas as pd
import plotly.express as px
from plotly.offline import plot 
import numpy as np
import kaleido 
import pickle

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, \
    KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, SCORERS

from sklearn.ensemble import RandomForestClassifier, \
    GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgbc

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.inspection import permutation_importance

#%% Settings

HOME_DIR = '/Users/bxp151/ml/ltfs'
DATA_DIR = '/data'
IMG_DIR = '/images'
PICKLE_DIR = '/pickles'


if not os.path.exists("images"):
    os.mkdir(HOME_DIR + IMG_DIR)
    
if not os.path.exists("pickles"):
    os.mkdir(HOME_DIR + PICKLE_DIR)  
    
    
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 500)

#%% Initial Exploration

df = pd.read_csv(HOME_DIR + DATA_DIR+ '/ltfs.csv', 
                  parse_dates = ['DisbursalDate'])


df.columns = df.columns.str.lower()

df.columns = df.columns.str.replace('.', '_', regex = False)

# pk = et.peek(df)
# ex = et.explore(df)

df['perform_cns_score_description'].value_counts()

# check target proportions
df['loan_default'].value_counts()[0]/len(df) # 78% negative
df['loan_default'].value_counts()[1]/len(df) # 22% positive

X = df.drop(['loan_default'], axis=1)
y = df['loan_default']

#%% Split data


split = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)

for train_idx, test_idx in split.split(X, y):
    train, trainX, trainy  = df.iloc[train_idx], X.iloc[train_idx], y.iloc[train_idx]
    test, testX, testy = df.iloc[test_idx], X.iloc[test_idx], y.iloc[test_idx]

# verify class balance is maintained 
trainy.value_counts() / len(trainy)
testy.value_counts() / len(testy)

#%% Baseline Models - data exploration
base_vars = ['disbursed_amount' , 'asset_cost' , 'ltv' ,  
             'pri_no_of_accts' , 'pri_active_accts' , 'pri_overdue_accts' , 
             'pri_current_balance' , 'pri_sanctioned_amount' , 
             'pri_disbursed_amount' , 'primary_instal_amt' , 
             'new_accts_in_last_six_months' , 
             'delinquent_accts_in_last_six_months' , 'no_of_inquiries' ]

trX_base = trainX[base_vars]
try_base = trainy.copy()

desc = trX_base.describe()

# check distributions of features
trX_base.hist()
et.skewstats(trX_base)

'''
There were 376 negative balances in pri_active_accts , leaving them alone 
for now
'''
np.sum(trX_base["pri_current_balance"] < 0 )


'''
there isn't much visible seperation in the classes using these features
'''
# for col in trX_base.columns:
#     plot(px.box(trX_base, x= try_base , y = col))


#%% Baseline Models - Naive

# labels and the predicted probability only knowing the training labels
naive = pd.DataFrame({'y_true': trainy ,
                      'y_score': (trainy.value_counts() / len(trainy))[1]} )

base01_naive = roc_auc_score(naive['y_true'], naive['y_score'])

#%% base_cv

'''function to calculate the cv score for the baseline estimators'''

def base_cv(estimator):
    cv = cross_val_score(estimator = estimator, 
                        X = trX_base , 
                        y=try_base, 
                        scoring='roc_auc', 
                        cv=5)
    return np.mean(cv)
#%% Baseline Models

'''
Gradient Boost cv score of 0.6237614889712806 would land 551 out of 1328 on 
the leaderboard.  Will continue with adaboost, gradient boost and xgboost for
the next set of evaluations
'''

# bagging trees
base02_bag = base_cv(RandomForestClassifier(max_features = None, random_state=42))

# random forest
base03_rf = base_cv(RandomForestClassifier(max_features = 'sqrt', random_state=42))

# adaboost: 0.6181177808312794
base04_ada = base_cv(AdaBoostClassifier(random_state=42))

# gradient boost: 0.6237614889712806
base05_gb = base_cv(GradientBoostingClassifier(random_state=42))

# XGBoost: 0.6190087582124482
base06_xgb = base_cv(xgbc.XGBClassifier(use_label_encoder=False, random_state=42))

# Nueral Network
base07_nn = base_cv(MLPClassifier(max_iter=1000, random_state=42))

#%% Feature Engineering - Exploration

trX_feat, try_feat = trainX.copy(), trainy.copy()

# combine all IDs except mobile ID into a single feature
trX_feat["n_ids_shared"] =  trX_feat["mobileno_avl_flag"] + \
trX_feat["aadhar_flag"] + trX_feat["pan_flag"] + trX_feat["voterid_flag"] + \
trX_feat["driving_flag"] + trX_feat["passport_flag"]

# drop all ID features
trX_feat = trX_feat.drop(["mobileno_avl_flag","aadhar_flag","pan_flag",
                          "voterid_flag","driving_flag","passport_flag"], axis = 1)

# combine pri and sec features
trX_feat["no_of_accts"] = trX_feat["pri_no_of_accts"] + trX_feat["sec_no_of_accts"]
trX_feat["active_accts"] = trX_feat["pri_active_accts"] + trX_feat["sec_active_accts"]
trX_feat["overdue_accts"] = trX_feat["pri_overdue_accts"] + trX_feat["sec_overdue_accts"]
trX_feat["current_balance"] = trX_feat["pri_current_balance"] + trX_feat["sec_current_balance"]
trX_feat["sanctioned_amount"] = trX_feat["pri_sanctioned_amount"] + trX_feat["sec_sanctioned_amount"]
trX_feat["disbursed_amount"] = trX_feat["pri_disbursed_amount"] + trX_feat["sec_disbursed_amount"]
trX_feat["install_amt"] = trX_feat["primary_instal_amt"] + trX_feat["sec_instal_amt"]

# drop all original pri and sec features
trX_feat = trX_feat.drop(["pri_no_of_accts","pri_active_accts","pri_overdue_accts",
                          "pri_current_balance","pri_sanctioned_amount",
                          "pri_disbursed_amount","sec_no_of_accts",
                          "sec_active_accts","sec_overdue_accts","sec_current_balance",
                          "sec_sanctioned_amount","sec_disbursed_amount",
                          "primary_instal_amt","sec_instal_amt"], axis = 1)


def months(x):
    '''
    Parameters
    ----------
    x : String in the form 'Xyrs Ymon' where X & Y can be one or two digits

    Returns
    -------
    t : total months
    '''
    
    y, m = x.split(sep = ' ')
    y, _ = y.split(sep = 'y')
    m, _ = m.split(sep = 'm')
    t = int(y) * 12 + int(m)
    return t

# Convert average_acct_age 
trX_feat["average_acct_age"] = trX_feat["average_acct_age"].apply(months, convert_dtype=True)

# Convert credit_history_length
trX_feat["credit_history_length"] = trX_feat["credit_history_length"].apply(months, convert_dtype=True)


'''
Create a function to fix the ambibuity in the date of birth

1. Isolate the year into a series and manually examine
2. Figure out the cutoff
3. Append 19 or 20 where appropriate

'''


def fixdob(x):
    '''
    Parameters
    ----------
    x : Date of birth in form DD-MM-YY

    Returns
    -------
    dob : Date of birth in form DD-MM-YYYY
            If YY < 48 then 20YY
            IF YY >=48 then 19YY

    '''
    
    day_month = x[0:6]
    year = x[6:8]
    
    if int(year) < 48:
        dob = day_month + '20' + year
    else:
        dob = day_month + '19' + year
    return dob

# fix DOBs and convert to datetime object
trX_feat['date_of_birth'] = trX_feat['date_of_birth'].apply(fixdob, convert_dtype=True)
trX_feat['date_of_birth'] = pd.to_datetime(trX_feat['date_of_birth'])

# create age (in months)  disbursaldate - date_of_birth 
def to_months(d1, d2):
    return d1.month - d2.month + 12*(d1.year - d2.year)

trX_feat['age'] = trX_feat.apply(lambda x: to_months(x.disbursaldate, x.date_of_birth), axis=1)

# How long each account was open in relationship to the credit history 
trX_feat['prop_account_age'] = trX_feat['average_acct_age'] /  \
    (trX_feat['credit_history_length'])
    
trX_feat['prop_account_age'].fillna(0, inplace=True)
    
# Set all n/a in employment_type to "missing"
trX_feat['employment_type'].fillna("not listed", inplace=True)

# drop features
trX_feat = trX_feat.drop(['uniqueid', 'supplier_id' , 'current_pincode_id' , 
                          'date_of_birth' , 'disbursaldate' , 
                          'employee_code_id'], axis=1)

'''
Explore all categorical variables 

Average default in train/test
0    0.782928
1    0.217072

'''

def crosstab(col):
    '''
    Returns a contingency table with normalized columns sorted by default
    '''
    prop = pd.crosstab(trX_feat[col], try_feat, normalize = 'index')
    num = pd.crosstab(trX_feat[col], try_feat, margins=True)
    ct = prop.join(num, lsuffix="_prop", 
              rsuffix="_num").sort_values("1_prop", ascending=False)
    return ct
    

crosstab('employment_type')
crosstab('manufacturer_id')
crosstab('state_id') 
crosstab('perform_cns_score_description')
crosstab('branch_id')

# convert features to categorical
trX_feat[['employment_type', 'branch_id', 'manufacturer_id', 'state_id',
         'perform_cns_score_description']] = \
trX_feat[['employment_type', 'branch_id', 'manufacturer_id', 'state_id',
         'perform_cns_score_description']].astype('category')


#%% Function to automate feature engineering steps

def feateng(df):
    
    df = df.copy()
    # combine all IDs except mobile ID into a single feature
    df["n_ids_shared"] =  df["mobileno_avl_flag"] + \
    df["aadhar_flag"] + df["pan_flag"] + df["voterid_flag"] + \
    df["driving_flag"] + df["passport_flag"]

    # drop all ID features
    df = df.drop(["mobileno_avl_flag","aadhar_flag","pan_flag",
                              "voterid_flag","driving_flag","passport_flag"], axis = 1)

    # combine pri and sec features
    df["no_of_accts"] = df["pri_no_of_accts"] + df["sec_no_of_accts"]
    df["active_accts"] = df["pri_active_accts"] + df["sec_active_accts"]
    df["overdue_accts"] = df["pri_overdue_accts"] + df["sec_overdue_accts"]
    df["current_balance"] = df["pri_current_balance"] + df["sec_current_balance"]
    df["sanctioned_amount"] = df["pri_sanctioned_amount"] + df["sec_sanctioned_amount"]
    df["disbursed_amount"] = df["pri_disbursed_amount"] + df["sec_disbursed_amount"]
    df["install_amt"] = df["primary_instal_amt"] + df["sec_instal_amt"]

    # drop all original pri and sec features
    df = df.drop(["pri_no_of_accts","pri_active_accts","pri_overdue_accts",
                              "pri_current_balance","pri_sanctioned_amount",
                              "pri_disbursed_amount","sec_no_of_accts",
                              "sec_active_accts","sec_overdue_accts","sec_current_balance",
                              "sec_sanctioned_amount","sec_disbursed_amount",
                              "primary_instal_amt","sec_instal_amt"], axis = 1)


    def months(x):
        '''
        Parameters
        ----------
        x : String in the form 'Xyrs Ymon' where X & Y can be one or two digits

        Returns
        -------
        t : total months
        '''
        
        y, m = x.split(sep = ' ')
        y, _ = y.split(sep = 'y')
        m, _ = m.split(sep = 'm')
        t = int(y) * 12 + int(m)
        return t

    # Convert average_acct_age 
    df["average_acct_age"] = df["average_acct_age"].apply(months, convert_dtype=True)

    # Convert credit_history_length
    df["credit_history_length"] = df["credit_history_length"].apply(months, convert_dtype=True)


    '''
    Create a function to fix the ambiguity in the date of birth

    1. Isolate the year into a series and manually examine
    2. Figure out the cutoff
    3. Append 19 or 20 where appropriate

    '''


    def fixdob(x):
        '''
        Parameters
        ----------
        x : Date of birth in form DD-MM-YY

        Returns
        -------
        dob : Date of birth in form DD-MM-YYYY
                If YY < 48 then 20YY
                IF YY >=48 then 19YY

        '''
        
        day_month = x[0:6]
        year = x[6:8]
        
        if int(year) < 48:
            dob = day_month + '20' + year
        else:
            dob = day_month + '19' + year
        return dob

    # fix DOBs and convert to datetime object
    df['date_of_birth'] = df['date_of_birth'].apply(fixdob, convert_dtype=True)
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'])

    # create age (in months)  disbursaldate - date_of_birth 
    def to_months(d1, d2):
        return d1.month - d2.month + 12*(d1.year - d2.year)

    df['age'] = df.apply(lambda x: to_months(x.disbursaldate, x.date_of_birth), axis=1)

    # How long each account was open in relationship to the credit history 
    df['prop_account_age'] = df['average_acct_age'] /  \
        (df['credit_history_length'])
        
    df['prop_account_age'].fillna(0, inplace=True)
        
    # Set all n/a in employment_type to "missing"
    df['employment_type'].fillna("not listed", inplace=True)

    # drop features
    df = df.drop(['uniqueid', 'supplier_id' , 'current_pincode_id' , 
                              'date_of_birth' , 'disbursaldate' , 
                              'employee_code_id'], axis=1)


    # convert features to categorical
    df[['employment_type', 'branch_id', 'manufacturer_id', 'state_id',
             'perform_cns_score_description']] = \
    df[['employment_type', 'branch_id', 'manufacturer_id', 'state_id',
             'perform_cns_score_description']].astype('category')

    return df


trX_final = feateng(trainX)
try_final = trainy.copy()
#%% Pipeline

num_feat = trX_final.select_dtypes('number').columns
cat_feat = trX_final.select_dtypes('category').columns

num_pipe = Pipeline([
    ('select', et.ColumnSelect(num_feat))
    ])

cat_pipe = Pipeline([
    ('select', et.ColumnSelect(cat_feat)),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

all_pipe = FeatureUnion([
    ('numeric', num_pipe),
    ('category', cat_pipe)
    ])


#%% Gradient Boost 

'''
This section takes a long time to run so the pickled objects are loaded instead
Load gb_inner, gb_outer, gb_inner_fit
'''
gb_outer = pickle.load(open(HOME_DIR + PICKLE_DIR + '/gb_outer.obj', 'rb'))
gb_inner = pickle.load(open(HOME_DIR + PICKLE_DIR + '/gb_inner_fit.obj', 'rb'))

'''

# tuning parameters
gb_params = {'learning_rate': [0.15,0.1],
          'n_estimators':  [100,300,500],
          }

# Inner fold to find the best hyperparameters
inner_cv = KFold(n_splits=5, shuffle = True, random_state=42)

# Outer fold to calculate the generalization error
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)



gb_inner = GridSearchCV(estimator = GradientBoostingClassifier(), 
                            param_grid = gb_params, 
                            scoring='roc_auc', 
                            refit=True, 
                            cv=inner_cv)

gb_outer = cross_val_score(estimator=gb_inner, 
                        X=all_pipe.fit_transform(trX_final), 
                        y=try_final, 
                        cv=outer_cv)


gb_inner.fit(X=all_pipe.fit_transform(trX_final), 
             y=try_final)

'''

# 0.6590126418    202/1328
gb_outer_auc = np.mean(gb_outer)

# generate predicted probabilities
gb_y_hat = gb_inner.predict_proba(X = all_pipe.transform(feateng(testX)))[:,1]

# 0.66075    
roc_auc_score(testy, gb_y_hat)



#%% Calculate feature importance using shuffling

num_cols = list(num_feat.copy())
cat_cols = list(cat_pipe.steps[1][1].get_feature_names(cat_feat))
all_cols = num_cols + cat_cols

# verify the lengths of the columns match
assert(len(all_cols) == all_pipe.transform(trX_final).shape[1])


perm_imp = permutation_importance(estimator=gb_inner, 
                            X=all_pipe.transform(feateng(testX)).toarray(), 
                            y=testy, 
                            n_repeats=10, 
                            random_state=42)

feat_imp = pd.DataFrame({"features": all_cols,
                         "importance": perm_imp.importances_mean,
                         "std" : perm_imp.importances_std}).sort_values \
                        ("importance",ascending=False)
                        
#%% Function to cacluate probability that the model's results occured by chance

def target_shuffling(estimator, trainX, trainY, n_iters, scorefunc, 
                     random_state=0, verbose = False):
    '''
    Model agnostic tehcnique invented by John Elder of Elder Research.  The
    results show the probability that the model's results occured by chance
    (p-value)
    
    For n_iters:
        
        1. Shuffle the target 
        2. Fit unshuffled input to shuffled target using estimator 
        3. Make predictions using unshuffled inputs
        4. Score predictions against shuffled target using scoring function
        5. Store and return predictions 
    
    The distribtuion of scores can be used to plot a histogram in order to 
    determine p-value
        
    Parameters
    ----------
    estimator: object
        A final model estimator that will be evaluated

    trainX : {array-like, sparse matrix} of shape (n_samples, n_features)
        The training input samples.
  
    trainY : array-like of shape (n_samples,)
        The training target
            
    n_iters : int
        The number of times to shuffle, refit and score model.
        
    scorefun : function
        The scoring function. For example mean_squared_error() 
        
    random_state : int, default = None
        Controls the randomness of the target shuffling. Set for repeatable
        results  
    
    verbose : boolean, default = False
        The scoring function. For example sklearn.metrics.mean_squared_error() 
        
    Returns
    -------
    scores : array of shape (n_iters)
        These are scores calculated for each shuffle

    '''
 
    for i in range(n_iters):    
        
        # 1. Shuffle the training target 
        np.random.default_rng(seed = random_state).shuffle(trainY)
        random_state += 1

        # 2. Fit unshuffled input to shuffled target using estimator
        estimator.fit(trainX, trainY)
        
        # calculate feature importance using permutation
        
        # 3. Make predictions using unshuffled inputs
        y_hat = estimator.predict_proba(trainX)[:,1]
        
        # 4. Score predictions against shuffled target using scoring function
        score = scorefunc(trainY, y_hat)
        
        # 5. Store and return predictions 
        if i == 0:
            allscores = np.array(score)
        else:
            allscores = np.append(allscores, score)
        
        if verbose:
            print("Shuffle: " + str(i+1) + "\t\tScore: " + str(score))
    
    return allscores


#%% Execute target shuffling

'''
This section takes a long time to run so the pickled objects are loaded instead
Load gb_scores
'''
gb_scores = pickle.load(open(HOME_DIR + PICKLE_DIR + '/gb_scores.obj', 'rb'))


'''
gb_inner.best_params_['random_state'] = 42 

gb_estimator = GradientBoostingClassifier(**gb_inner.best_params_)


gb_scores = target_shuffling(estimator = gb_estimator,
                  trainX = all_pipe.fit_transform(trX_final), 
                  trainY = np.array(try_final), 
                  n_iters=200, 
                  random_state=0,
                  verbose=True,
                  scorefunc=roc_auc_score)
'''

#%% Plot feature importance results

top5_feat = feat_imp.iloc[0:5,0:2].sort_values("importance")

fig1 = px.bar(top5_feat, 
              x="importance",
              y="features", 
              title = "Top 5 Features",
              orientation="h",
              labels=dict(features=""))

plot(fig1)
#fig1.write_image(HOME_DIR + IMG_DIR + "/fig1.png")



#%% Plot target shuffling result

fig2 = px.histogram(pd.DataFrame(gb_scores, columns=["AUC"]), x = "AUC")

fig2.add_vline(x=gb_outer_auc, line_dash = "dash")
fig2.update_layout(xaxis_range=[0.58,0.68], yaxis_range=[0,42.5])


fig2.add_annotation(x=gb_outer_auc, y=38.5,
            text="Best Model",
            showarrow=False,
            yshift=5,
            xshift=-50,
            font=dict(
                size=16
                ))

fig2.add_annotation(x=0.611, y=38.5,
            text="Shuffled Models",
            showarrow=False,
            yshift=5,
            font=dict(
                size=16
                ))

plot(fig2)
#fig2.write_image(HOME_DIR + IMG_DIR + "/fig2.png")
