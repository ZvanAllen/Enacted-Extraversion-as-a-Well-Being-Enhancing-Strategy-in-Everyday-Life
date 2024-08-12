import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg

# Study 1

# Load data
data = pd.read_csv('WB1 Data - Final - Reduced.csv')
data['Condition'] = data['Condition'].astype('category')

# Select cases where 3+ daily logs completed
data1 = data[data['Activities_Completed'] >= 3]

# Create pre/post dataset with complete cases only
data_prepost = data1[['ID', 'Condition', 'T1PA', 'T7PA'] + 
                     [col for col in data1.columns if col.startswith('Daily_')]]
data1_prepost = data_prepost.dropna()

# Create long format dataset
data1_long = pd.melt(data1[['ID', 'Condition', 'T1PA', 'T7PA']], 
                     id_vars=['ID', 'Condition'],
                     value_vars=['T1PA', 'T7PA'],
                     var_name='time', value_name='score')
data1_long['time'] = data1_long['time'].map({'T1PA': 'Pretest', 'T7PA': 'Posttest'})

# Hypothesis 1: Daily positive affect (PA)
ttest_result = stats.ttest_ind(data1.loc[data1['Condition'] == 0, 'Daily_PA_Mean'],
                               data1.loc[data1['Condition'] == 1, 'Daily_PA_Mean'])
print("T-test result for daily PA:", ttest_result)

# Cohen's d
cohens_d = (data1.loc[data1['Condition'] == 1, 'Daily_PA_Mean'].mean() - 
            data1.loc[data1['Condition'] == 0, 'Daily_PA_Mean'].mean()) / \
           np.sqrt((data1.loc[data1['Condition'] == 1, 'Daily_PA_Mean'].var() + 
                    data1.loc[data1['Condition'] == 0, 'Daily_PA_Mean'].var()) / 2)
print("Cohen's d:", cohens_d)

# Hypothesis 2: PA over time
anova_result = pg.mixed_anova(data=data1_long, dv='score', between='Condition', 
                              within='time', subject='ID')
print("ANOVA result for PA over time:")
print(anova_result)

# Hypothesis 3: Regressions on Daily PA
model1 = ols('Daily_PA_Mean ~ C(Condition) + ExtCentered', data=data1).fit()
print(model1.summary())

model2 = ols('Daily_PA_Mean ~ C(Condition) + ExtCentered + Cond_x_Extraversion', data=data1).fit()
print(model2.summary())

# Table 1 (Means/SD)
print(data1_prepost.groupby('Condition').describe())

daily_vars = ['Daily_PA_Mean', 'Daily_NA_Mean', 'Daily_Fatigue_Mean', 'Daily_Serenity_Mean',
              'Daily_Authenticity_Mean', 'Daily_SVS_Mean', 'Daily_Effort_Mean']
print(data1[['Condition'] + daily_vars].groupby('Condition').describe())

# Table 2 (Mixed ANOVAs)
for var in ['PA', 'NA', 'Fatigue', 'Serenity']:
    data_long = pd.melt(data1_prepost[['ID', 'Condition', f'T1{var}', f'T7{var}']], 
                        id_vars=['ID', 'Condition'],
                        value_vars=[f'T1{var}', f'T7{var}'],
                        var_name='time', value_name='score')
    data_long['time'] = data_long['time'].map({f'T1{var}': 'Pretest', f'T7{var}': 'Posttest'})
    
    anova_result = pg.mixed_anova(data=data_long, dv='score', between='Condition', 
                                  within='time', subject='ID')
    print(f"ANOVA result for {var}:")
    print(anova_result)

# Study 2

# Load data
data = pd.read_csv('WB2 Data - Final - Reduced.csv')
data['Condition'] = data['Condition'].astype('category')

# Select cases where 3+ daily logs completed
data2 = data[data['LogsCompleted'] >= 3]

# Create pre/post dataset
data_prepost = data2[['ID', 'Condition', 'Exclude_All'] + 
                     [col for col in data2.columns if col.startswith('T1') or col.startswith('T7') or col.startswith('Daily_')]]
data2_prepost = data_prepost[data_prepost['Exclude_All'] == 0]

# Hypothesis 1: ANOVA for Daily PA
anova_model = ols('Daily_PA_Mean ~ C(Condition)', data=data2).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)
print("ANOVA for Daily PA:")
print(anova_table)

# Follow-up t-tests
conditions = [(0, 2), (0, 1), (1, 2)]
for c1, c2 in conditions:
    subset = data2[data2['Condition'].isin([c1, c2])]
    ttest_result = stats.ttest_ind(subset[subset['Condition'] == c1]['Daily_PA_Mean'],
                                   subset[subset['Condition'] == c2]['Daily_PA_Mean'])
    print(f"T-test for conditions {c1} vs {c2}:")
    print(ttest_result)

# Hypothesis 2: Mixed ANOVA for PA
data2_PA_long = pd.melt(data2_prepost[['ID', 'Condition', 'T1PA', 'T7PA']], 
                        id_vars=['ID', 'Condition'],
                        value_vars=['T1PA', 'T7PA'],
                        var_name='time', value_name='score')
data2_PA_long['time'] = data2_PA_long['time'].map({'T1PA': 'Pretest', 'T7PA': 'Posttest'})

anova_result = pg.mixed_anova(data=data2_PA_long, dv='score', between='Condition', 
                              within='time', subject='ID')
print("Mixed ANOVA for PA:")
print(anova_result)

# Follow-up paired t-tests
for condition in [0, 1, 2]:
    subset = data2_prepost[data2_prepost['Condition'] == condition]
    ttest_result = stats.ttest_rel(subset['T1PA'], subset['T7PA'])
    print(f"Paired t-test for condition {condition}:")
    print(ttest_result)

# Hypothesis 3: Regression
model = ols('Daily_PA_Mean ~ Dummy_Extraversion + Dummy_Introversion + ExtCentered', data=data2).fit()
print(model.summary())

# Study 3

# Load data
data3 = pd.read_csv('WB3 Data - Final - Reduced.csv')
data3['Condition'] = data3['Condition'].astype('category')
data3 = data3[data3['TOTALDAILIES'] >= 3]

# Create pre/post dataset
data_prepost = data3[['ID', 'Condition', 'Include'] + 
                     [col for col in data3.columns if col.startswith('T1') or col.startswith('T7')]]
data3_prepost = data_prepost[data_prepost['Include'] == 0]

# Hypothesis 1: ANOVA for Daily PA
anova_model = ols('Daily_PA_Mean ~ C(Condition)', data=data3).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)
print("ANOVA for Daily PA:")
print(anova_table)

# Follow-up t-tests
conditions = [(0, 2), (0, 1), (1, 2)]
for c1, c2 in conditions:
    subset = data3[data3['Condition'].isin([c1, c2])]
    ttest_result = stats.ttest_ind(subset[subset['Condition'] == c1]['Daily_PA_Mean'],
                                   subset[subset['Condition'] == c2]['Daily_PA_Mean'])
    print(f"T-test for conditions {c1} vs {c2}:")
    print(ttest_result)

# Hypothesis 2: Mixed ANOVA for PA
data3_PA_long = pd.melt(data3_prepost[['ID', 'Condition', 'T1PA', 'T7PA']], 
                        id_vars=['ID', 'Condition'],
                        value_vars=['T1PA', 'T7PA'],
                        var_name='time', value_name='score')
data3_PA_long['time'] = data3_PA_long['time'].map({'T1PA': 'Pretest', 'T7PA': 'Posttest'})

anova_result = pg.mixed_anova(data=data3_PA_long, dv='score', between='Condition', 
                              within='time', subject='ID')
print("Mixed ANOVA for PA:")
print(anova_result)

# Hypothesis 3: Regression
model = ols('Daily_PA_Mean ~ Dummy_NonSocial + Dummy_Social + ExtCentered', data=data3).fit()
print(model.summary())