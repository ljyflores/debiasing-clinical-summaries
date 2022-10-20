import pandas as pd

race_keywords = ['AFRICAN-AMERICAN',
                'AFRICAN AMERICAN',
                'AFRICAN',
                'BLACK',
                'CREOLE',
                'CAUCASIAN',
                'WHITE']

def remove_keys(s, keywords):
    for key in keywords:
        if key in s:
            s = s.replace(key,'')
    return s

def clean_string(s):
    s = s.replace(',',' ')
    s = s.replace('-',' ')
    s = s.replace('\n',' ')
    return s

# Read data and drop nulls
df_adm = pd.read_csv('data/ADMISSIONS.csv')
df_nte = pd.read_csv('data/NOTEEVENTS.csv')
df_sev = pd.read_csv('data/apsiii-score.csv')

df_nte = df_nte.loc[~df_nte.HADM_ID.isnull()].reset_index(drop=True)
df_nte['HADM_ID'] = df_nte['HADM_ID'].apply(int)

df_adm = df_adm[['SUBJECT_ID','HADM_ID','ETHNICITY','DIAGNOSIS']]

# Filter only to nursing notes
df_nte = df_nte[['SUBJECT_ID','HADM_ID','CHARTDATE','CATEGORY','TEXT']]
df_nte = df_nte.loc[df_nte.CATEGORY.str.contains('Nursing')]
df_nte = df_nte.drop_duplicates().reset_index(drop=True)
        
df_sev = df_sev[['subject_id','hadm_id','apsiii']].drop_duplicates()

df = df_adm.merge(df_nte, on=['SUBJECT_ID','HADM_ID'])
df = df.merge(df_sev, left_on=['SUBJECT_ID','HADM_ID'], right_on=['subject_id','hadm_id'])

# Filter to only white/black patients
df = df.loc[df.ETHNICITY.str.contains('WHITE')|df.ETHNICITY.str.contains('BLACK')]
df['ETHNICITY'] = df['ETHNICITY'].apply(lambda x: 'WHITE' if 'WHITE' in x else 'BLACK')

# Removing race-related keywords from text
df['TEXT'] = df['TEXT'].apply(lambda x: x.upper())
df['TEXT'] = df['TEXT'].apply(lambda x: remove_keys(x, race_keywords))

# Clean punctuations
df['TEXT'] = df['TEXT'].apply(clean_string)

df = df.drop(['subject_id','hadm_id'], axis=1)
df = df.dropna(how='any', axis=0)

df.to_csv('data/preprocessed.csv')