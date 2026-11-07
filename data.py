"""Deterministic cohort construction. No learned preprocessing belongs here."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

FEATURES = ['Age','EDUC','SES','MMSE','eTIV','nWBV','ASF','Gender_Encoded']
NUMERIC = ['Age','EDUC','SES','MMSE','eTIV','nWBV','ASF']
ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT/'data/demo/oasis_synthetic.csv'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_cohort(raw):
    """Latest observed visit per subject; contemporaneous CDR>0, not prognosis."""
    required = ['Subject ID','Visit','CDR','M/F'] + NUMERIC
    absent = sorted(set(required) - set(raw.columns))
    if absent:
        raise ValueError(f'Missing required columns: {absent}')
    df = raw.copy()
    if df['Subject ID'].isna().any() or df['Subject ID'].astype(str).str.strip().eq('').any():
        raise ValueError('Every record needs a nonempty Subject ID.')
    df['Subject ID'] = df['Subject ID'].astype(str)
    for col in ['Visit','CDR'] + NUMERIC:
        df[col] = pd.to_numeric(df[col], errors='raise')
        if np.isinf(df[col].to_numpy(dtype=float)).any():
            raise ValueError(f'Infinite values in {col}')
    if df[['Visit','CDR']].isna().any().any():
        raise ValueError('Visit and CDR must be observed; never impute the outcome.')
    if (df['Visit'] < 0).any() or not df['CDR'].isin([0,0.5,1,2,3]).all():
        raise ValueError('Unexpected visit index or CDR label.')
    df = df.drop_duplicates()
    if df.duplicated(['Subject ID','Visit']).any():
        raise ValueError('Conflicting records for the same subject and visit.')
    if not df['M/F'].isin(['M','F']).all():
        raise ValueError('M/F must contain M or F for this dataset schema.')
    df = df.sort_values(['Subject ID','Visit'], kind='stable').groupby('Subject ID', sort=True).tail(1)
    df = df.sort_values('Subject ID').reset_index(drop=True)
    df['Gender_Encoded'] = df['M/F'].map({'M':0,'F':1}).astype(float)
    y = (df['CDR'] > 0).astype(int)
    if y.nunique() != 2:
        raise ValueError('Both outcome classes are required.')
    return df, df[FEATURES].astype(float), y


def load_cohort(path=DEMO, data_kind='synthetic', source=None):
    path = Path(path)
    if data_kind not in ['synthetic','real']:
        raise ValueError('data_kind must be synthetic or real')
    if data_kind == 'real' and digest(path) == digest(DEMO):
        raise ValueError('The bundled synthetic file cannot be labeled real.')
    if data_kind == 'real' and not source:
        raise ValueError('Real-data runs require a source citation with --source.')
    raw = pd.read_csv(path)
    cohort, X, y = make_cohort(raw)
    meta = {'data_kind':data_kind, 'sha256':digest(path), 'source':source or 'Synthetic CSV supplied in MusaHanafi1/alzheimer-xai archive; generation code unavailable',
            'raw_records':len(raw), 'subjects':len(cohort), 'excluded_visits':len(raw)-len(cohort),
            'positive_subjects':int(y.sum()), 'negative_subjects':int((1-y).sum()),
            'selection':'Latest observed visit per subject', 'target':'CDR > 0 at the same visit',
            'missing_predictors':{k:int(v) for k,v in X.isna().sum().items()}}
    return cohort, X, y, meta
