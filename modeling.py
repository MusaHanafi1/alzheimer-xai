"""Fold-local preprocessing, training-only model selection and threshold choice."""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score


def subject_split(cohort, y, seed=42, fraction=0.2):
    train, test = train_test_split(np.arange(len(cohort)), test_size=fraction, random_state=seed, stratify=y)
    if set(cohort.iloc[train]['Subject ID']) & set(cohort.iloc[test]['Subject ID']):
        raise AssertionError('Subject overlap between development and test data')
    return train, test


def candidates(seed=42, use_xgboost=True):
    def pipe(model):
        return Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
                         ('scale',StandardScaler()),('model',model)])
    options = {
      'prevalence_baseline':(pipe(DummyClassifier(strategy='prior')),{}),
      'logistic_regression':(pipe(LogisticRegression(class_weight='balanced',max_iter=2000,random_state=seed)),{'model__C':[0.1,1.0,10.0]}),
      'random_forest':(pipe(RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=seed,n_jobs=1)),{'model__max_depth':[3,None],'model__min_samples_leaf':[3,6]})}
    if use_xgboost:
        from xgboost import XGBClassifier
        options['xgboost']=(pipe(XGBClassifier(n_estimators=120,learning_rate=0.05,eval_metric='logloss',random_state=seed,n_jobs=1,tree_method='hist')),
            {'model__max_depth':[2,3],'model__reg_lambda':[1.0,5.0]})
    return options


def cv_splitter(seed=42, n_splits=5):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def select_threshold(y, probability, target_recall=0.8):
    """Training OOF predictions only. Max precision subject to recall >= target."""
    if not 0 < target_recall <= 1:
        raise ValueError('target_recall must be in (0,1]')
    rows=[]
    for threshold in np.unique(np.r_[0.0,np.linspace(0.01,0.99,99)]):
        pred=(np.asarray(probability)>=threshold).astype(int)
        precision=precision_score(y,pred,zero_division=0)
        recall=recall_score(y,pred,zero_division=0)
        rows.append({'threshold':float(threshold),'precision':float(precision),'recall':float(recall)})
    eligible=[r for r in rows if r['recall'] >= target_recall]
    selected=max(eligible,key=lambda r:(r['precision'],r['threshold']))
    return selected,rows
