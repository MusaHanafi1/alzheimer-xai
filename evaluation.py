"""Evaluation functions with explicit uncertainty and small-sample reporting."""
import numpy as np
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,
                            average_precision_score,brier_score_loss,log_loss,confusion_matrix)


def metrics(y,probability,threshold=0.5):
    y=np.asarray(y);p=np.asarray(probability);pred=(p>=threshold).astype(int)
    both=len(np.unique(y))==2
    return {'roc_auc':float(roc_auc_score(y,p)) if both else None,
        'average_precision':float(average_precision_score(y,p)) if both else None,
        'accuracy':float(accuracy_score(y,pred)), 'precision':float(precision_score(y,pred,zero_division=0)),
        'recall':float(recall_score(y,pred,zero_division=0)), 'f1':float(f1_score(y,pred,zero_division=0)),
        'brier':float(brier_score_loss(y,p)), 'log_loss':float(log_loss(y,p,labels=[0,1])),
        'threshold':float(threshold), 'n':int(len(y)), 'positive_n':int(y.sum()),
        'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}


def bootstrap_intervals(y,p,threshold=0.5,repeats=1000,seed=42):
    """Stratified percentile bootstrap conditional on this test set and trained model."""
    y=np.asarray(y);p=np.asarray(p);rng=np.random.default_rng(seed)
    keys=['roc_auc','average_precision','accuracy','precision','recall','f1','brier']
    values={k:[] for k in keys}
    groups=[np.flatnonzero(y==c) for c in [0,1]]
    if any(len(x)==0 for x in groups):raise ValueError('Both classes required for bootstrap')
    for _ in range(repeats):
        idx=np.concatenate([rng.choice(g,size=len(g),replace=True) for g in groups])
        m=metrics(y[idx],p[idx],threshold)
        for k in keys:values[k].append(m[k])
    return {k:{'lower':float(np.quantile(v,.025)),'upper':float(np.quantile(v,.975))} for k,v in values.items()}


def subgroup_metrics(cohort,y,p,threshold):
    rows=[]
    subgroup_defs={'M/F':cohort['M/F'], 'Age band':np.where(cohort['Age']<75,'Below 75','75 and above')}
    for feature,values in subgroup_defs.items():
        for level in sorted(set(values)):
            mask=np.asarray(values)==level
            row=metrics(np.asarray(y)[mask],np.asarray(p)[mask],threshold)
            row.update({'feature':feature,'group':str(level),'interpretation':'Descriptive only; small test subgroups do not establish fairness'})
            rows.append(row)
    return rows
