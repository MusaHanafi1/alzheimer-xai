"""Run the benchmark from a declared CSV and emit an auditable run directory."""
import argparse,hashlib,json,platform,sys,subprocess
from pathlib import Path
import importlib.metadata as md
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV,cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from dementia_ml.data import DEMO,FEATURES,load_cohort
from dementia_ml.modeling import candidates,subject_split,cv_splitter,select_threshold
from dementia_ml.evaluation import metrics,bootstrap_intervals,subgroup_metrics


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cohort,X,y,meta=load_cohort(args.data,args.data_kind,args.source)
    train_idx,test_idx=subject_split(cohort,y,args.seed)
    Xtrain,Xtest=X.iloc[train_idx],X.iloc[test_idx]
    ytrain,ytest=y.iloc[train_idx],y.iloc[test_idx]
    if ytrain.value_counts().min()<args.folds:raise ValueError('Insufficient subjects per class for chosen CV folds')
    cv=cv_splitter(args.seed,args.folds);models={};selection=[]
    for name,(estimator,grid) in candidates(args.seed,not args.no_xgboost).items():
        search=GridSearchCV(estimator,grid,scoring={'roc_auc':'roc_auc','average_precision':'average_precision','brier':'neg_brier_score'},
            refit='roc_auc',cv=cv,n_jobs=1,error_score='raise',return_train_score=False)
        search.fit(Xtrain,ytrain);models[name]=search.best_estimator_
        selection.append({'model':name,'selection_cv_roc_auc':float(search.best_score_),
            'selection_cv_roc_auc_std':float(search.cv_results_['std_test_roc_auc'][search.best_index_]),
            'parameters':search.best_params_})
        print(f'{name}: selection CV ROC-AUC={search.best_score_:.4f}',flush=True)
    eligible=[r for r in selection if r['model']!='prevalence_baseline']
    winner=max(eligible,key=lambda r:r['selection_cv_roc_auc'])['model']
    base=clone(models[winner])
    # Calibrate the complete pipeline inside each training fold. No holdout access.
    calibrated=CalibratedClassifierCV(estimator=base,method='sigmoid',cv=cv_splitter(args.seed+1,3),ensemble=True,n_jobs=1)
    oof=cross_val_predict(calibrated,Xtrain,ytrain,cv=cv,method='predict_proba',n_jobs=1)[:,1]
    chosen,threshold_curve=select_threshold(ytrain,oof,args.target_recall)
    calibrated.fit(Xtrain,ytrain)
    p=calibrated.predict_proba(Xtest)[:,1];threshold=chosen['threshold']
    # Raw candidate holdout comparisons are descriptive; never change winner from these scores.
    comparison=[]
    for name,model in models.items():
        row=metrics(ytest,model.predict_proba(Xtest)[:,1]);row['model']=name;comparison.append(row)
    final_metrics=metrics(ytest,p,threshold)
    # Save input hashes + memberships; IDs are SHA-256 hashes, never model features.
    def hash_id(x):return hashlib.sha256(str(x).encode()).hexdigest()
    splits={'development':[hash_id(i) for i in cohort.iloc[train_idx]['Subject ID']],
            'test':[hash_id(i) for i in cohort.iloc[test_idx]['Subject ID']]}
    provenance={'seed':args.seed,'selection_folds':args.folds,'development_subjects':len(train_idx),'test_subjects':len(test_idx),
        'subject_overlap':0,'dataset':meta,'feature_columns':FEATURES,'excluded_predictors':['CDR','Group','Subject ID','MRI ID','Visit','MR Delay','Hand'],
        'selection_rule':'Highest development CV ROC-AUC among non-dummy candidates; grid search on development data only',
        'calibration':'Prespecified sigmoid calibration, 3 folds, entire preprocessing pipeline cloned within each fold',
        'threshold_rule':f'Maximize precision among training OOF thresholds with recall >= {args.target_recall}',
        'threshold_selection':chosen,'threshold_selection_caveat':'OOF predictions reuse hyperparameters selected on development data. Training estimates are not unbiased performance estimates.',
        'holdout_policy':'Seeded holdout excluded from model selection, calibration and threshold choice; single final run, not a permanently sealed future benchmark',
        'interval_method':f'{args.bootstrap} stratified bootstrap replicates; percentile 95% intervals, conditional on fixed model and observed test class counts; not retraining uncertainty',
        'python':platform.python_version(),'dependencies':{k:md.version(k) for k in ['numpy','pandas','scikit-learn','scipy','matplotlib','joblib']}}
    if not args.no_xgboost:provenance['dependencies']['xgboost']=md.version('xgboost')
    report={'schema_version':2,'label':'SYNTHETIC DATA DEMONSTRATION' if args.data_kind=='synthetic' else 'RESEARCH DATA EVALUATION',
            'provenance':provenance,'selected_model':winner,'selection_cv':selection,'raw_model_holdout_comparison':comparison,
            'final_calibrated_model':final_metrics,'final_at_0_5':metrics(ytest,p,.5),
            'confidence_intervals':bootstrap_intervals(ytest,p,threshold,args.bootstrap,args.seed),
            'subgroups':subgroup_metrics(cohort.iloc[test_idx].reset_index(drop=True),ytest.to_numpy(),p,threshold),
            'limitations':['Synthetic demo scores do not establish real-world or clinical performance.',
                'One latest observed visit per subject; contemporaneous CDR status, not future dementia risk or Alzheimer pathology.',
                'Small test set and subgroups; results are sensitive to the split.',
                'MMSE and CDR measure related aspects of cognition; performance may reflect construct overlap.',
                'No external validation or clinical-use claim.','Selection CV scores are used for selection, not reported as unbiased final estimates.']}
    if args.data_kind=='real':report['limitations'][0]='Clinical utility requires independent external validation; this is research evaluation.'
    (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (out/'split_manifest.json').write_text(json.dumps(splits,indent=2))
    joblib.dump({'model':calibrated,'threshold':threshold,'features':FEATURES,'data_kind':args.data_kind,'dataset_sha256':meta['sha256']},out/'model.joblib')
    # Training-only permutation importance avoids using holdout for feature decisions.
    imp=permutation_importance(calibrated,Xtrain,ytrain,scoring='roc_auc',n_repeats=10,random_state=args.seed,n_jobs=1)
    pd.DataFrame({'feature':FEATURES,'mean_decrease_auc':imp.importances_mean,'std':imp.importances_std}).sort_values('mean_decrease_auc',ascending=False).to_csv(out/'feature_importance.csv',index=False)
    if args.data_kind=='synthetic':
        pd.DataFrame({'actual':ytest.to_numpy(),'probability':p,'age':cohort.iloc[test_idx]['Age'].to_numpy(),'sex':cohort.iloc[test_idx]['M/F'].to_numpy()}).to_csv(out/'synthetic_holdout_predictions.csv',index=False)
        # Local explanation: SHAP explains the selected uncalibrated pipeline, not calibrated probabilities.
        if args.shap:
            import shap
            fitted=models[winner]
            transformed=fitted[:-1].transform(Xtrain)
            example=fitted[:-1].transform(Xtest.iloc[:1])
            explainer=shap.Explainer(fitted[-1].predict_proba,shap.maskers.Independent(transformed),algorithm='permutation',feature_names=FEATURES,seed=args.seed)
            explanation=explainer(example,max_evals=2*len(FEATURES)+1)
            (out/'shap_example.json').write_text(json.dumps({'explained_model':winner+' (uncalibrated)', 'explained_output':'Positive-class probability, not the calibrated final probability',
                'base_probability':float(explanation.base_values[0,1]),'positive_probability':float(fitted.predict_proba(Xtest.iloc[:1])[0,1]),
                'contributions':dict(zip(FEATURES,explanation.values[0,:,1].astype(float).tolist()))},indent=2))
    from dementia_ml.reporting import build_report
    build_report(out)
    print(json.dumps({'selected_model':winner,'test_subjects':len(test_idx),'final_calibrated':final_metrics,'output':str(out)},indent=2),flush=True)
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=DEMO);p.add_argument('--data-kind',choices=['synthetic','real'],default='synthetic')
    p.add_argument('--source',help='Dataset provenance/citation; required for real data')
    p.add_argument('--output',default='reports/demo');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--folds',type=int,default=5);p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--target-recall',type=float,default=.8);p.add_argument('--no-xgboost',action='store_true')
    p.add_argument('--shap',action='store_true',help='Generate one synthetic example explanation (requires optional shap)')
    return p

if __name__=='__main__':
    args=parser().parse_args()
    if args.folds<2 or args.bootstrap<100:raise SystemExit('Use folds >= 2 and bootstrap >= 100.')
    run(args)
