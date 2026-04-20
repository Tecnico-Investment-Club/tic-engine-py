import numpy as np
from functions.Chapter_7 import PurgedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from scipy.stats import rv_continuous,kstest
import pandas as pd
import matplotlib as mpl


class MyPipeline(Pipeline):
    """
    Inherit all methods from sklearn's `Pipeline`
    Overwrite the inherited `fit` method with a new one that handles the argument `sample weight`
    After which it redirects to the parent class
    """
    def fit(self, X, y, sample_weight = None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + '__sample_weight'] = sample_weight
        return super(MyPipeline, self).fit(X, y, **fit_params)
    

def clfHyperFit(feat, lbl, t1, pipe_clf, param_grid,
                cv = 3, bagging = [0, None, 1.], n_jobs = -1, pctEmbargo = 0, **fit_params):
    """
    Grid Search with purged K-Fold Cross Validation
    :params feat: features
    :params lbl: labels
    :params t1: vertical barriers
    :params pipe_clf: classification pipeline
    :params param_grid: parameter grid
    :params cv: int, cross validation fold
    :params bagging: bagging parameter?
    :params n_jobs: CPUs
    :params pctEmbargo: float, % of embargo
    :params **fit_params: 
    :return gs:
    """
   
    if set(lbl.values) == {0, 1}: # if label values are 0 or 1
        scoring = 'f1' # f1 for meta-labeling
    else:
        scoring = 'neg_log_loss' # symmetric towards all cases
    #1) hyperparameter search, on train data
    # prepare the training sets and the validation sets for CV (find their indices)
    inner_cv = PurgedKFold(n_splits = cv, t1 = t1, pctEmbargo = pctEmbargo) # purged
    # perform grid search
    gs = GridSearchCV(estimator = pipe_clf, param_grid = param_grid, 
                        scoring = scoring, cv = inner_cv, n_jobs = n_jobs)
    # best estimator and the best parameter
    gs = gs.fit(feat, lbl, **fit_params).best_estimator_ # this is a pipeline
 # 2) fit validated model on the entirety of the data
    if bagging[1] > 0:  # max_samples > 0
        final_estimator = gs.named_steps['clf']  # Get the final estimator. The estimator step of the pipeline
        bagging_clf = BaggingClassifier(base_estimator=final_estimator,
                                        n_estimators=int(bagging[0]), max_samples=float(bagging[1]),
                                        max_features=float(bagging[2]), n_jobs=n_jobs)
        # Pass the sample_weight directly to the fit method of the final estimator
        bagging_clf = bagging_clf.fit(feat, lbl, sample_weight=fit_params.get('sample_weight', None))
        gs = Pipeline([('bag', bagging_clf)])
    return gs

def clfHyperFitRand(feat, lbl, t1, cv, pipe_clf, param_grid, bagging, rndSearchIter=0, pctEmbargo=0, n_jobs=-1, **fit_params):
    """
    Randomized (or Grid) Search with purged K-Fold Cross Validation
    :params feat: features
    :params lbl: labels
    :params t1: vertical barriers
    :params pipe_clf: classification pipeline
    :params param_grid: parameter grid
    :params bagging: bagging parameter?
    :params rndSearchIter: int, 0 for normal search, > 0 for randomized search
    :params pctEmbargo: float, % of embargo
    :params n_jobs: CPUs
    :params **fit_params:
    :return gs:
    """
    if set(lbl.values) == {0, 1}:  # if label values are 0 or 1
        scoring = 'f1'  # f1 for meta-labeling
    else:
        scoring = 'neg_log_loss'  # symmetric towards all cases

    # 1) hyperparameter search, on train data
    inner_cv = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)
    
    if rndSearchIter > 0:
        gs = RandomizedSearchCV(estimator=pipe_clf, param_distributions=param_grid,
                                 scoring=scoring, cv=inner_cv, n_jobs=n_jobs, n_iter=rndSearchIter)
    else:
        gs = GridSearchCV(estimator=pipe_clf, param_grid=param_grid,
                          scoring=scoring, cv=inner_cv, n_jobs=n_jobs)

    gs = gs.fit(feat, lbl, **fit_params).best_estimator_  # this is a pipeline

    # 2) fit validated model on the entirety of the data
    if bagging[1] > 0:  # max_samples > 0
        final_estimator = gs.named_steps['clf'] # Get the final estimator. The estimator step of the pipeline
        bagging_clf = BaggingClassifier(base_estimator=final_estimator,
                                        n_estimators=int(bagging[0]), max_samples=float(bagging[1]),
                                        max_features=float(bagging[2]), n_jobs=n_jobs)
        bagging_clf = bagging_clf.fit(feat, lbl, sample_weight=fit_params.get('sample_weight', None))
        gs = Pipeline([('bag', bagging_clf)])
    
    return gs

#———————————————————————————————————————
class logUniform_gen(rv_continuous):
    # random numbers log-uniformly distributed between 1 and e
    def _cdf(self,x):
        return np.log(x/self.a)/np.log(self.b/self.a)
    
def logUniform(a=1,b=np.exp(1)):
    return logUniform_gen(a=a,b=b,name='logUniform')
