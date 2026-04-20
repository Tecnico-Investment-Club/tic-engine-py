import pandas as pd
import numpy as np
from sklearn.model_selection._split import _BaseKFold


def getTrainTimes(t1,testTimes):
    """
    Given testTimes, find the times of the training observations.
    Purge from the training set all observations whose labels overlapped in time with those labels included in the testing set
    :params t1: event timestamps
        —t1.index: Time when the observation started.
        —t1.value: Time when the observation ended.
    :params testTimes: pd.series, Times of testing observations.
    :return trn: pd.df, purged training set
    """
    # copy t1 to trn
    trn = t1.copy(deep = True)
    # for every times of testing obervation
    for i, j in testTimes.items():
        # cond 1: train starts within test
        df0 = trn[(trn.index >= i) & (trn.index <= j)].index
        # cond 2: train ends within test
        df1 = trn[(i <= trn) & (trn <= j)].index 
        # cond 3: train envelops test
        df2 = trn[(trn.index <= i) & (j <= trn)].index 
        # drop the data that satisfy cond 1 & 2 & 3
        trn = trn.drop(df0.union(df1).union(df2))
    return trn


def getEmbargoTimes_book(times,pctEmbargo):
    #This function corresponds to the book implementation We currently do not use it.
    # Get embargo time for each bar
    step=int(times.shape[0]*pctEmbargo)
    if step==0:
        mbrg=pd.Series(times,index=times)
    else:
        mbrg=pd.Series(data=times[step:],index=times[:-step])
        final_embargo = pd.Series(data=times[-1], index=times[-step:])
        mbrg = pd.concat([mbrg, final_embargo])    
    return mbrg

def getEmbargoTimes(train_times, pctEmbargo):
    """
    Apply an embargo on the training dataset by removing a percentage of the first observations.

    Parameters:
    - train_times (pd.Series): Training dataset.
        - Index: Start time of each training observation.
        - Values: End time of each training observation.
    - pctEmbargo (float): Percentage of the training dataset to embargo.

    Returns:
    - pd.Series: Filtered training dataset with embargoed observations removed.
    """
    # Calculate the embargo size (number of observations to remove)
    n_embargo = int(len(train_times) * pctEmbargo)
    
    # Remove the first `n_embargo` observations from the training dataset
    embargoed_train = train_times.iloc[n_embargo:]
    
    return embargoed_train


class PurgedKFold(_BaseKFold):
    """
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in between
    """
    def __init__(self, n_splits=3, t1=None, pctEmbargo=0.):
        if not isinstance(t1, pd.Series):
            # if t1 is not a pd.Series, raise error
            raise ValueError('Label Through Dates must be a pd.Series')
        # inherit _BaseKFold, no shuffle
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1  # specify the vertical barrier
        self.pctEmbargo = pctEmbargo  # specify the embargo parameter (% of the bars)

    def split(self, X, y=None, groups=None):
        """
        :param X: the regressors, features
        :param y: the regressands, labels
        :param groups: None

        : return
            + train_indices: generator, the indices of training dataset 
            + test_indices: generator, the indices of the testing dataset
        """
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError('X and t1 must have the same index')

        # Create an array from 0 to (X.shape[0]-1)
        indices = np.arange(X.shape[0])

        # Size of the embargo
        mbrg = int(X.shape[0] * self.pctEmbargo)

        # Split the data into n_splits equal parts
        fold_size = X.shape[0] // self.n_splits
        remainder = X.shape[0] % self.n_splits

        # Loop to generate train-test splits for each fold
        start = 0
        for i in range(self.n_splits):  # For each fold
            # End of the current fold
            end = start + fold_size + (1 if i < remainder else 0)  # Distribute the remainder over the first folds
            test_indices = indices[start:end]
            train_indices = np.concatenate((indices[:start], indices[end:]))

            # Apply embargo to the training indices: training data cannot include data after the test data + embargo period
            maxT1Idx = self.t1.index.searchsorted(self.t1.iloc[test_indices].max())  # Get the index of the last test data point
            train_indices = np.concatenate((indices[:maxT1Idx], indices[maxT1Idx + mbrg:]))  # Apply embargo

            # Yield the train-test split for the current fold
            yield train_indices, test_indices

            # Update the start position for the next fold
            start = end



def cvScore(clf, X, y, sample_weight, scoring = 'neg_log_loss', t1 = None, cv = None, cvGen = None, pctEmbargo = 0):
    """
    Address two sklearn bugs
    1) Scoring functions do not know classes_
    2) cross_val_score will give different results because it weights to the fit method, but not to the log_loss method
    
    :params pctEmbargo: float, % of the bars will be embargoed
    """
    if scoring not in ['neg_log_loss', 'accuracy']:
        # if not using 'neg_log_loss' or 'accuracy' to score, raise error
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss,accuracy_score # import log_loss and accuracy_score
    #   from clfSequential import PurgedKFold # the original code assume they are stored in different folder
    if cvGen is None: # if there is no predetermined splits of the test sets and the training sets
        # use the PurgedKFold to generate splits of the test sets and the training sets
        cvGen = PurgedKFold(n_splits = cv,t1 = t1,pctEmbargo = pctEmbargo) # purged
    score = [] # store the CV scores
    # for each fold
    for train,test in cvGen.split(X = X):
        # fit the model
        fit = clf.fit(X = X.iloc[train, : ], y = y.iloc[train], sample_weight = sample_weight.iloc[train].values)
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X.iloc[test, : ]) # predict the probabily
            # neg log loss to evaluate the score
            score_ = -1 * log_loss(y.iloc[test], prob, sample_weight = sample_weight.iloc[test].values, labels = clf.classes_)
        else:
            pred = fit.predict(X.iloc[test, : ]) # predict the label
            # predict the accuracy score
            score_ = accuracy_score(y.iloc[test], pred, sample_weight = sample_weight.iloc[test].values)
        score.append(score_)
    return np.array(score)