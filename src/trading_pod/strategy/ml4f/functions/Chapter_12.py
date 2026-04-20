from functions.Chapter_7 import getTrainTimes, getEmbargoTimes
import itertools
import pandas as pd
import numpy as np
from functions.Chapter_9 import clfHyperFitRand
from functions.Chapter_3 import getEvents_zero, getBins_meta



def cpcv_backtest(new_feat, bins, T1, pd_closes_total, new_tEvents, ptSl_meta, trgt, minRet, t1_meta,
                  pipe_clf, param_grid, sample_weights, N=6, k=2, pctEmbargo=0.1, cv_splits=3, rndSearchIter=10):
    """
    CPCV backtesting algorithm that trains a first model and a meta labeler for each training/testing split.
    
    Parameters:
        new_feat         : DataFrame with features (indexed by bar beginning)
        bins             : DataFrame with label bins for the first model (column 'bin')
        T1               : Series or DataFrame with bar end times (used for purging/embargo)
        pd_closes_total  : DataFrame with closing prices (used to compute vertical barriers)
        new_tEvents      : Series/DataFrame with event times
        ptSl_meta        : Profit-taking/stop-loss parameters for meta events
        trgt             : Target value for vertical barrier calculations
        minRet           : Minimum return to consider an event
        t1_meta          : Series/DataFrame with t1 times for meta events (aligned with new_feat)
        pipe_clf         : A pipeline classifier (used by clfHyperFitRand)
        param_grid       : Parameter grid for hyperparameter tuning (used by clfHyperFitRand)
        sample_weights   : Sample weights (passed to clfHyperFitRand)
        N                : Number of groups to partition the data (default=5)
        k                : Number of groups to use as the test set in each split (default=1)
        pctEmbargo       : Percentage of embargo (default=0.1)
        cv_splits        : Number of CV splits used internally (default=3)
        rndSearchIter    : Number of random search iterations (default=10)
        
    Returns:
        A list with the data segmentation
        A list of dictionaries with the following keys for each split:
            'train_idx'    : indices used for training
            'test_idx'     : indices used for testing
            'model1_preds' : predictions from the first model on test set
            'meta_preds'   : predictions from the meta labeler on test set
            'meta_probs'   : probabilities in tuple format (p(0),p(1))
            'splits_id' : id of the partions of data in a model's test set
    """
    # Total number of observations
    T = len(new_feat)
    
    # Determine the size of each group (first N-1 groups have floor(T/N) observations)
    group_size = T // N
    groups = []
    for i in range(N):
        if i < N - 1:
            start = i * group_size
            end = (i + 1) * group_size
        else:
            # Last group takes the remainder
            start = i * group_size
            end = T
        groups.append((start, end))
    
    # segmented timespan (in int indices)
    path = []
    for i in groups:
        start,end = i
        path.append([n for n in range(start,end)])

    # Generate all possible training/testing splits:
    # Each split: select k groups for testing; remaining groups for training.
    splits = []
    splits_id = []
    ends_for_embargo = []
    for test_groups in itertools.combinations(range(N), k):
        train_groups = [g for g in range(N) if g not in test_groups]
        train_idx = []
        for g in train_groups:
            start, end = groups[g]
            train_idx.extend(range(start, end))
        test_idx = []
        ends = []
        ids = []
        for g in test_groups:
            start, end = groups[g]
            test_idx.extend(range(start, end))
            ends.append(end-1)
            ids.append(g)
        splits.append((train_idx, test_idx))
        ends_for_embargo.append(ends)
        splits_id.append(ids)

    results = []
  
    
    for i,(train_idx, test_idx) in enumerate(splits):
        # Subset the data for training and testing
        feat_train = new_feat.iloc[train_idx]
        feat_test = new_feat.iloc[test_idx]
        bins_train = bins.iloc[train_idx]
        bins_test = bins.iloc[test_idx]
        T1_train = T1.iloc[train_idx]
        T1_test = T1.iloc[test_idx]
        sample_weights_train = sample_weights.iloc[train_idx]

        #Apply Purging
        train_times = getTrainTimes(T1_train, T1_test)
        
        for ending in ends_for_embargo[i]:
            if (train_times.index > T1.iloc[ending]).any():
                train_times.loc[train_times.index > T1.iloc[ending]] = getEmbargoTimes(train_times[train_times.index > T1.iloc[ending]], pctEmbargo)
           

        train_idx = list(train_times.index)
        feat_train = feat_train.loc[train_idx]
        bins_train = bins_train.loc[train_idx]
        T1_train= T1_train.loc[train_idx]
        train_idx = pd.to_datetime(train_idx).tz_localize(None)
        sample_weights_train = sample_weights_train.loc[train_idx]

        train_idx = new_feat.index.get_indexer(train_times.index)
        train_idx = train_idx[train_idx >= 0]  # Remove any -1 (unmatched) indices


        # === First Model Training ===
        # Train model1 on training data using clfHyperFitRand.
        # Here, bins_train['bin'] are the labels for the first model.
        model1 = clfHyperFitRand(
            feat=feat_train,
            lbl=bins_train['bin'],
            t1=T1_train,
            cv=cv_splits,
            pipe_clf=pipe_clf,
            param_grid=param_grid,
            bagging=[0, 0, 0],
            rndSearchIter=rndSearchIter,
            pctEmbargo=pctEmbargo,
            n_jobs=-1,
            clf__sample_weight=sample_weights_train
        )
        
        # Generate predictions (sides) from the first model for the training set.
        sides_train = model1.predict(feat_train)
        sides_train = pd.Series(sides_train, index=feat_train.index)
        
        # === Meta Labeler Setup ===
        # Compute vertical barriers and bins for the meta labeler on the training set.
        # Note: We use the training subset of pd_closes_total, new_tEvents, and t1_meta based on feat_train's index.
        events_meta_train = getEvents_zero(
            pd_closes_total,
            new_tEvents.intersection(feat_train.index),
            ptSl_meta,
            trgt,
            minRet,
            1,  
            t1_meta.loc[feat_train.index],
            side=sides_train
        )
        bin_meta_train = getBins_meta(events_meta_train, pd_closes_total) 
        bin_meta_train['bin'] = bin_meta_train['bin'].astype(int) 
        # Train the meta labeler using the meta bins.
        meta_labeler = clfHyperFitRand(
            feat=feat_train,
            lbl=bin_meta_train['bin'],
            t1=events_meta_train['t1'],
            cv=cv_splits,
            pipe_clf=pipe_clf,
            param_grid=param_grid,
            bagging=[0, 0, 0],
            rndSearchIter=rndSearchIter,
            pctEmbargo=pctEmbargo,
            n_jobs=-1,
            clf__sample_weight=sample_weights_train
        )
        
        # === Forecasting on the Testing Set ===
        # Obtain predictions from both models on the test set.
        model1_preds = model1.predict(feat_test)
        meta_preds = meta_labeler.predict(feat_test)
        #meta_probs = meta_labeler.predict_proba(feat_test)[:, 1]
        #meta_probs = meta_labeler.predict_proba(feat_test)
        meta_probs = meta_labeler.predict_proba(feat_test)
        print(f"[Split {i}] meta_labeler.classes_: {meta_labeler.classes_}")
        print(f"[Split {i}] predict_proba shape: {meta_probs.shape}")

        # Handle both binary and single-class cases
        if meta_probs.shape[1] == 2:
            # Normal binary case → take probability of class 1
            meta_probs = meta_probs[:, 1]
        else:
            # Only one class was seen during training → fallback
            single_class = meta_labeler.classes_[0]
            if single_class == 1:
                # model only saw class 1 → prob of 1 = 1 for all
                meta_probs = np.ones(len(meta_probs))
                print(f"[Split {i}] Only class 1 seen → setting all probs to 1.")
            else:
                # model only saw class 0 → prob of 1 = 0 for all
                meta_probs = np.zeros(len(meta_probs))
                print(f"[Split {i}] Only class 0 seen → setting all probs to 0.")

        
        # Store outputs from this backtest path.
        results.append({
            'test_splits': splits_id[i],
            'train_idx': train_idx,
            'test_idx': test_idx,
            'model1_preds': model1_preds,
            'meta_preds': meta_preds,
            'meta_probs': meta_probs
        })
    
    print(f"Completed {len(results)} splits.")
    return path, results


def path_finder(path: list[list[int]], results: list[dict], k = 2, N = 6):
   N_PATHS = int(((k/N)*len(results)))
   ids = []
   for i in results:
      ids.append(i['test_splits'].copy()) 

   paths = []
   for _ in range(N_PATHS):
        model_preds = np.array([])
        meta_preds = np.array([])
        meta_probs = np.array([])
        for p in range(len(path)):
           for model,i in enumerate(ids):
              if p in i:
                 i.remove(p)
                 if results[model]['test_idx'][0] == path[p][0]:
                    model_preds = np.concatenate([model_preds, results[model]['model1_preds'][:len(path[p])]])
                    meta_preds = np.concatenate([meta_preds, results[model]['meta_preds'][:len(path[p])]])
                    meta_probs = np.concatenate([meta_probs, results[model]['meta_probs'][:len(path[p])]])
                 else:
                    model_preds = np.concatenate([model_preds, results[model]['model1_preds'][-(len(path[p])):]])
                    meta_preds = np.concatenate([meta_preds, results[model]['meta_preds'][-(len(path[p])):]])
                    meta_probs = np.concatenate([meta_probs, results[model]['meta_probs'][-(len(path[p])):]])
                 break
        paths.append([model_preds,meta_preds,meta_probs])
   return paths

