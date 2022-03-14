#!/usr/bin/env python
# coding: utf-8
# %%
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
#from sklearn.ensemble import VotingClassifier
#from sklearn.ensemble import StackingClassifier
import time
import SklearnModels as sm
import DataModeling8w as dm
from multiprocessing import Pool
import multiprocessing
from functools import partial
from sklearn.preprocessing import StandardScaler
'''
X: The attribute characteristics of the dataset 
y: The label of the dataset
k: The number of nearest neighbors, int 
N: The top-N best models of each nearest neighbor, int
time_per_model: The time(s) limit for each model to train
data_pre_processing: Whether preprocessing dataset, bool
'''


class Automl:
    #time_per_model=360

    def __init__(
            self,
            data_pre_processing=False,
            system='linux',
            N_jobs = -1,
            verbose = False,
            time_per_model=360):#,
        
           # address_databases='./DataBases/')
        self.verbose = verbose
        self.n_estimators = 200
        self.scaler = StandardScaler()
        self.data_pre_processing = data_pre_processing
        self.k = 20
        self.N = 10
        self.address_data_feats_featurized = './DataBases/'+'data_feats_featurized.csv'#address_data_feats_featurized
        self.address_pipeline = './DataBases/'+'pipelines.json'#address_pipeline
        self.address_Top50 = './DataBases/'+'datasetTop50.csv'#address_Top50
        self.DoEnsembel = True#DoEnsembel
        self.y = []
        self.time_per_model = time_per_model
        #sm.time_per_model = time1
        self.ensemble_clf = []
        self.N_jobs = N_jobs
        if system=='linux':
            #multiprocessing.set_start_method('spawn',force=True)
            multiprocessing.set_start_method('forkserver',force=True)

    def pre_processing_X(self, X):
        col = list(X.columns)
        for j in col:
            if X[j].dtypes == 'object' or X[j].dtypes == 'O':
                b = X[j].unique()
                for i in range(len(b)):
                    X[j].loc[X[j] == b[i]] = i
                X[j] = X[j].astype("int")
        
            #print('The runtime of preprocessing is {}.\n'.format(
            #   time.perf_counter() - t))
        return X

    def fit(self, Xtrain, ytrain):
        X = Xtrain.copy(deep=True)
        y = ytrain.copy(deep=True)
        self.y = ytrain.copy(deep=True)

        preprocessing_dics, model_dics = dm.data_modeling(
            X, y, self.k, self.N, self.address_data_feats_featurized,
            self.address_pipeline, self.address_Top50, self.verbose).result  #_preprocessor
        # print('#######################################')
        n = len(preprocessing_dics)
        y = y.astype('int')
        accuracy = []
        great_models = []
#         if self.N_jobs>0:
#             pool = Pool(processes=self.N_jobs)  #()
#         else:
        pool=Pool()
        all_results = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                        test_size=0.25,
                                                        random_state=0)

        td = time.perf_counter()
        for i in range(n):
           # t_m=time.perf_counter()
            if model_dics[i][0] == 'xgradient_boosting':
                worker = sm.XGB                
                    
            elif model_dics[i][0] == 'gradient_boosting':
                worker = sm.GradientBoosting

            elif model_dics[i][0] == 'lda':
                worker = sm.LDA
                
            elif model_dics[i][0] == 'extra_trees':
                worker = sm.ExtraTrees

            elif model_dics[i][0] == 'random_forest':
                worker = sm.RandomForest

            elif model_dics[i][0] == 'decision_tree':
                worker = sm.DecisionTree
                
            elif model_dics[i][0] == 'libsvm_svc':
                worker = sm.SVM

            elif model_dics[i][0] == 'k_nearest_neighbors':
                worker = sm.KNN

            elif model_dics[i][0] == 'bernoulli_nb':
                worker = sm.BernoulliNB

            elif model_dics[i][0] == 'multinomial_nb':
                worker = sm.MultinomialNB

            elif model_dics[i][0] == 'qda':
                worker = sm.QDA

            else:
                worker = sm.GaussianNB
             
            abortable_func = partial(sm.abortable_worker, worker, timeout=self.time_per_model) 

            all_results.append(
                        pool.apply_async(abortable_func,
                                         args=(
                                             X_train,
                                             X_test,
                                             y_train,
                                             y_test,
                                             model_dics[i],
                                             self.data_pre_processing,
                                             preprocessing_dics[i],
                                         )))
                
        pool.close()
        pool.join()
        if self.verbose:
            print('The time of pools is: {}'.format(time.perf_counter() -
                                                      td))
        td = time.perf_counter()
        model_name = []
        
        all_results = np.array([
            sub_res.get() for sub_res in all_results
        ])
        #print(all_results)
        if None in all_results:
            all_results = all_results[all_results != None]
        #print(all_results)
        Y_hat = []
        for a in all_results:
            #a=sub_res.get()
            Y_hat.append(a[3])
            model_name.append(a[0])
            accuracy.append(a[2])
            great_models.append(a[1])
        if self.verbose:
            print('The time of individuals is: {}'.format(time.perf_counter() -
                                                      td))
        Y_hat=np.array(Y_hat)
        
        sort_id0 = sorted(range(len(accuracy)),
                          key=lambda m: accuracy[m],
                          reverse=True)
        
        mean_acc = np.mean(accuracy)#np.median(accuracy)#
        #mean_f1 = np.mean(f1_scores)
        estimators_stacking = []  #[great_models[sort_id[0]]]
        #X_val_predictions = [all_results[sort_id[0]][-1]]
        id_n = len(sort_id0)
        id_i = 0
        base_acc_s = []  #[accuracy[sort_id[0]]]
        
        pre=[]
        while accuracy[sort_id0[id_i]] > mean_acc: 
            pre.append(sort_id0[id_i])
                
            id_i += 1
        
        Y_hat=Y_hat[pre]
        n_pre=len(Y_hat)
         
        Res_=[] 
        
        td = time.perf_counter()
        pool = Pool()  
        for i in range(n_pre):
            Res_.append(pool.apply_async(self.Sum_diff,args=(i,n_pre,Y_hat,)))   
        pool.close()
        pool.join()
        res_=[] 
        Sort=[]
        #fa=0
        for s in Res_: 
            aa=s.get()
#             if aa[0]:
#                 fa=max(fa,min(aa[0]))
            res_.append(aa[0])
            Sort.append(aa[1])
        if self.verbose:
            print('The time of pools2 is: {}'.format(time.perf_counter() -
                                                      td))
        c = sorted(range(len(Sort)), key=lambda k: Sort[k])
        res_ = np.array(res_)[c]
        
        Rubbish=set()
        
        final=[]
        for i in range(n_pre):
            if i not in Rubbish:
                final.append(pre[i])
                for k in range(len(res_[i])):
                    if res_[i][k] == 0: 
                        Rubbish.add(i+k+1)
        
        #print(final)
        if len(final)==1:
            self.DoEnsembel=False
        estimators_stacking=[great_models[i] for i in final]#.append(great_models[sort_id0[id_i]])
        base_acc_s=[accuracy[i] for i in final]#.append(accuracy[sort_id0[id_i]])
        
       # print(self.imbalance)#, fa)
        if self.verbose:
            print(id_n, len(base_acc_s))
            print(base_acc_s, mean_acc)
        #print(base_f1_s, mean_f1)
        from mlxtend.classifier import StackingClassifier
        #from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier

        #from sklearn.preprocessing import StandardScaler
        #scaler = StandardScaler()
        # Don't cheat - fit only on training data
        #self.scaler.fit(X_val_predictions)
        #X_val_predictions_1 = self.scaler.transform(X_val_predictions)
        # apply same transformation to test data
        #X_test = self.scaler.transform(X_test)
        #meta_clf=LogisticRegression(n_jobs=-1, max_iter=500, multi_class= 'multinomial')
        if self.DoEnsembel:
            te = time.perf_counter()
            meta_clf = RandomForestClassifier(n_jobs=-1,
                                              n_estimators=self.n_estimators)
            #meta_clf=LogisticRegression(n_jobs=-1, max_iter=300)#, multi_class= 'ovr')
            #
            #meta_clf = lgb(n_jobs=-1, n_estimators=100)#, learning_rate=0.05)
            #             eclf_stacking = StackingCVClassifier(classifiers=estimators_stacking,
            #                             use_probas=True,cv=3,
            #                             meta_classifier=meta_clf,
            #                             random_state=42)

            eclf_stacking = StackingClassifier(classifiers=estimators_stacking,
                                               meta_classifier=meta_clf,
                                               use_probas=True,
                                               preprocessing=self.data_pre_processing,
                                               fit_base_estimators=False)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            #accuracy.append(
            eclf_stacking = eclf_stacking.fit(X_train, y_train)
            self.ensemble_clf = [estimators_stacking, eclf_stacking]
            if self.verbose:
                print('Ensemble val score:',
                  accuracy_score(y_test, eclf_stacking.predict(X_test)))
            
                print('The time of ensemble is: {}'.format(time.perf_counter() -
                                                       te))
            #print(self.ensemble_clf)
            #return meta_clf
        else:

            self.clf = [model_name[sort_id0[0]], great_models[sort_id0[0]]]
            if self.verbose:
                print(self.clf)
            #allresult = [great_models[sort_id[0]], accuracy[sort_id[0]]]
            return self
        
    def Sum_diff(self,i,n,Y_hat):
        res=[]
        for j in range(i+1,n):
            res.append(np.sum(Y_hat[i]!=Y_hat[j])) 
        return [res,i]

    def predict(self, Xtest):
        X_Test = Xtest.copy(deep=True)
        X_Test = self.pre_processing_X(X_Test)
        if self.DoEnsembel:

            # X_test_predictions = self.scaler.transform(X_test_predictions)
            ypre = self.ensemble_clf[1].predict(X_Test)
        else:
            if self.clf[0] == 'mnb':
                from sklearn import preprocessing
                min_max_scaler = preprocessing.MinMaxScaler()
                X_Test = min_max_scaler.fit_transform(X_Test)
            if self.data_pre_processing:
                X_Test=sm.Preprocessing(X_Test, self.clf[1][1])
           # t = time.perf_counter()
            
            ypre = self.clf[1][0].predict(X_Test)
        if self.y.dtypes == 'object' or self.y.dtypes == 'O':
            b = self.y.unique()
            return [b[i] for i in ypre]
        return ypre
