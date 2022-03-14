# -*- coding: utf-8 -*-
# +
import numpy as np
import pandas as pd
import json
import MetaFeaturesCalculation as MFC
#from MFs import MetaFeaturesCalculation as MFC


class get_model:
    def __init__(self, X, y, k, N, add_dff, add_pipeline, add_dT50, verbose):
        self.metadatas = MFC.metafeature_calculate(X, y, verbose).newdataset_metafeas
        #self.verbose = verbose
        self.data_feats_featurized = pd.read_csv(add_dff)
        self.data_feats_featurized = self.data_feats_featurized.iloc[:, 1:]
        with open(add_pipeline, 'r') as jsonfile:
            self.pipelines = json.load(jsonfile)
        self.dT50 = pd.read_csv(add_dT50)
        self.Model = self.choose_model(X, y, k, N)

    def kn_dist(self, X, y, k):
        Weights = [
            4.284254337774223, 1.0884829057306453, 4.60797648353123,
            8.346083710685244, 0.5216970771922169, 6.064006338779894,
            2.9732120470445165, 1.7402763881073966
        ]  #from offline stage2
        w1, w2, w3, w4, w5, w6, w7, w8 = np.array(Weights)  # / sum(Weights)
        #dff = self.data_feats_featurized.dropna()
        dff = self.data_feats_featurized.dropna().iloc[:,
                                                       list(range(0, 5)) +
                                                       list(range(7, 18)) +
                                                       list(range(19, 21)) +
                                                       [29, 31, 35] +
                                                       list(range(37, 46))]
        process1 = abs(self.metadatas[1:] - dff)
        process1 = (process1 - process1.min()) / (process1.max() -
                                                  process1.min())
        #dist_ = np.sum(process1.iloc[:, :5], axis=1) * w1 + np.sum(
        #    process1.iloc[:, 7:11], axis=1) * w2 + np.sum(
        #        process1.iloc[:, 11:17], axis=1) * w3 + np.sum(
        #            process1.iloc[:, [17, 19, 20]], axis=1) * w4 + np.sum(
        #                process1.iloc[:, [31, 29]],
        #                axis=1) * w5 + process1.iloc[:, 35] * w6 + np.sum(
        #                    process1.iloc[:, 37:41], axis=1) * w7 + np.sum(
        #                        process1.iloc[:, 41:46], axis=1) * w8
        dist_ = np.sum(process1.iloc[:, :5], axis=1) * w1 + np.sum(
            process1.iloc[:, 5:9], axis=1) * w2 + np.sum(
                process1.iloc[:, 9:15], axis=1) * w3 + np.sum(
                    process1.iloc[:, 15:18], axis=1) * w4 + np.sum(
                        process1.iloc[:, [18, 19]],
                        axis=1) * w5 + process1.iloc[:, 20] * w6 + np.sum(
                            process1.iloc[:, 21:25], axis=1) * w7 + np.sum(
                                process1.iloc[:, 25:30], axis=1) * w8
        knd = list(np.argsort(dist_)[:k]) 
        return knd

    def choose_model(self, X, y, k, N):
        kn_list = self.kn_dist(X, y, k)
        model_id = []
        for i in kn_list:
            a = self.dT50.iloc[i, :N]
            model_id.extend(a)
        model = []
        model_id = list(set(model_id))  ##去重
        for i in model_id:
            model.append(self.pipelines[i])
        return model, self.metadatas[0]
