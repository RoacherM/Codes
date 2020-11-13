#!/opt/miniconda/bin/python
# coding: utf-8

# -------------------------------------------------
#    FilePath: /project/ACLSUTER.py
#    Description:     短文本聚类程序
#    Author:     WY
#    Date: 2020/06/19
#    LastEditTime: 2020/06/30
# -------------------------------------------------

import SIM_UTILS


class Models():
    def __init__(self):
        pass

    def Kmeans(self, texts):
        pass

    def DBSCAN(self, texts):
        pass

    def GMM(self, texts):
        pass


if __name__ == '__main__':

    # import numpy as np
    # A = SIM_UTILS.normalize(np.random.rand(768, 1000))
    # print(A[0])
    # print(SIM_UTILS.matrixD(A, A))
    states_needed = set(["mt", "wa", "or", "id", "nv", "ut", "ca", "az"])
    stations = {}
    stations['kone'] = set(["id", "nv", "ut"])
    stations['ktwo'] = set(["wa", "id", "mt"])
    stations['kthree'] = set(["or", "nv", "ca"])
    stations['kfour'] = set(["nv", "ut"])
    stations['kfive'] = set(["ca", "za"])
    final_stations = set()

    # while states_needed:
    #     best_station = None
    #     states_covered = set()
    #     for station, states in stations.items():
    #         covered = states_needed & states
    #         print(covered)
    #         if len(covered) > len(states_covered):
    #             best_station = station
    #             states_covered = covered

    #     states_needed -= states_covered
    #     final_stations.add(best_station)

    # print(final_stations)
    print(SIM_UTILS.cut('2018年03月02日05时30分，民警王益根在巡逻中发现一名流浪人员。'))