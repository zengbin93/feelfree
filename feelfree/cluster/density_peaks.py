# coding: utf-8
"""
special thanks to https://github.com/jasonwbw/DensityPeakCluster
基于密度峰的快速聚类方法

"""
from typing import Callable, List, Union

from .base import elu_distance


class DensityPeaks:
    def __init__(self, dc: float,
                 distance_func: Callable = elu_distance,
                 verbose: bool = True):
        self.dc = dc
        self.distance_func = distance_func
        self.cache = dict()
        self.verbose = verbose

    def _calculate_local_density(self, X: List[List[Union[int, float]]]):
        """计算每个样本的局部密度"""
        for i in range(len(X)):
            distances = [self.distance_func(X[i], row) for row in X]
            neighbors = [i for i, dist in enumerate(distances) if dist < self.dc]
            self.cache[i] = {
                "distances": distances,
                "neighbors": neighbors,
                "local_density": len(neighbors) - 1
            }

    def _calculate_min_distance(self):
        """计算每个样本与具有更高局部密度样本点的最小距离"""
        for i in self.cache.keys():
            ld = self.cache[i]['local_density']
            distances = self.cache[i]['distances']
            high_ld = [k for k, v in self.cache.items() if v['local_density'] > ld]
            if high_ld:
                min_dist = min([distances[k] for k in high_ld])
            else:
                min_dist = max(distances)

            self.cache[i]['min_distance'] = min_dist
            self.cache[i]['min_distance_index'] = distances.index(min_dist)

    def train(self, X: List[List[Union[int, float]]]):
        self._calculate_local_density(X)
        self._calculate_min_distance()

        # 确认类簇中心
        local_density_th = max([v['local_density'] for k, v in self.cache.items()]) * 0.618
        min_distance_th = max([v['min_distance'] for k, v in self.cache.items()]) * 0.618

        if self.verbose:
            print("local_density_th = {}; min_distance_th = {}.".format(local_density_th, min_distance_th))

        centre = [k for k, v in self.cache.items()
                  if v['local_density'] > local_density_th and v['min_distance'] > min_distance_th]
        clusters = {k: self.cache[k]['neighbors'] for k in centre}

        if self.verbose:
            print("类簇中心：{}".format(centre))
            print("初始类簇：{}".format(clusters))

        for i, row in enumerate(X):
            neighbor = self.cache[i]["min_distance_index"]
            for k, v in clusters.items():
                if neighbor in v:
                    clusters[k].append(i)
        return {k: list(set(v)) for k, v in clusters.items()}
