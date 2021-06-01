# -*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
import time
from feelfree.cluster.density_peaks import DensityPeaks
from feelfree.cluster.base import elu_distance, tfidf_features


def load_dataset():
    # 西瓜数据集4.0  编号，密度，含糖率
    # 数据集来源：《机器学习》第九章 周志华教授
    data = '''
    1,0.697,0.460,
    2,0.774,0.376,
    3,0.634,0.264,
    4,0.608,0.318,
    5,0.556,0.215,
    6,0.403,0.237,
    7,0.481,0.149,
    8,0.437,0.211,
    9,0.666,0.091,
    10,0.243,0.267,
    11,0.245,0.057,
    12,0.343,0.099,
    13,0.639,0.161,
    14,0.657,0.198,
    15,0.360,0.370,
    16,0.593,0.042,
    17,0.719,0.103,
    18,0.359,0.188,
    19,0.339,0.241,
    20,0.282,0.257,
    21,0.748,0.232,
    22,0.714,0.346,
    23,0.483,0.312,
    24,0.478,0.437,
    25,0.525,0.369,
    26,0.751,0.489,
    27,0.532,0.472,
    28,0.473,0.376,
    29,0.725,0.445,
    30,0.446,0.459'''

    data_ = data.strip().split(',')
    dataset = [(float(data_[i]), float(data_[i + 1])) for i in range(1, len(data_) - 1, 3)]
    return np.array(dataset)


def show_dataset():
    dataset = load_dataset()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataset[:, 0], dataset[:, 1])
    plt.title("Dataset")
    plt.show()


def load_docs():
    docs = [
        "百度深度学习中文情感分析工具Senta试用及在线测试",
        "情感分析是自然语言处理里面一个热门话题",
        "AI Challenger 2018 文本挖掘类竞赛相关解决方案及代码汇总",
        "深度学习实践：从零开始做电影评论文本情感分析",
        "BERT相关论文、文章和代码资源汇总",
        "将不同长度的句子用BERT预训练模型编码，映射到一个固定长度的向量上",
        "自然语言处理工具包spaCy介绍",
        "现在可以快速测试一下spaCy的相关功能，我们以英文数据为例，spaCy目前主要支持英文和德文"
    ]
    return docs


def test_density_peaks():
    data = load_dataset()
    dp = DensityPeaks(dc=0.1, distance_func=elu_distance)
    clusters = dp.train(data)
    print("聚类结果：{}".format(clusters))


def test_density_peaks_performance():
    np.random.seed = '1224'
    for n in [100, 1000]:
        st = time.time()
        data = np.random.rand(n, 2)
        dp = DensityPeaks(dc=0.1, distance_func=elu_distance, verbose=False)
        _ = dp.train(data)
        print("聚类耗时：{} s".format(int(time.time() - st)))
