import numpy as np


# 可以根据需要调整p值，表示Minkowski距离中的幂指数
def _distance(x1, x2, model='manhattan', p=3):
    distance = 0
    # 曼哈顿距离度量
    if model == 'manhattan':
        distance = np.sum(np.abs(x1 - x2))
    # 欧几里得距离度量
    elif model == 'euclidean':
        distance = np.sqrt(np.sum(np.square(x1 - x2)))
    # 闵可夫斯基距离度量
    elif model == 'minkowski':
        distance = np.power(np.sum(np.power(np.abs(x1 - x2), p)), 1/p)
    return distance