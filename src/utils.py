"""
导入库

import logging: 用于日志记录。
import sys: 用于访问与 Python 解释器交互的变量和函数。
import torch: PyTorch库，用于深度学习和张量操作。
import numpy as np: NumPy库，用于数值计算。
import argparse: 用于命令行参数解析。
from sklearn.metrics import roc_auc_score: 从scikit-learn库导入roc_auc_score函数，用于计算ROC曲线下的面积（AUC）。
str2bool 函数

用于将字符串参数转换为布尔值。
接收一个字符串输入 (v)，判断其值是否是表示True（如 "yes", "true", "t", "y", "1"）或False（如 "no", "false", "f", "n", "0"）。
如果输入不是布尔值或上述字符串之一，则引发一个参数类型错误。
setuplogging 函数

设置日志记录器的配置。
创建一个日志记录器对象并将其级别设置为INFO。
创建一个日志处理器，将日志消息输出到标准输出（控制台）。
设置日志消息的格式（包含日志级别、时间和消息内容）。
将处理器添加到日志记录器。
acc 函数

计算分类任务中的准确率。
y_true 是真实标签，y_hat 是模型的预测值。
使用torch.argmax函数找出预测结果中的最大值索引（即预测标签）。
计算正确预测的数量hit，然后计算并返回准确率。
dcg_score 函数

计算给定排名结果的折扣累积增益（DCG）。
y_true 是实际标签，y_score 是模型分数。
首先根据模型得分对结果排序，然后计算收益（gains）和折扣（discounts），最后计算DCG值。
ndcg_score 函数

计算标准化折扣累积增益（NDCG）。
NDCG 是 DCG 的归一化形式，用于评估排序模型的性能。
通过计算实际DCG（actual）和理想情况下的DCG（best），返回实际DCG与理想DCG的比值。
mrr_score 函数

计算平均倒数排名（MRR）。
y_true 是实际标签，y_score 是模型分数。
根据模型得分对结果排序，计算每个相关结果的倒数排名得分，并返回MRR值。
总结
该代码主要用于模型训练和评估的辅助工具，提供了日志记录、布尔参数处理、分类准确率和排序模型性能的多种评估指标。
"""
import logging
import sys
import torch
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")



def setuplogging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)

def acc(y_true, y_hat):
    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

