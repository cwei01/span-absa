#!/usr/bin/env python
# -*- coding:utf-8 -*-
#!/usr/bin/env python
# -*- coding:utf-8 -*
#
import random



import torch
import math
import numpy as np
random.seed(20)
#np.random.seed(20)

# a=torch.tensor(2,2,dtype=torch.int64)
# b=torch.sigmoid(a)
# c=torch.softmax(a,dim=-1)
# def _get_best_indexes(logits, n_best_size):
#     index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
#     best_indexes = []
#     for i in range(len(index_and_score)):
#         if i >= n_best_size:
#             break
#         best_indexes.append(index_and_score[i][0])
#     return best_indexes
# print(a)
# print(b)
# print(c)
# x= torch.zeros(2,3,2, dtype=torch.int64)
# y= torch.ones(2,1,2, dtype=torch.int64)
# print(x)
# print(y)
# print(x.add_(y))
# a=0.5
# def _get_best_indexes(logits, n_best_size):
#     index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
#     best_indexes = []
#     for i in range(len(index_and_score)):
#         if i >= n_best_size:
#             break
#         best_indexes.append(index_and_score[i][0])
#     return best_indexes
# print(_get_best_indexes(a,5))
# a=torch.tensor([[ 1,  2,  3],[ 1,  2,  3],[ 1,  2,  3]])
# b=torch.triu(a).view(-1)
# c=torch.nonzero(b)
# print(b)
# print(b[c].view(-1))
# a =torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
# dia_index=torch.tensor([[i for i in range(3)]])
# c=a.gather(0,dia_index)
# for i in range(4):
#   print(random.random())
# print(random.random())
for _ in range(0):
  t=2
  print(t)




