import numpy as np
from collections import Counter

# 假设有三个模型的概率输出
probs_model1 = [[0.2, 0.8], [0.6, 0.4]]
probs_model2 = [[0.6, 0.4], [0.2, 0.8]]
probs_model3 = [[0.3, 0.7], [0.2, 0.8]]

# 阈值二值化和硬投票
threshold = 0.5

# max_indices = np.argmax(probs_model1, axis=1)
pred_model1 = np.argmax(probs_model1, axis=1)
pred_model2 = np.argmax(probs_model2, axis=1)
pred_model3 = np.argmax(probs_model3, axis=1)
predictions=np.array([pred_model1.tolist(), pred_model2.tolist(), pred_model3.tolist()])

# 硬投票集成
count_1 = np.sum(predictions == 1, axis=0)
count_0 = np.sum(predictions == 0, axis=0)

# 输出数量较多的数
output = np.where(count_1 > count_0, 1, 0)

print(output)


