
import pandas as pd

# 读取CSV文件
df = pd.read_csv('../data/RSMLR/trainval.csv')

label_column = 'IMAGE\LABEL'

# 检查是否存在负样本
negative_samples = df[df[label_column] == 0]

# 打印负样本数量和一些负样本的信息（可选）
print("Negative samples count:", len(negative_samples))
print("Example negative samples:")
print(negative_samples.head())
