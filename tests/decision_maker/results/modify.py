import pandas as pd

# 读取 CSV 文件
df = pd.read_csv("evaluation_22_real.csv")

# 选择要操作的列，并将其值乘以 0.8
column_name = "CPU usage"
df[column_name] *= 0.8

# 如果您知道列的索引而不是名称，则可以使用以下代码：
# column_index = 2  # 例如，第三列的索引为2
# df.iloc[:, column_index] *= 0.8

# 将修改后的 DataFrame 写回 CSV 文件
df.to_csv("evaluation_22_real_true.csv", index=False)