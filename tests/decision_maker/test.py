import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set(style="whitegrid")
# 读取数据
df = pd.read_csv('results/evaluation.csv')
"""
案例4：
绘制分割小提琴以比较跨色调变量
"""
sns.violinplot(x="group", y="Normalized objective", hue="algorithm",
               data=df, palette="muted", split=True, inner=None, cut=0)
plt.xticks(rotation=-90)
plt.show()