import seaborn as sns
import matplotlib.pyplot as plt
# set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
sns.set(style="darkgrid")

# 加载示例数据集
df = sns.load_dataset('iris')
df.head()

# 使用kdeplot函数绘制密度分布图
# Make default density plot
sns.kdeplot([1,2,3,4,5,6,7,9,10])
plt.savefig("test.png")