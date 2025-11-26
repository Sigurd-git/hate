import pandas as pd

# 1. 读取你合并好的数据
df = pd.read_excel("/Users/baoxuan/Desktop/研究生研究/llm毕业论文/3_cleanenglishdata.csv")  # 文件名按实际路径改

# 2. 按 label/2classes 列分成 0 和 1 两类
# 注意列名中有斜杠，直接用字符串索引即可
col = "label/2classes"

df_0 = df[df[col] == 0]
df_1 = df[df[col] == 1]

# 3. 各自随机抽取 1200 行（如果样本不够会报错）
sample_0 = df_0.sample(n=1200, random_state=42)
sample_1 = df_1.sample(n=1200, random_state=42)

# 4. 合并并打乱顺序
sampled = pd.concat([sample_0, sample_1], ignore_index=True)
sampled = sampled.sample(frac=1, random_state=42).reset_index(drop=True)

# 5. 保留表头，保存为新的文件（Excel 或 CSV 都可以）
sampled.to_excel("englishhatedata_2400_balanced.xlsx", index=False)
# 或者：
# sampled.to_csv("englishhatedata_2400_balanced.csv", index=False, encoding="utf-8")

print("Done. Saved to 4_englishhatedata_2400_balanced.xlsx")