# 该文件目标是：根据训练集内容，查看训练集中房价的基本统计信息，以及峰度和偏度
# 通过取对数的方法来缓解极端房价&右偏分布
# 并绘制：1、原始房价分布图；2、取对数后，绘制对数房价分布图

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==========================================
# 1. 设置项目路径
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

# 如果 figures 文件夹不存在，则自动创建
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. 读取训练集
# ==========================================

train_path = DATA_DIR / "train.csv"

if not train_path.exists():
    raise FileNotFoundError(
        f"没有找到训练数据：{train_path}"
    )

train = pd.read_csv(train_path)

print("训练集形状：", train.shape)


# ==========================================
# 3. 检查预测目标
# ==========================================

target_column = "SalePrice"

if target_column not in train.columns:
    raise KeyError(
        f"训练集中不存在目标字段：{target_column}"
    )

sale_price = train[target_column]


# ==========================================
# 4. 查看房价的基本统计信息
# ==========================================

print("\n========== SalePrice 基本统计 ==========")

print(sale_price.describe())

print("\n房价中位数：")
print(sale_price.median())

print("\n房价众数：")
print(sale_price.mode().iloc[0])

print("\n房价最小值：")
print(sale_price.min())

print("\n房价最大值：")
print(sale_price.max())

print("\n房价缺失值数量：")
print(sale_price.isnull().sum())


# ==========================================
# 5. 查看偏度和峰度
# ==========================================

skewness = sale_price.skew()
kurtosis = sale_price.kurt()

print("\n========== 分布形态 ==========")
print(f"偏度 Skewness：{skewness:.4f}")
print(f"峰度 Kurtosis：{kurtosis:.4f}")


# ==========================================
# 6. 绘制原始房价分布图
# ==========================================

plt.figure(figsize=(10, 6))

plt.hist(
    sale_price,
    bins=40,
    edgecolor="black"
)

plt.title("Distribution of SalePrice")
plt.xlabel("SalePrice")
plt.ylabel("Frequency")
plt.grid(alpha=0.2)

plt.tight_layout()

raw_figure_path = FIGURE_DIR / "saleprice_distribution.png"

plt.savefig(
    raw_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n原始房价分布图已保存：")
print(raw_figure_path)


# ==========================================
# 7. 对房价进行 log1p 转换
# ==========================================

log_sale_price = np.log1p(sale_price)

print("\n========== log1p(SalePrice) 基本统计 ==========")
print(log_sale_price.describe())

print("\n对数转换后的偏度：")
print(f"{log_sale_price.skew():.4f}")

print("\n对数转换后的峰度：")
print(f"{log_sale_price.kurt():.4f}")


# ==========================================
# 8. 绘制对数房价分布图
# ==========================================

plt.figure(figsize=(10, 6))

plt.hist(
    log_sale_price,
    bins=40,
    edgecolor="black"
)

plt.title("Distribution of log1p(SalePrice)")
plt.xlabel("log1p(SalePrice)")
plt.ylabel("Frequency")
plt.grid(alpha=0.2)

plt.tight_layout()

log_figure_path = FIGURE_DIR / "log_saleprice_distribution.png"

plt.savefig(
    log_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n对数房价分布图已保存：")
print(log_figure_path)


# ==========================================
# 9. 比较转换前后的偏度
# ==========================================

comparison = pd.DataFrame(
    {
        "Target": [
            "SalePrice",
            "log1p(SalePrice)"
        ],
        "Skewness": [
            sale_price.skew(),
            log_sale_price.skew()
        ],
        "Kurtosis": [
            sale_price.kurt(),
            log_sale_price.kurt()
        ]
    }
)

print("\n========== 转换前后对比 ==========")
print(comparison)

print("\n目标变量分析完成。")