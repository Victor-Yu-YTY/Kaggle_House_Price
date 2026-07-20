from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==========================================
# 1. 设置路径
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. 读取清洗后的训练集
# ==========================================

train_path = PROCESSED_DATA_DIR / "train_clean.csv"

if not train_path.exists():
    raise FileNotFoundError(
        f"没有找到清洗后的训练集：{train_path}\n"
        "请先运行 04_missing_value_handling.py"
    )

train = pd.read_csv(train_path)

print("训练集形状：", train.shape)
print("训练集缺失值总数：", train.isna().sum().sum())


# ==========================================
# 3. 检查目标字段
# ==========================================

target_column = "SalePrice"

if target_column not in train.columns:
    raise KeyError(
        f"训练集中不存在目标字段：{target_column}"
    )

sale_price = train[target_column]
log_sale_price = np.log1p(sale_price)


# ==========================================
# 4. 找出数值型字段
# ==========================================

numeric_columns = train.select_dtypes(
    include=["number"]
).columns.tolist()

print("\n数值型字段数量（包括Id和SalePrice）：")
print(len(numeric_columns))

# Id只是编号，不是正常的房屋特征
numeric_feature_columns = [
    column
    for column in numeric_columns
    if column not in ["Id", "SalePrice"]
]

print("\n真正用于分析的数值型特征数量：")
print(len(numeric_feature_columns))


# ==========================================
# 5. 计算与原始房价的相关系数
# ==========================================

raw_correlations = train[
    numeric_feature_columns
].corrwith(sale_price)


# ==========================================
# 6. 计算与对数房价的相关系数
# ==========================================

log_correlations = train[
    numeric_feature_columns
].corrwith(log_sale_price)


# ==========================================
# 7. 整理相关性结果
# ==========================================

correlation_summary = pd.DataFrame(
    {
        "correlation_with_saleprice": raw_correlations,
        "correlation_with_log_saleprice": log_correlations,
    }
)

correlation_summary["absolute_log_correlation"] = (
    correlation_summary[
        "correlation_with_log_saleprice"
    ].abs()
)

correlation_summary = correlation_summary.sort_values(
    by="absolute_log_correlation",
    ascending=False
)

print("\n========== 与对数房价相关性最高的前20个特征 ==========")

print(
    correlation_summary.head(20).to_string()
)


# ==========================================
# 8. 保存相关系数表
# ==========================================

correlation_table_path = (
    TABLE_DIR / "numeric_feature_correlations.csv"
)

correlation_summary.to_csv(
    correlation_table_path,
    index=True,
    index_label="feature"
)

print("\n数值特征相关性表已保存：")
print(correlation_table_path)


# ==========================================
# 9. 绘制前20个重要特征相关系数柱状图
# ==========================================

top_20 = correlation_summary.head(20).copy()

# 横向柱状图从低到高排列
top_20 = top_20.sort_values(
    by="correlation_with_log_saleprice",
    ascending=True
)

plt.figure(figsize=(10, 8))

plt.barh(
    top_20.index,
    top_20["correlation_with_log_saleprice"]
)

plt.title("Top Numeric Features Correlated with Log SalePrice")
plt.xlabel("Correlation with log1p(SalePrice)")
plt.ylabel("Feature")
plt.axvline(x=0, linewidth=1)

plt.tight_layout()

bar_figure_path = (
    FIGURE_DIR / "numeric_feature_correlations.png"
)

plt.savefig(
    bar_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n数值特征相关性柱状图已保存：")
print(bar_figure_path)


# ==========================================
# 10. 绘制重要特征相关系数矩阵
# ==========================================

top_10_features = (
    correlation_summary.head(10).index.tolist()
)

heatmap_data = train[top_10_features].copy()
heatmap_data["LogSalePrice"] = log_sale_price

correlation_matrix = heatmap_data.corr()

plt.figure(figsize=(11, 9))

image = plt.imshow(
    correlation_matrix,
    vmin=-1,
    vmax=1
)

plt.colorbar(
    image,
    label="Correlation"
)

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation_matrix.index)),
    correlation_matrix.index
)

# 在每个格子中显示相关系数
for row in range(len(correlation_matrix.index)):
    for column in range(len(correlation_matrix.columns)):
        value = correlation_matrix.iloc[row, column]

        plt.text(
            column,
            row,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

plt.title("Correlation Matrix of Important Numeric Features")
plt.tight_layout()

matrix_figure_path = (
    FIGURE_DIR / "important_numeric_correlation_matrix.png"
)

plt.savefig(
    matrix_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n重要数值特征相关矩阵已保存：")
print(matrix_figure_path)


# ==========================================
# 11. 绘制前6个重要数值特征与房价的关系图
# ==========================================

top_6_features = (
    correlation_summary.head(6).index.tolist()
)

print("\n用于绘制散点图的前6个特征：")
print(top_6_features)

for feature in top_6_features:

    plt.figure(figsize=(8, 6))

    plt.scatter(
        train[feature],
        log_sale_price,
        alpha=0.5
    )

    plt.title(
        f"{feature} vs log1p(SalePrice)"
    )

    plt.xlabel(feature)
    plt.ylabel("log1p(SalePrice)")
    plt.grid(alpha=0.2)

    plt.tight_layout()

    figure_path = (
        FIGURE_DIR
        / f"{feature}_vs_log_saleprice.png"
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"关系图已保存：{figure_path}")


# ==========================================
# 12. 检查典型大面积异常样本
# ==========================================

if "GrLivArea" in train.columns:

    large_house_candidates = train.loc[
        train["GrLivArea"] > 4000,
        [
            "Id",
            "GrLivArea",
            "OverallQual",
            "SalePrice"
        ]
    ].sort_values(
        by="GrLivArea",
        ascending=False
    )

    print("\n========== GrLivArea大于4000的样本 ==========")
    print(large_house_candidates.to_string(index=False))

    outlier_path = (
        TABLE_DIR / "large_house_outlier_candidates.csv"
    )

    large_house_candidates.to_csv(
        outlier_path,
        index=False
    )

    print("\n疑似大面积异常样本已保存：")
    print(outlier_path)


print("\n数值型特征分析完成。")