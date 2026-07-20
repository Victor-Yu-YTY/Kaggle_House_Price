# 异常值处理
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==========================================
# 1. 设置项目路径
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

print("原始训练集形状：", train.shape)
print("原始训练集缺失值总数：", train.isna().sum().sum())


# ==========================================
# 3. 检查必要字段
# ==========================================

required_columns = [
    "Id",
    "SalePrice",
    "GrLivArea",
    "OverallQual",
]

missing_columns = [
    column
    for column in required_columns
    if column not in train.columns
]

if missing_columns:
    raise KeyError(
        f"训练集中缺少必要字段：{missing_columns}"
    )


# ==========================================
# 4. 创建对数房价
# ==========================================

train_analysis = train.copy()

train_analysis["LogSalePrice"] = np.log1p(
    train_analysis["SalePrice"]
)


# ==========================================
# 5. 查看 GrLivArea 较大的房屋
# ==========================================

large_house_candidates = train_analysis.loc[
    train_analysis["GrLivArea"] > 4000
].copy()

candidate_columns = [
    "Id",
    "GrLivArea",
    "OverallQual",
    "SalePrice",
    "LogSalePrice",
]

# 有这些字段就一并展示
optional_columns = [
    "TotalBsmtSF",
    "1stFlrSF",
    "GarageCars",
    "GarageArea",
    "Neighborhood",
    "YearBuilt",
]

for column in optional_columns:
    if column in large_house_candidates.columns:
        candidate_columns.append(column)

large_house_candidates = large_house_candidates[
    candidate_columns
].sort_values(
    by="GrLivArea",
    ascending=False
)

print("\n========== GrLivArea 大于 4000 的样本 ==========")
print(large_house_candidates.to_string(index=False))

candidate_path = (
    TABLE_DIR / "large_house_candidates_detailed.csv"
)

large_house_candidates.to_csv(
    candidate_path,
    index=False
)

print("\n大面积房屋候选表已保存：")
print(candidate_path)


# ==========================================
# 6. 定义明显异常值规则
# ==========================================

# 面积极大，但售价明显偏低
outlier_condition = (
    (train_analysis["GrLivArea"] > 4000)
    & (train_analysis["SalePrice"] < 300000)
)

outliers = train_analysis.loc[
    outlier_condition
].copy()

outliers = outliers[
    candidate_columns
].sort_values(
    by="GrLivArea",
    ascending=False
)

print("\n========== 根据规则识别出的明显异常样本 ==========")

if outliers.empty:
    print("没有找到符合规则的异常样本。")
else:
    print(outliers.to_string(index=False))

removed_outliers_path = (
    TABLE_DIR / "removed_outliers.csv"
)

outliers.to_csv(
    removed_outliers_path,
    index=False
)

print("\n异常样本表已保存：")
print(removed_outliers_path)


# ==========================================
# 7. 绘制删除前的散点图
# ==========================================

plt.figure(figsize=(10, 7))

plt.scatter(
    train_analysis["GrLivArea"],
    train_analysis["LogSalePrice"],
    alpha=0.5,
    label="Normal samples"
)

if not outliers.empty:
    plt.scatter(
        outliers["GrLivArea"],
        outliers["LogSalePrice"],
        s=100,
        marker="x",
        label="Outliers"
    )

plt.title("GrLivArea vs LogSalePrice Before Outlier Removal")
plt.xlabel("GrLivArea")
plt.ylabel("log1p(SalePrice)")
plt.grid(alpha=0.2)
plt.legend()

plt.tight_layout()

before_figure_path = (
    FIGURE_DIR / "grlivarea_before_outlier_removal.png"
)

plt.savefig(
    before_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n删除异常值前的图已保存：")
print(before_figure_path)


# ==========================================
# 8. 删除明确异常样本
# ==========================================

train_model = train.loc[
    ~outlier_condition
].copy()

# 重新整理索引
train_model = train_model.reset_index(drop=True)

print("\n========== 删除结果 ==========")
print("删除的样本数量：", outlier_condition.sum())
print("删除前训练集形状：", train.shape)
print("删除后训练集形状：", train_model.shape)


# ==========================================
# 9. 对比删除前后的相关系数
# ==========================================

comparison_features = [
    "GrLivArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "GarageArea",
    "GarageCars",
    "OverallQual",
]

comparison_rows = []

for feature in comparison_features:

    if feature not in train.columns:
        continue

    before_log_target = np.log1p(
        train["SalePrice"]
    )

    after_log_target = np.log1p(
        train_model["SalePrice"]
    )

    before_correlation = train[
        feature
    ].corr(
        before_log_target
    )

    after_correlation = train_model[
        feature
    ].corr(
        after_log_target
    )

    comparison_rows.append(
        {
            "feature": feature,
            "correlation_before": before_correlation,
            "correlation_after": after_correlation,
            "change": (
                after_correlation
                - before_correlation
            ),
        }
    )

correlation_comparison = pd.DataFrame(
    comparison_rows
)

print("\n========== 删除前后相关系数对比 ==========")
print(correlation_comparison.to_string(index=False))

comparison_path = (
    TABLE_DIR / "outlier_correlation_comparison.csv"
)

correlation_comparison.to_csv(
    comparison_path,
    index=False
)

print("\n相关系数对比表已保存：")
print(comparison_path)


# ==========================================
# 10. 绘制删除后的散点图
# ==========================================

log_sale_price_after = np.log1p(
    train_model["SalePrice"]
)

plt.figure(figsize=(10, 7))

plt.scatter(
    train_model["GrLivArea"],
    log_sale_price_after,
    alpha=0.5
)

plt.title("GrLivArea vs LogSalePrice After Outlier Removal")
plt.xlabel("GrLivArea")
plt.ylabel("log1p(SalePrice)")
plt.grid(alpha=0.2)

plt.tight_layout()

after_figure_path = (
    FIGURE_DIR / "grlivarea_after_outlier_removal.png"
)

plt.savefig(
    after_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n删除异常值后的图已保存：")
print(after_figure_path)


# ==========================================
# 11. 保存后续建模使用的数据
# ==========================================

train_model_path = (
    PROCESSED_DATA_DIR / "train_model.csv"
)

train_model.to_csv(
    train_model_path,
    index=False
)

print("\n用于后续建模的训练集已保存：")
print(train_model_path)

print("\n异常值分析完成。")