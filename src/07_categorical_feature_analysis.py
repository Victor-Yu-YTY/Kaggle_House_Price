# 类别型特征分析
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
# 2. 读取异常值处理后的训练集
# ==========================================

train_path = PROCESSED_DATA_DIR / "train_model.csv"

if not train_path.exists():
    raise FileNotFoundError(
        f"没有找到建模训练集：{train_path}\n"
        "请先运行 06_outlier_analysis.py"
    )

train = pd.read_csv(train_path)

print("训练集形状：", train.shape)
print("缺失值总数：", train.isna().sum().sum())


# ==========================================
# 3. 检查目标字段
# ==========================================

if "SalePrice" not in train.columns:
    raise KeyError("训练集中不存在 SalePrice 字段。")

train["LogSalePrice"] = np.log1p(
    train["SalePrice"]
)


# ==========================================
# 4. 将编码型数值字段转换为类别字段
# ==========================================

# MSSubClass 虽然用数字表示，但数字代表房屋类型编号，
# 不应被理解为连续数值大小。
if "MSSubClass" in train.columns:
    train["MSSubClass"] = (
        train["MSSubClass"].astype(str)
    )


# ==========================================
# 5. 找出所有类别型字段
# ==========================================

categorical_columns = train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\n类别型字段数量：", len(categorical_columns))
print("\n类别型字段：")
print(categorical_columns)


# ==========================================
# 6. 定义 Eta Squared 计算函数
# ==========================================

def calculate_eta_squared(
    categories: pd.Series,
    values: pd.Series
) -> float:
    """
    计算类别变量对连续目标变量的解释程度。

    Eta Squared 取值范围为 0 到 1：
    越接近1，说明不同类别之间的目标值差异越明显。
    """

    data = pd.DataFrame(
        {
            "category": categories,
            "value": values
        }
    ).dropna()

    if data.empty:
        return np.nan

    overall_mean = data["value"].mean()

    total_sum_of_squares = (
        (data["value"] - overall_mean) ** 2
    ).sum()

    if total_sum_of_squares == 0:
        return 0.0

    between_group_sum_of_squares = 0.0

    for _, group in data.groupby(
        "category",
        observed=False
    ):
        group_size = len(group)
        group_mean = group["value"].mean()

        between_group_sum_of_squares += (
            group_size
            * (group_mean - overall_mean) ** 2
        )

    eta_squared = (
        between_group_sum_of_squares
        / total_sum_of_squares
    )

    return float(eta_squared)


# ==========================================
# 7. 统计类别字段的基本信息
# ==========================================

summary_rows = []

for column in categorical_columns:

    value_counts = train[column].value_counts(
        dropna=False
    )

    unique_count = train[column].nunique(
        dropna=False
    )

    rare_category_count = (
        value_counts < 10
    ).sum()

    eta_squared = calculate_eta_squared(
        train[column],
        train["LogSalePrice"]
    )

    summary_rows.append(
        {
            "feature": column,
            "unique_count": unique_count,
            "smallest_category_size": (
                int(value_counts.min())
            ),
            "largest_category_size": (
                int(value_counts.max())
            ),
            "rare_category_count_less_than_10": (
                int(rare_category_count)
            ),
            "eta_squared": eta_squared,
        }
    )

categorical_summary = pd.DataFrame(
    summary_rows
)

categorical_summary = categorical_summary.sort_values(
    by="eta_squared",
    ascending=False
)

print(
    "\n========== 类别型特征影响程度前20名 =========="
)

print(
    categorical_summary.head(20).to_string(
        index=False
    )
)


# ==========================================
# 8. 保存类别特征汇总表
# ==========================================

summary_path = (
    TABLE_DIR / "categorical_feature_summary.csv"
)

categorical_summary.to_csv(
    summary_path,
    index=False
)

print("\n类别特征汇总表已保存：")
print(summary_path)


# ==========================================
# 9. 绘制类别特征影响程度柱状图
# ==========================================

top_20 = categorical_summary.head(20).copy()

top_20 = top_20.sort_values(
    by="eta_squared",
    ascending=True
)

plt.figure(figsize=(10, 8))

plt.barh(
    top_20["feature"],
    top_20["eta_squared"]
)

plt.title(
    "Categorical Feature Importance by Eta Squared"
)
plt.xlabel("Eta Squared")
plt.ylabel("Categorical Feature")
plt.grid(axis="x", alpha=0.2)

plt.tight_layout()

importance_figure_path = (
    FIGURE_DIR
    / "categorical_feature_eta_squared.png"
)

plt.savefig(
    importance_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n类别特征影响程度图已保存：")
print(importance_figure_path)


# ==========================================
# 10. 分析影响程度最高的前8个类别特征
# ==========================================

top_features = (
    categorical_summary
    .head(8)["feature"]
    .tolist()
)

print("\n重点分析的类别特征：")
print(top_features)


for feature in top_features:

    feature_data = train[
        [
            feature,
            "SalePrice",
            "LogSalePrice"
        ]
    ].copy()

    # 避免标签显示异常
    feature_data[feature] = (
        feature_data[feature]
        .fillna("Missing")
        .astype(str)
    )

    category_statistics = (
        feature_data
        .groupby(
            feature,
            observed=False
        )
        .agg(
            sample_count=(
                "SalePrice",
                "size"
            ),
            median_saleprice=(
                "SalePrice",
                "median"
            ),
            mean_saleprice=(
                "SalePrice",
                "mean"
            ),
            mean_log_saleprice=(
                "LogSalePrice",
                "mean"
            )
        )
        .sort_values(
            by="median_saleprice",
            ascending=False
        )
    )

    print(
        f"\n========== {feature} 类别统计 =========="
    )
    print(category_statistics.to_string())

    # 保存每个重要类别特征的统计表
    statistics_path = (
        TABLE_DIR
        / f"{feature}_category_statistics.csv"
    )

    category_statistics.to_csv(
        statistics_path,
        index=True,
        index_label=feature
    )

    # 按对数房价中位数排序，用于绘制箱线图
    category_order = (
        feature_data
        .groupby(
            feature,
            observed=False
        )["LogSalePrice"]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    # 类别太多时，只展示样本量最大的25类
    if len(category_order) > 25:
        largest_categories = (
            feature_data[feature]
            .value_counts()
            .head(25)
            .index
            .tolist()
        )

        category_order = [
            category
            for category in category_order
            if category in largest_categories
        ]

    plot_data = []
    plot_labels = []

    for category in category_order:

        category_values = feature_data.loc[
            feature_data[feature] == category,
            "LogSalePrice"
        ]

        plot_data.append(
            category_values.to_numpy()
        )

        plot_labels.append(
            f"{category} (n={len(category_values)})"
        )

    figure_height = max(
        6,
        len(plot_labels) * 0.38
    )

    plt.figure(
        figsize=(11, figure_height)
    )

    plt.boxplot(
        plot_data,
        labels=plot_labels,
        vert=False,
        showfliers=False
    )

    plt.title(
        f"{feature} vs log1p(SalePrice)"
    )
    plt.xlabel("log1p(SalePrice)")
    plt.ylabel(feature)
    plt.grid(axis="x", alpha=0.2)

    plt.tight_layout()

    boxplot_path = (
        FIGURE_DIR
        / f"{feature}_category_boxplot.png"
    )

    plt.savefig(
        boxplot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"图片已保存：{boxplot_path}")


# ==========================================
# 11. 单独分析 Neighborhood
# ==========================================

if "Neighborhood" in train.columns:

    neighborhood_summary = (
        train
        .groupby(
            "Neighborhood",
            observed=False
        )
        .agg(
            sample_count=(
                "SalePrice",
                "size"
            ),
            median_saleprice=(
                "SalePrice",
                "median"
            ),
            mean_saleprice=(
                "SalePrice",
                "mean"
            )
        )
        .sort_values(
            by="median_saleprice",
            ascending=False
        )
    )

    print(
        "\n========== Neighborhood 房价统计 =========="
    )
    print(neighborhood_summary.to_string())

    neighborhood_path = (
        TABLE_DIR
        / "neighborhood_price_summary.csv"
    )

    neighborhood_summary.to_csv(
        neighborhood_path,
        index=True
    )

    print("\n社区房价统计表已保存：")
    print(neighborhood_path)


# ==========================================
# 12. 找出稀有类别较多的字段
# ==========================================

rare_feature_summary = (
    categorical_summary[
        categorical_summary[
            "rare_category_count_less_than_10"
        ] > 0
    ]
    .sort_values(
        by="rare_category_count_less_than_10",
        ascending=False
    )
)

rare_path = (
    TABLE_DIR
    / "rare_categorical_features.csv"
)

rare_feature_summary.to_csv(
    rare_path,
    index=False
)

print("\n========== 含有稀有类别的字段 ==========")
print(
    rare_feature_summary.head(20).to_string(
        index=False
    )
)

print("\n稀有类别汇总表已保存：")
print(rare_path)

print("\n类别型特征分析完成。")