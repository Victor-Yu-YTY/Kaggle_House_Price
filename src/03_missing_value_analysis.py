# 处理缺失值
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ==========================================
# 1. 设置项目路径
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. 读取训练集和测试集
# ==========================================

train_path = DATA_DIR / "train.csv"
test_path = DATA_DIR / "test.csv"

if not train_path.exists():
    raise FileNotFoundError(f"没有找到训练集：{train_path}")

if not test_path.exists():
    raise FileNotFoundError(f"没有找到测试集：{test_path}")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("训练集形状：", train.shape)
print("测试集形状：", test.shape)


# ==========================================
# 3. 定义缺失值统计函数
# ==========================================

def get_missing_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    统计每个字段的缺失数量、缺失比例、数据类型和唯一值数量。
    """

    missing_count = data.isnull().sum()
    missing_rate = missing_count / len(data) * 100

    summary = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_rate_percent": missing_rate,
            "data_type": data.dtypes.astype(str),
            "unique_count": data.nunique(dropna=True),
        }
    )

    # 只保留存在缺失值的字段
    summary = summary[summary["missing_count"] > 0]

    # 按缺失比例从高到低排序
    summary = summary.sort_values(
        by="missing_rate_percent",
        ascending=False
    )

    return summary


# ==========================================
# 4. 分别统计训练集和测试集缺失值
# ==========================================

train_missing = get_missing_summary(train)
test_missing = get_missing_summary(test)

print("\n========== 训练集缺失值统计 ==========")
print(train_missing.to_string())

print("\n========== 测试集缺失值统计 ==========")
print(test_missing.to_string())


# ==========================================
# 5. 保存缺失值统计表
# ==========================================

train_output_path = TABLE_DIR / "train_missing_summary.csv"
test_output_path = TABLE_DIR / "test_missing_summary.csv"

train_missing.to_csv(
    train_output_path,
    index=True,
    index_label="feature"
)

test_missing.to_csv(
    test_output_path,
    index=True,
    index_label="feature"
)

print("\n训练集缺失值统计表已保存：")
print(train_output_path)

print("\n测试集缺失值统计表已保存：")
print(test_output_path)


# ==========================================
# 6. 按缺失比例划分字段
# ==========================================

def print_missing_groups(
    missing_summary: pd.DataFrame,
    dataset_name: str
) -> None:

    high_missing = missing_summary[
        missing_summary["missing_rate_percent"] >= 50
    ]

    medium_missing = missing_summary[
        (missing_summary["missing_rate_percent"] >= 20)
        & (missing_summary["missing_rate_percent"] < 50)
    ]

    low_missing = missing_summary[
        missing_summary["missing_rate_percent"] < 20
    ]

    print(f"\n========== {dataset_name}缺失程度分类 ==========")

    print("\n缺失比例大于或等于50%的字段：")
    if high_missing.empty:
        print("无")
    else:
        print(high_missing.index.tolist())

    print("\n缺失比例在20%到50%之间的字段：")
    if medium_missing.empty:
        print("无")
    else:
        print(medium_missing.index.tolist())

    print("\n缺失比例小于20%的字段：")
    if low_missing.empty:
        print("无")
    else:
        print(low_missing.index.tolist())


print_missing_groups(train_missing, "训练集")
print_missing_groups(test_missing, "测试集")


# ==========================================
# 7. 比较训练集与测试集缺失情况
# ==========================================

train_missing_for_merge = train_missing[
    ["missing_count", "missing_rate_percent"]
].rename(
    columns={
        "missing_count": "train_missing_count",
        "missing_rate_percent": "train_missing_rate"
    }
)

test_missing_for_merge = test_missing[
    ["missing_count", "missing_rate_percent"]
].rename(
    columns={
        "missing_count": "test_missing_count",
        "missing_rate_percent": "test_missing_rate"
    }
)

missing_comparison = train_missing_for_merge.join(
    test_missing_for_merge,
    how="outer"
)

missing_comparison = missing_comparison.fillna(0)

missing_comparison["rate_difference"] = (
    missing_comparison["test_missing_rate"]
    - missing_comparison["train_missing_rate"]
).abs()

missing_comparison = missing_comparison.sort_values(
    by="rate_difference",
    ascending=False
)

comparison_path = TABLE_DIR / "train_test_missing_comparison.csv"

missing_comparison.to_csv(
    comparison_path,
    index=True,
    index_label="feature"
)

print("\n========== 训练集和测试集缺失率差异最大的字段 ==========")
print(missing_comparison.head(20).to_string())

print("\n训练集与测试集缺失值对比表已保存：")
print(comparison_path)


# ==========================================
# 8. 绘制训练集缺失率柱状图
# ==========================================

top_missing = train_missing.head(25).sort_values(
    by="missing_rate_percent",
    ascending=True
)

plt.figure(figsize=(10, 8))

plt.barh(
    top_missing.index,
    top_missing["missing_rate_percent"]
)

plt.title("Top Missing Features in Training Data")
plt.xlabel("Missing Rate (%)")
plt.ylabel("Feature")

plt.tight_layout()

figure_path = FIGURE_DIR / "train_missing_rate.png"

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n训练集缺失率图已保存：")
print(figure_path)


# ==========================================
# 9. 检查目标字段是否缺失
# ==========================================

target_column = "SalePrice"

if target_column in train.columns:
    target_missing_count = train[target_column].isnull().sum()

    print("\n========== 目标字段检查 ==========")
    print("SalePrice缺失数量：", target_missing_count)

    if target_missing_count == 0:
        print("SalePrice不存在缺失，可以正常用于模型训练。")
    else:
        print("警告：SalePrice存在缺失值，需要进一步处理。")


print("\n缺失值分析完成。")