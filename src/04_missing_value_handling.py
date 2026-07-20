from pathlib import Path

import pandas as pd


# ==========================================
# 1. 项目路径
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. 读取训练集和测试集
# ==========================================

train_path = RAW_DATA_DIR / "train.csv"
test_path = RAW_DATA_DIR / "test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# 保留原始数据不被修改
train_clean = train.copy()
test_clean = test.copy()

print("处理前训练集缺失值总数：", train_clean.isna().sum().sum())
print("处理前测试集缺失值总数：", test_clean.isna().sum().sum())


# ==========================================
# 3. 类别型字段：缺失表示没有该设施
# ==========================================

none_category_columns = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "MasVnrType",
]

for column in none_category_columns:
    if column in train_clean.columns:
        train_clean[column] = train_clean[column].fillna("NoFeature")

    if column in test_clean.columns:
        test_clean[column] = test_clean[column].fillna("NoFeature")


# ==========================================
# 4. 数值型字段：缺失表示数量或面积为0
# ==========================================

zero_numeric_columns = [
    "GarageYrBlt",
    "GarageCars",
    "GarageArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "MasVnrArea",
]

for column in zero_numeric_columns:
    if column in train_clean.columns:
        train_clean[column] = train_clean[column].fillna(0)

    if column in test_clean.columns:
        test_clean[column] = test_clean[column].fillna(0)


# ==========================================
# 5. LotFrontage：按社区中位数填补
# ==========================================

if (
    "LotFrontage" in train_clean.columns
    and "Neighborhood" in train_clean.columns
):
    # 只使用训练集计算各社区的中位数，避免使用测试集信息
    neighborhood_medians = train_clean.groupby(
        "Neighborhood"
    )["LotFrontage"].median()

    overall_median = train_clean["LotFrontage"].median()

    train_clean["LotFrontage"] = (
        train_clean["LotFrontage"]
        .fillna(
            train_clean["Neighborhood"].map(
                neighborhood_medians
            )
        )
        .fillna(overall_median)
    )

    test_clean["LotFrontage"] = (
        test_clean["LotFrontage"]
        .fillna(
            test_clean["Neighborhood"].map(
                neighborhood_medians
            )
        )
        .fillna(overall_median)
    )


# ==========================================
# 6. 少量缺失的类别字段：使用训练集众数
# ==========================================

mode_fill_columns = [
    "MSZoning",
    "Utilities",
    "Exterior1st",
    "Exterior2nd",
    "Electrical",
    "KitchenQual",
    "SaleType",
]

for column in mode_fill_columns:
    if column not in train_clean.columns:
        continue

    mode_values = train_clean[column].mode(dropna=True)

    if mode_values.empty:
        fill_value = "Unknown"
    else:
        fill_value = mode_values.iloc[0]

    train_clean[column] = train_clean[column].fillna(
        fill_value
    )

    if column in test_clean.columns:
        test_clean[column] = test_clean[column].fillna(
            fill_value
        )


# ==========================================
# 7. Functional 缺失时按典型功能处理
# ==========================================

if "Functional" in train_clean.columns:
    train_clean["Functional"] = (
        train_clean["Functional"].fillna("Typ")
    )

if "Functional" in test_clean.columns:
    test_clean["Functional"] = (
        test_clean["Functional"].fillna("Typ")
    )


# ==========================================
# 8. 兜底处理剩余缺失值
# ==========================================

feature_columns = [
    column
    for column in train_clean.columns
    if column != "SalePrice"
]

for column in feature_columns:
    train_has_missing = train_clean[column].isna().any()
    test_has_missing = test_clean[column].isna().any()

    if not train_has_missing and not test_has_missing:
        continue

    # 数值型字段使用训练集中位数
    if pd.api.types.is_numeric_dtype(train_clean[column]):
        fill_value = train_clean[column].median()

    # 类别型字段使用训练集众数
    else:
        mode_values = train_clean[column].mode(dropna=True)

        if mode_values.empty:
            fill_value = "Unknown"
        else:
            fill_value = mode_values.iloc[0]

    train_clean[column] = train_clean[column].fillna(
        fill_value
    )

    test_clean[column] = test_clean[column].fillna(
        fill_value
    )

    print(
        f"兜底填补字段：{column}，填充值：{fill_value}"
    )


# ==========================================
# 9. 检查处理结果
# ==========================================

train_remaining_missing = train_clean.isna().sum()
train_remaining_missing = train_remaining_missing[
    train_remaining_missing > 0
]

test_remaining_missing = test_clean.isna().sum()
test_remaining_missing = test_remaining_missing[
    test_remaining_missing > 0
]

print("\n========== 处理结果 ==========")

print(
    "处理后训练集缺失值总数：",
    train_clean.isna().sum().sum()
)

print(
    "处理后测试集缺失值总数：",
    test_clean.isna().sum().sum()
)

print("\n训练集剩余缺失字段：")
print(train_remaining_missing)

print("\n测试集剩余缺失字段：")
print(test_remaining_missing)


# ==========================================
# 10. 保存清洗后的数据
# ==========================================

processed_train_path = (
    PROCESSED_DATA_DIR / "train_clean.csv"
)

processed_test_path = (
    PROCESSED_DATA_DIR / "test_clean.csv"
)

train_clean.to_csv(
    processed_train_path,
    index=False
)

test_clean.to_csv(
    processed_test_path,
    index=False
)

print("\n清洗后的训练集已保存：")
print(processed_train_path)

print("\n清洗后的测试集已保存：")
print(processed_test_path)


# ==========================================
# 11. 保存缺失值处理前后对比表
# ==========================================

all_columns = sorted(
    set(train.columns).union(test.columns)
)

report_rows = []

for column in all_columns:
    report_rows.append(
        {
            "feature": column,
            "train_missing_before": (
                train[column].isna().sum()
                if column in train.columns
                else None
            ),
            "train_missing_after": (
                train_clean[column].isna().sum()
                if column in train_clean.columns
                else None
            ),
            "test_missing_before": (
                test[column].isna().sum()
                if column in test.columns
                else None
            ),
            "test_missing_after": (
                test_clean[column].isna().sum()
                if column in test_clean.columns
                else None
            ),
        }
    )

missing_report = pd.DataFrame(report_rows)

report_path = TABLE_DIR / "missing_value_handling_report.csv"

missing_report.to_csv(
    report_path,
    index=False
)

print("\n缺失值处理报告已保存：")
print(report_path)

print("\n缺失值处理完成。")