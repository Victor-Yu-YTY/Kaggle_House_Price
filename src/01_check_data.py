from pathlib import Path

import pandas as pd


# ==========================================
# 1. 设置项目路径
# ==========================================

# 当前文件位置：
# HousePrice/src/01_check_data.py
#
# parents[0] 是 src
# parents[1] 是 HousePrice
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"


# ==========================================
# 2. 检查文件是否存在
# ==========================================

print("项目根目录：", PROJECT_ROOT)
print("数据目录：", DATA_DIR)

required_files = [
    TRAIN_PATH,
    TEST_PATH,
    SAMPLE_SUBMISSION_PATH,
]

for file_path in required_files:
    if file_path.exists():
        print(f"找到文件：{file_path.name}")
    else:
        raise FileNotFoundError(
            f"没有找到文件：{file_path}\n"
            "请检查文件是否放在 data/raw 文件夹中。"
        )


# ==========================================
# 3. 读取数据
# ==========================================

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)


# ==========================================
# 4. 查看数据规模
# ==========================================

print("\n========== 数据规模 ==========")
print("训练集形状：", train.shape)
print("测试集形状：", test.shape)
print("提交样例形状：", sample_submission.shape)


# ==========================================
# 5. 查看前5行
# ==========================================

print("\n========== 训练集前5行 ==========")
print(train.head())

print("\n========== 测试集前5行 ==========")
print(test.head())

print("\n========== 提交样例前5行 ==========")
print(sample_submission.head())


# ==========================================
# 6. 检查预测目标
# ==========================================

print("\n========== 预测目标检查 ==========")

if "SalePrice" in train.columns:
    print("训练集中存在 SalePrice，可以用于训练模型。")
else:
    print("警告：训练集中没有找到 SalePrice。")

if "SalePrice" not in test.columns:
    print("测试集中不存在 SalePrice，这是正常的。")
else:
    print("警告：测试集中出现了 SalePrice。")


# ==========================================
# 7. 查看字段类型
# ==========================================

numeric_columns = train.select_dtypes(include="number").columns
categorical_columns = train.select_dtypes(exclude="number").columns

print("\n========== 特征类型 ==========")
print("数值型字段数量：", len(numeric_columns))
print("类别型字段数量：", len(categorical_columns))

print("\n前10个数值型字段：")
print(numeric_columns[:10].tolist())

print("\n前10个类别型字段：")
print(categorical_columns[:10].tolist())


# ==========================================
# 8. 查看房价统计信息
# ==========================================

print("\n========== SalePrice统计信息 ==========")
print(train["SalePrice"].describe())


# ==========================================
# 9. 查看缺失值
# ==========================================

missing_values = train.isnull().sum()
missing_values = missing_values[missing_values > 0]
missing_values = missing_values.sort_values(ascending=False)

print("\n========== 缺失值最多的前20个字段 ==========")
print(missing_values.head(20))

print("\n存在缺失值的字段总数：", len(missing_values))


# ==========================================
# 10. 检查重复数据
# ==========================================

print("\n========== 重复值检查 ==========")
print("训练集重复行数量：", train.duplicated().sum())
print("测试集重复行数量：", test.duplicated().sum())

print("\n数据检查完成。")