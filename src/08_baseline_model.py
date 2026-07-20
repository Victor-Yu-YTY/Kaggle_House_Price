# 完整流程
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==========================================
# 1. 设置项目路径
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
SUBMISSION_DIR = PROJECT_ROOT / "outputs" / "submissions"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. 自定义房价数据清洗器
# ==========================================

class HousePriceCleaner(BaseEstimator, TransformerMixin):
    """
    在每一折训练数据中学习填补规则，
    再将规则应用到对应验证集，避免数据泄漏。
    """

    def __init__(self):
        self.none_category_columns = [
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

        self.zero_numeric_columns = [
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

        self.mode_fill_columns = [
            "MSZoning",
            "Utilities",
            "Exterior1st",
            "Exterior2nd",
            "Electrical",
            "KitchenQual",
            "SaleType",
        ]

        # 数字只是类别编号，不代表连续大小
        self.numeric_as_category_columns = [
            "MSSubClass",
        ]

    def fit(self, X, y=None):
        data = X.copy()

        # 将类别编号转换成字符串
        for column in self.numeric_as_category_columns:
            if column in data.columns:
                data[column] = data[column].astype(str)

        # 只从当前训练折学习社区临街长度中位数
        if (
            "Neighborhood" in data.columns
            and "LotFrontage" in data.columns
        ):
            self.neighborhood_frontage_medians_ = (
                data.groupby("Neighborhood")["LotFrontage"].median()
            )

            self.overall_frontage_median_ = (
                data["LotFrontage"].median()
            )
        else:
            self.neighborhood_frontage_medians_ = pd.Series(
                dtype=float
            )
            self.overall_frontage_median_ = 0.0

        # 学习指定类别字段的众数
        self.mode_fill_values_ = {}

        for column in self.mode_fill_columns:
            if column not in data.columns:
                continue

            mode_values = data[column].mode(dropna=True)

            if mode_values.empty:
                self.mode_fill_values_[column] = "Unknown"
            else:
                self.mode_fill_values_[column] = mode_values.iloc[0]

        # 学习剩余字段的兜底填充值
        self.numeric_fill_values_ = {}
        self.category_fill_values_ = {}

        for column in data.columns:

            if pd.api.types.is_numeric_dtype(data[column]):
                median_value = data[column].median()

                if pd.isna(median_value):
                    median_value = 0

                self.numeric_fill_values_[column] = median_value

            else:
                mode_values = data[column].mode(dropna=True)

                if mode_values.empty:
                    fill_value = "Unknown"
                else:
                    fill_value = mode_values.iloc[0]

                self.category_fill_values_[column] = fill_value

        return self

    def transform(self, X):
        data = X.copy()

        # 将编号型类别变量转成字符串
        for column in self.numeric_as_category_columns:
            if column in data.columns:
                data[column] = data[column].astype(str)

        # 类别缺失表示不存在相应设施
        for column in self.none_category_columns:
            if column in data.columns:
                data[column] = data[column].fillna(
                    "NoFeature"
                )

        # 数值缺失表示数量或面积为0
        for column in self.zero_numeric_columns:
            if column in data.columns:
                data[column] = data[column].fillna(0)

        # 按训练折中的社区中位数填充 LotFrontage
        if (
            "Neighborhood" in data.columns
            and "LotFrontage" in data.columns
        ):
            mapped_medians = data["Neighborhood"].map(
                self.neighborhood_frontage_medians_
            )

            data["LotFrontage"] = (
                data["LotFrontage"]
                .fillna(mapped_medians)
                .fillna(self.overall_frontage_median_)
            )

        # 使用训练折众数填充少量随机缺失类别
        for column, fill_value in self.mode_fill_values_.items():
            if column in data.columns:
                data[column] = data[column].fillna(fill_value)

        # Functional 缺失通常表示典型功能
        if "Functional" in data.columns:
            data["Functional"] = data["Functional"].fillna(
                "Typ"
            )

        # 兜底处理剩余数值型字段
        for column, fill_value in self.numeric_fill_values_.items():
            if column in data.columns:
                data[column] = data[column].fillna(fill_value)

        # 兜底处理剩余类别型字段
        for column, fill_value in self.category_fill_values_.items():
            if column in data.columns:
                data[column] = data[column].fillna(fill_value)

        return data


# ==========================================
# 3. 创建模型流水线函数
# ==========================================

def build_pipeline(model):
    """
    创建完整流水线：
    业务缺失值处理
    → 数值标准化
    → 类别独热编码
    → 回归模型
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                make_column_selector(
                    dtype_include=np.number
                ),
            ),
            (
                "categorical",
                categorical_pipeline,
                make_column_selector(
                    dtype_exclude=np.number
                ),
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            (
                "cleaner",
                HousePriceCleaner()
            ),
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            ),
        ]
    )

    return pipeline


# ==========================================
# 4. 读取原始数据
# ==========================================

train_path = RAW_DATA_DIR / "train.csv"
test_path = RAW_DATA_DIR / "test.csv"

if not train_path.exists():
    raise FileNotFoundError(
        f"没有找到训练集：{train_path}"
    )

if not test_path.exists():
    raise FileNotFoundError(
        f"没有找到测试集：{test_path}"
    )

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("原始训练集形状：", train.shape)
print("原始测试集形状：", test.shape)


# ==========================================
# 5. 删除已经确认的两个异常样本
# ==========================================

outlier_condition = (
    (train["GrLivArea"] > 4000)
    & (train["SalePrice"] < 300000)
)

print("识别出的异常样本数量：", outlier_condition.sum())

train = train.loc[
    ~outlier_condition
].copy()

train = train.reset_index(drop=True)

print("异常值处理后训练集形状：", train.shape)


# ==========================================
# 6. 拆分特征和目标变量
# ==========================================

# Id只是编号，不作为模型特征
X = train.drop(
    columns=["SalePrice", "Id"]
)

# 使用对数房价作为训练目标
y = np.log1p(
    train["SalePrice"]
)

test_ids = test["Id"].copy()

X_test = test.drop(
    columns=["Id"]
)

print("\n训练特征形状：", X.shape)
print("测试特征形状：", X_test.shape)


# ==========================================
# 7. 设置五折交叉验证
# ==========================================

cross_validation = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ==========================================
# 8. 定义模型
# ==========================================

# 最简单的参照模型：永远预测训练集均值
dummy_pipeline = build_pipeline(
    DummyRegressor(
        strategy="mean"
    )
)

# 第一个正式基线模型：岭回归
ridge_pipeline = build_pipeline(
    Ridge(
        alpha=10.0,
        solver="lsqr"
    )
)


# ==========================================
# 9. 定义模型评估函数
# ==========================================

def evaluate_model(
    model_name,
    pipeline,
    features,
    target
):
    """
    使用5折交叉验证计算每一折RMSE。
    """

    negative_mse_scores = cross_val_score(
        pipeline,
        features,
        target,
        cv=cross_validation,
        scoring="neg_mean_squared_error"
    )

    rmse_scores = np.sqrt(
        -negative_mse_scores
    )

    print(f"\n========== {model_name} ==========")

    for fold_number, score in enumerate(
        rmse_scores,
        start=1
    ):
        print(
            f"第{fold_number}折 RMSE：{score:.6f}"
        )

    print(
        f"平均 RMSE：{rmse_scores.mean():.6f}"
    )

    print(
        f"RMSE 标准差：{rmse_scores.std():.6f}"
    )

    return rmse_scores


# ==========================================
# 10. 评估 Dummy 基准模型
# ==========================================

dummy_scores = evaluate_model(
    model_name="Dummy Mean Baseline",
    pipeline=dummy_pipeline,
    features=X,
    target=y
)


# ==========================================
# 11. 评估岭回归模型
# ==========================================

ridge_scores = evaluate_model(
    model_name="Ridge Regression Baseline",
    pipeline=ridge_pipeline,
    features=X,
    target=y
)


# ==========================================
# 12. 保存交叉验证结果
# ==========================================

cv_results = pd.DataFrame(
    {
        "fold": [1, 2, 3, 4, 5],
        "dummy_rmse": dummy_scores,
        "ridge_rmse": ridge_scores,
    }
)

summary_row = pd.DataFrame(
    {
        "fold": ["mean"],
        "dummy_rmse": [dummy_scores.mean()],
        "ridge_rmse": [ridge_scores.mean()],
    }
)

cv_results_with_mean = pd.concat(
    [
        cv_results,
        summary_row
    ],
    ignore_index=True
)

cv_result_path = (
    TABLE_DIR / "baseline_cv_results.csv"
)

cv_results_with_mean.to_csv(
    cv_result_path,
    index=False
)

print("\n交叉验证结果已保存：")
print(cv_result_path)


# ==========================================
# 13. 使用全部训练数据训练岭回归
# ==========================================

print("\n开始使用全部训练数据训练岭回归……")

ridge_pipeline.fit(
    X,
    y
)

print("完整模型训练完成。")


# ==========================================
# 14. 保存训练好的模型
# ==========================================

model_path = (
    MODEL_DIR / "ridge_baseline.joblib"
)

joblib.dump(
    ridge_pipeline,
    model_path
)

print("\n模型已保存：")
print(model_path)


# ==========================================
# 15. 预测Kaggle测试集
# ==========================================

test_log_predictions = ridge_pipeline.predict(
    X_test
)

# 从对数房价恢复为实际房价
test_predictions = np.expm1(
    test_log_predictions
)

# 防止出现负数预测
test_predictions = np.maximum(
    test_predictions,
    0
)


# ==========================================
# 16. 生成提交文件
# ==========================================

submission = pd.DataFrame(
    {
        "Id": test_ids,
        "SalePrice": test_predictions,
    }
)

submission_path = (
    SUBMISSION_DIR
    / "ridge_baseline_submission.csv"
)

submission.to_csv(
    submission_path,
    index=False
)

print("\n提交文件已生成：")
print(submission_path)

print("\n提交文件形状：")
print(submission.shape)

print("\n提交文件前5行：")
print(submission.head())

print("\n提交文件缺失值总数：")
print(submission.isna().sum().sum())

print("\n基线模型阶段完成。")