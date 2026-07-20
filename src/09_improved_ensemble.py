from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    KFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==========================================
# 1. 项目路径
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
# 2. RMSE计算函数
# ==========================================

def calculate_rmse(
    true_values: pd.Series | np.ndarray,
    predictions: np.ndarray,
) -> float:
    """
    计算对数房价上的RMSE。
    """

    mse = mean_squared_error(
        true_values,
        predictions,
    )

    return float(np.sqrt(mse))


# ==========================================
# 3. 特征工程
# ==========================================

def add_engineered_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    根据已有房屋字段构造更有业务含义的组合特征。
    """

    result = data.copy()

    def sum_columns(
        columns: list[str],
    ) -> pd.Series:
        """
        将实际存在的指定字段相加。
        缺失值按0参与求和。
        """

        existing_columns = [
            column
            for column in columns
            if column in result.columns
        ]

        if not existing_columns:
            return pd.Series(
                0.0,
                index=result.index,
            )

        return (
            result[existing_columns]
            .fillna(0)
            .sum(axis=1)
        )

    # --------------------------------------
    # 房屋总使用面积
    # --------------------------------------

    result["TotalSF"] = sum_columns(
        [
            "TotalBsmtSF",
            "1stFlrSF",
            "2ndFlrSF",
        ]
    )

    # --------------------------------------
    # 总卫生间数量
    # 半卫生间按0.5个计算
    # --------------------------------------

    full_bath = result.get(
        "FullBath",
        pd.Series(0, index=result.index),
    ).fillna(0)

    half_bath = result.get(
        "HalfBath",
        pd.Series(0, index=result.index),
    ).fillna(0)

    basement_full_bath = result.get(
        "BsmtFullBath",
        pd.Series(0, index=result.index),
    ).fillna(0)

    basement_half_bath = result.get(
        "BsmtHalfBath",
        pd.Series(0, index=result.index),
    ).fillna(0)

    result["TotalBathrooms"] = (
        full_bath
        + 0.5 * half_bath
        + basement_full_bath
        + 0.5 * basement_half_bath
    )

    # --------------------------------------
    # 门廊、露台和木质平台总面积
    # --------------------------------------

    result["TotalPorchSF"] = sum_columns(
        [
            "OpenPorchSF",
            "3SsnPorch",
            "EnclosedPorch",
            "ScreenPorch",
            "WoodDeckSF",
        ]
    )

    # --------------------------------------
    # 房龄与翻修年限
    # --------------------------------------

    if {
        "YrSold",
        "YearBuilt",
    }.issubset(result.columns):
        result["HouseAge"] = (
            result["YrSold"]
            - result["YearBuilt"]
        ).clip(lower=0)

    if {
        "YrSold",
        "YearRemodAdd",
    }.issubset(result.columns):
        result["RemodAge"] = (
            result["YrSold"]
            - result["YearRemodAdd"]
        ).clip(lower=0)

    if {
        "YrSold",
        "GarageYrBlt",
    }.issubset(result.columns):
        result["GarageAge"] = np.where(
            result["GarageYrBlt"].fillna(0) > 0,
            (
                result["YrSold"]
                - result["GarageYrBlt"]
            ).clip(lower=0),
            0,
        )

    # --------------------------------------
    # 是否具有对应设施
    # --------------------------------------

    if "TotalBsmtSF" in result.columns:
        result["HasBasement"] = (
            result["TotalBsmtSF"]
            .fillna(0)
            > 0
        ).astype(int)

    if "GarageArea" in result.columns:
        result["HasGarage"] = (
            result["GarageArea"]
            .fillna(0)
            > 0
        ).astype(int)

    if "Fireplaces" in result.columns:
        result["HasFireplace"] = (
            result["Fireplaces"]
            .fillna(0)
            > 0
        ).astype(int)

    if "PoolArea" in result.columns:
        result["HasPool"] = (
            result["PoolArea"]
            .fillna(0)
            > 0
        ).astype(int)

    if "2ndFlrSF" in result.columns:
        result["HasSecondFloor"] = (
            result["2ndFlrSF"]
            .fillna(0)
            > 0
        ).astype(int)

    # --------------------------------------
    # 质量与面积交互特征
    # --------------------------------------

    if {
        "OverallQual",
        "GrLivArea",
    }.issubset(result.columns):
        result["OverallQual_GrLivArea"] = (
            result["OverallQual"]
            * result["GrLivArea"]
        )

    if {
        "OverallQual",
        "TotalSF",
    }.issubset(result.columns):
        result["OverallQual_TotalSF"] = (
            result["OverallQual"]
            * result["TotalSF"]
        )

    if {
        "GarageCars",
        "GarageArea",
    }.issubset(result.columns):
        result["GarageScore"] = (
            result["GarageCars"].fillna(0)
            * result["GarageArea"].fillna(0)
        )

    # --------------------------------------
    # 数字编码但实际属于类别的字段
    # --------------------------------------

    categorical_number_columns = [
        "MSSubClass",
        "MoSold",
    ]

    for column in categorical_number_columns:
        if column in result.columns:
            result[column] = (
                result[column]
                .fillna(-1)
                .astype(int)
                .astype(str)
            )

    return result


# ==========================================
# 4. 建立Ridge流水线
# ==========================================

def build_ridge_pipeline(
    features: pd.DataFrame,
    alpha: float,
) -> Pipeline:
    """
    建立数值处理、类别编码和Ridge模型流水线。
    """

    numeric_columns = (
        features
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    categorical_columns = [
        column
        for column in features.columns
        if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ]
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                Ridge(
                    alpha=alpha,
                    solver="lsqr",
                ),
            ),
        ]
    )


# ==========================================
# 5. 主程序
# ==========================================

def main() -> None:

    # --------------------------------------
    # 读取原始数据
    # --------------------------------------

    train_path = RAW_DATA_DIR / "train.csv"
    test_path = RAW_DATA_DIR / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"没有找到训练数据：{train_path}"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"没有找到测试数据：{test_path}"
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print("原始训练集形状：", train.shape)
    print("原始测试集形状：", test.shape)

    # --------------------------------------
    # 删除已经确认的两个异常样本
    # --------------------------------------

    outlier_condition = (
        (train["GrLivArea"] > 4000)
        & (train["SalePrice"] < 300000)
    )

    print(
        "删除的异常样本数量：",
        int(outlier_condition.sum()),
    )

    train = (
        train.loc[~outlier_condition]
        .copy()
        .reset_index(drop=True)
    )

    # --------------------------------------
    # 保存测试集Id
    # --------------------------------------

    test_ids = test["Id"].copy()

    # --------------------------------------
    # 拆分目标和特征
    # --------------------------------------

    y = np.log1p(
        train["SalePrice"]
    )

    X = train.drop(
        columns=[
            "SalePrice",
            "Id",
        ]
    )

    X_test = test.drop(
        columns=["Id"]
    )

    # --------------------------------------
    # 增加组合特征
    # --------------------------------------

    X = add_engineered_features(X)
    X_test = add_engineered_features(X_test)

    if X.columns.tolist() != X_test.columns.tolist():
        raise ValueError(
            "训练集和测试集特征列不一致。"
        )

    print("特征工程后训练集：", X.shape)
    print("特征工程后测试集：", X_test.shape)

    # --------------------------------------
    # 找到类别字段
    # --------------------------------------

    categorical_columns = (
        X.select_dtypes(
            include=[
                "object",
                "category",
            ]
        )
        .columns
        .tolist()
    )

    # CatBoost类别字段不能保留普通空值，
    # 将其统一转换成明确字符串
    for column in categorical_columns:
        X[column] = (
            X[column]
            .fillna("Missing")
            .astype(str)
        )

        X_test[column] = (
            X_test[column]
            .fillna("Missing")
            .astype(str)
        )

    print(
        "类别型特征数量：",
        len(categorical_columns),
    )

    # --------------------------------------
    # 五折交叉验证
    # --------------------------------------

    n_splits = 5

    cross_validation = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    # ======================================
    # 6. CatBoost五折训练
    # ======================================

    catboost_oof_predictions = np.zeros(
        len(X)
    )

    catboost_test_log_predictions = np.zeros(
        len(X_test)
    )

    catboost_fold_results = []

    print(
        "\n========== CatBoost五折训练 =========="
    )

    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        cross_validation.split(X),
        start=1,
    ):

        print(
            f"\n开始训练第{fold_number}折……"
        )

        X_train_fold = X.iloc[
            train_indices
        ]

        X_validation_fold = X.iloc[
            validation_indices
        ]

        y_train_fold = y.iloc[
            train_indices
        ]

        y_validation_fold = y.iloc[
            validation_indices
        ]

        catboost_model = CatBoostRegressor(
            iterations=5000,
            learning_rate=0.02,
            depth=6,
            loss_function="RMSE",
            eval_metric="RMSE",
            l2_leaf_reg=5,
            random_strength=0.5,
            bagging_temperature=0.3,
            random_seed=42 + fold_number,
            verbose=250,
            allow_writing_files=False,
            thread_count=-1,
        )

        catboost_model.fit(
            X_train_fold,
            y_train_fold,
            cat_features=categorical_columns,
            eval_set=(
                X_validation_fold,
                y_validation_fold,
            ),
            early_stopping_rounds=300,
            use_best_model=True,
        )

        validation_predictions = (
            catboost_model.predict(
                X_validation_fold
            )
        )

        catboost_oof_predictions[
            validation_indices
        ] = validation_predictions

        catboost_test_log_predictions += (
            catboost_model.predict(X_test)
            / n_splits
        )

        fold_rmse = calculate_rmse(
            y_validation_fold,
            validation_predictions,
        )

        best_iteration = (
            catboost_model.get_best_iteration()
        )

        print(
            f"第{fold_number}折 RMSE："
            f"{fold_rmse:.6f}"
        )

        print(
            f"第{fold_number}折最佳迭代次数："
            f"{best_iteration}"
        )

        catboost_fold_results.append(
            {
                "fold": fold_number,
                "rmse": fold_rmse,
                "best_iteration": best_iteration,
            }
        )

        model_path = (
            MODEL_DIR
            / f"catboost_fold_{fold_number}.cbm"
        )

        catboost_model.save_model(
            str(model_path)
        )

    catboost_oof_rmse = calculate_rmse(
        y,
        catboost_oof_predictions,
    )

    print(
        "\nCatBoost整体OOF RMSE："
        f"{catboost_oof_rmse:.6f}"
    )

    # 保存CatBoost折数结果
    catboost_results_path = (
        TABLE_DIR
        / "catboost_cv_results.csv"
    )

    pd.DataFrame(
        catboost_fold_results
    ).to_csv(
        catboost_results_path,
        index=False,
    )

    # ======================================
    # 7. 搜索更合适的Ridge alpha
    # ======================================

    print(
        "\n========== 搜索Ridge alpha =========="
    )

    alpha_candidates = [
        1.0,
        5.0,
        10.0,
        15.0,
        20.0,
        30.0,
        50.0,
    ]

    ridge_search_results = []

    best_alpha = None
    best_ridge_rmse = float("inf")

    for alpha in alpha_candidates:

        ridge_pipeline = build_ridge_pipeline(
            X,
            alpha=alpha,
        )

        negative_mse_scores = cross_val_score(
            ridge_pipeline,
            X,
            y,
            cv=cross_validation,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )

        fold_rmse_scores = np.sqrt(
            -negative_mse_scores
        )

        mean_rmse = float(
            fold_rmse_scores.mean()
        )

        print(
            f"alpha={alpha:<5} "
            f"平均RMSE={mean_rmse:.6f}"
        )

        ridge_search_results.append(
            {
                "alpha": alpha,
                "mean_rmse": mean_rmse,
                "rmse_std": float(
                    fold_rmse_scores.std()
                ),
            }
        )

        if mean_rmse < best_ridge_rmse:
            best_ridge_rmse = mean_rmse
            best_alpha = alpha

    print(
        "\n最佳Ridge alpha：",
        best_alpha,
    )

    print(
        "最佳Ridge平均RMSE："
        f"{best_ridge_rmse:.6f}"
    )

    pd.DataFrame(
        ridge_search_results
    ).to_csv(
        TABLE_DIR
        / "improved_ridge_alpha_search.csv",
        index=False,
    )

    # ======================================
    # 8. 生成Ridge OOF预测
    # ======================================

    best_ridge_pipeline = build_ridge_pipeline(
        X,
        alpha=float(best_alpha),
    )

    ridge_oof_predictions = cross_val_predict(
        best_ridge_pipeline,
        X,
        y,
        cv=cross_validation,
        n_jobs=-1,
    )

    ridge_oof_rmse = calculate_rmse(
        y,
        ridge_oof_predictions,
    )

    print(
        "\nRidge整体OOF RMSE："
        f"{ridge_oof_rmse:.6f}"
    )

    # 使用全部训练数据训练Ridge
    best_ridge_pipeline.fit(
        X,
        y,
    )

    ridge_test_log_predictions = (
        best_ridge_pipeline.predict(
            X_test
        )
    )

    ridge_model_path = (
        MODEL_DIR
        / "ridge_feature_engineered.joblib"
    )

    joblib.dump(
        best_ridge_pipeline,
        ridge_model_path,
    )

    # ======================================
    # 9. 自动搜索融合权重
    # ======================================

    print(
        "\n========== 搜索融合权重 =========="
    )

    blend_results = []

    best_catboost_weight = None
    best_blend_rmse = float("inf")

    # weight表示CatBoost权重
    for weight in np.arange(
        0.0,
        1.01,
        0.05,
    ):

        blended_oof_predictions = (
            weight
            * catboost_oof_predictions
            + (1.0 - weight)
            * ridge_oof_predictions
        )

        blend_rmse = calculate_rmse(
            y,
            blended_oof_predictions,
        )

        blend_results.append(
            {
                "catboost_weight": weight,
                "ridge_weight": 1.0 - weight,
                "oof_rmse": blend_rmse,
            }
        )

        print(
            f"CatBoost={weight:.2f}，"
            f"Ridge={1.0 - weight:.2f}，"
            f"RMSE={blend_rmse:.6f}"
        )

        if blend_rmse < best_blend_rmse:
            best_blend_rmse = blend_rmse
            best_catboost_weight = weight

    best_ridge_weight = (
        1.0
        - float(best_catboost_weight)
    )

    print(
        "\n最佳CatBoost权重："
        f"{best_catboost_weight:.2f}"
    )

    print(
        "最佳Ridge权重："
        f"{best_ridge_weight:.2f}"
    )

    print(
        "融合模型OOF RMSE："
        f"{best_blend_rmse:.6f}"
    )

    pd.DataFrame(
        blend_results
    ).to_csv(
        TABLE_DIR
        / "blend_weight_search.csv",
        index=False,
    )

    # ======================================
    # 10. 生成最终预测
    # ======================================

    blended_test_log_predictions = (
        float(best_catboost_weight)
        * catboost_test_log_predictions
        + best_ridge_weight
        * ridge_test_log_predictions
    )

    catboost_price_predictions = np.expm1(
        catboost_test_log_predictions
    )

    ridge_price_predictions = np.expm1(
        ridge_test_log_predictions
    )

    blended_price_predictions = np.expm1(
        blended_test_log_predictions
    )

    catboost_price_predictions = np.maximum(
        catboost_price_predictions,
        0,
    )

    ridge_price_predictions = np.maximum(
        ridge_price_predictions,
        0,
    )

    blended_price_predictions = np.maximum(
        blended_price_predictions,
        0,
    )

    # ======================================
    # 11. 保存三个提交文件
    # ======================================

    catboost_submission = pd.DataFrame(
        {
            "Id": test_ids,
            "SalePrice": catboost_price_predictions,
        }
    )

    ridge_submission = pd.DataFrame(
        {
            "Id": test_ids,
            "SalePrice": ridge_price_predictions,
        }
    )

    blend_submission = pd.DataFrame(
        {
            "Id": test_ids,
            "SalePrice": blended_price_predictions,
        }
    )

    catboost_submission_path = (
        SUBMISSION_DIR
        / "catboost_submission.csv"
    )

    ridge_submission_path = (
        SUBMISSION_DIR
        / "ridge_feature_engineered_submission.csv"
    )

    blend_submission_path = (
        SUBMISSION_DIR
        / "improved_blend_submission.csv"
    )

    catboost_submission.to_csv(
        catboost_submission_path,
        index=False,
    )

    ridge_submission.to_csv(
        ridge_submission_path,
        index=False,
    )

    blend_submission.to_csv(
        blend_submission_path,
        index=False,
    )

    # ======================================
    # 12. 保存模型对比结果
    # ======================================

    model_comparison = pd.DataFrame(
        {
            "model": [
                "Ridge",
                "CatBoost",
                "Ridge_CatBoost_Blend",
            ],
            "oof_rmse": [
                ridge_oof_rmse,
                catboost_oof_rmse,
                best_blend_rmse,
            ],
        }
    ).sort_values(
        by="oof_rmse"
    )

    model_comparison.to_csv(
        TABLE_DIR
        / "improved_model_comparison.csv",
        index=False,
    )

    print(
        "\n========== 本地模型对比 =========="
    )

    print(
        model_comparison.to_string(
            index=False
        )
    )

    print(
        "\n最终推荐提交文件："
    )

    print(
        blend_submission_path
    )

    print(
        "\n提交文件形状：",
        blend_submission.shape,
    )

    print(
        "提交文件缺失值：",
        int(
            blend_submission
            .isna()
            .sum()
            .sum()
        ),
    )

    print(
        "\n改进模型训练完成。"
    )


if __name__ == "__main__":
    main()