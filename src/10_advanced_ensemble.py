from itertools import product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


# ==========================================
# 1. 项目路径
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
SUBMISSION_DIR = PROJECT_ROOT / "outputs" / "submissions"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. RMSE
# ==========================================

def calculate_rmse(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> float:
    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )


# ==========================================
# 3. 特征工程
# ==========================================

def add_engineered_features(
    data: pd.DataFrame,
) -> pd.DataFrame:

    result = data.copy()

    def safe_sum(columns: list[str]) -> pd.Series:
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

    # 房屋总面积
    result["TotalSF"] = safe_sum(
        [
            "TotalBsmtSF",
            "1stFlrSF",
            "2ndFlrSF",
        ]
    )

    # 地上总面积
    result["TotalAboveGroundSF"] = safe_sum(
        [
            "1stFlrSF",
            "2ndFlrSF",
        ]
    )

    # 总卫生间数量
    result["TotalBathrooms"] = (
        result.get(
            "FullBath",
            pd.Series(0, index=result.index),
        ).fillna(0)
        + 0.5
        * result.get(
            "HalfBath",
            pd.Series(0, index=result.index),
        ).fillna(0)
        + result.get(
            "BsmtFullBath",
            pd.Series(0, index=result.index),
        ).fillna(0)
        + 0.5
        * result.get(
            "BsmtHalfBath",
            pd.Series(0, index=result.index),
        ).fillna(0)
    )

    # 总门廊、平台面积
    result["TotalPorchSF"] = safe_sum(
        [
            "OpenPorchSF",
            "EnclosedPorch",
            "3SsnPorch",
            "ScreenPorch",
            "WoodDeckSF",
        ]
    )

    # 房屋总室内相关面积
    result["TotalIndoorSF"] = safe_sum(
        [
            "GrLivArea",
            "TotalBsmtSF",
            "GarageArea",
        ]
    )

    # 出售时房龄
    if {
        "YrSold",
        "YearBuilt",
    }.issubset(result.columns):
        result["HouseAge"] = (
            result["YrSold"]
            - result["YearBuilt"]
        ).clip(lower=0)

    # 距离上次翻修的年数
    if {
        "YrSold",
        "YearRemodAdd",
    }.issubset(result.columns):
        result["RemodAge"] = (
            result["YrSold"]
            - result["YearRemodAdd"]
        ).clip(lower=0)

    # 车库年龄
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

    # 是否翻修过
    if {
        "YearBuilt",
        "YearRemodAdd",
    }.issubset(result.columns):
        result["IsRemodeled"] = (
            result["YearRemodAdd"]
            != result["YearBuilt"]
        ).astype(int)

    # 是否有相关设施
    facility_columns = {
        "HasBasement": "TotalBsmtSF",
        "HasGarage": "GarageArea",
        "HasFireplace": "Fireplaces",
        "HasPool": "PoolArea",
        "HasSecondFloor": "2ndFlrSF",
        "HasWoodDeck": "WoodDeckSF",
        "HasOpenPorch": "OpenPorchSF",
    }

    for new_column, source_column in facility_columns.items():
        if source_column in result.columns:
            result[new_column] = (
                result[source_column]
                .fillna(0)
                .gt(0)
                .astype(int)
            )

    # 总设施数量
    result["TotalFacilityCount"] = safe_sum(
        [
            column
            for column in facility_columns
            if column in result.columns
        ]
    )

    # 质量和面积交互
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
        "OverallQual",
        "YearBuilt",
    }.issubset(result.columns):
        result["Quality_YearBuilt"] = (
            result["OverallQual"]
            * result["YearBuilt"]
        )

    if {
        "GarageCars",
        "GarageArea",
    }.issubset(result.columns):
        result["GarageScore"] = (
            result["GarageCars"].fillna(0)
            * result["GarageArea"].fillna(0)
        )

    # 每个房间的平均居住面积
    if {
        "GrLivArea",
        "TotRmsAbvGrd",
    }.issubset(result.columns):
        room_count = (
            result["TotRmsAbvGrd"]
            .replace(0, np.nan)
        )

        result["AverageRoomArea"] = (
            result["GrLivArea"]
            / room_count
        ).fillna(0)

    # 数字形式保存，但本质是类别
    number_as_category_columns = [
        "MSSubClass",
        "MoSold",
        "YrSold",
    ]

    for column in number_as_category_columns:
        if column in result.columns:
            result[column] = (
                result[column]
                .fillna(-1)
                .astype(int)
                .astype(str)
            )

    return result


# ==========================================
# 4. 为CatBoost准备数据
# ==========================================

def prepare_catboost_data(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
):
    train_result = train_features.copy()
    test_result = test_features.copy()

    categorical_columns = (
        train_result.select_dtypes(
            include=[
                "object",
                "category",
            ]
        )
        .columns
        .tolist()
    )

    for column in categorical_columns:
        train_result[column] = (
            train_result[column]
            .fillna("Missing")
            .astype(str)
        )

        test_result[column] = (
            test_result[column]
            .fillna("Missing")
            .astype(str)
        )

    return (
        train_result,
        test_result,
        categorical_columns,
    )


# ==========================================
# 5. 为Ridge、XGBoost、LightGBM准备数据
# ==========================================

def prepare_one_hot_data(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
):
    train_result = train_features.copy()
    test_result = test_features.copy()

    categorical_columns = (
        train_result.select_dtypes(
            include=[
                "object",
                "category",
            ]
        )
        .columns
        .tolist()
    )

    numeric_columns = [
        column
        for column in train_result.columns
        if column not in categorical_columns
    ]

    # 类别缺失
    for column in categorical_columns:
        train_result[column] = (
            train_result[column]
            .fillna("Missing")
            .astype(str)
        )

        test_result[column] = (
            test_result[column]
            .fillna("Missing")
            .astype(str)
        )

    # 数值字段仅使用训练集的中位数
    for column in numeric_columns:
        median_value = train_result[column].median()

        if pd.isna(median_value):
            median_value = 0

        train_result[column] = (
            train_result[column]
            .fillna(median_value)
        )

        test_result[column] = (
            test_result[column]
            .fillna(median_value)
        )

    # 对明显右偏且非负的数值字段做log1p
    skewness = (
        train_result[numeric_columns]
        .skew()
        .abs()
    )

    skewed_columns = (
        skewness[skewness > 0.75]
        .index
        .tolist()
    )

    safe_skewed_columns = []

    for column in skewed_columns:
        train_minimum = train_result[column].min()
        test_minimum = test_result[column].min()

        if train_minimum >= 0 and test_minimum >= 0:
            safe_skewed_columns.append(column)

    train_result[safe_skewed_columns] = np.log1p(
        train_result[safe_skewed_columns]
    )

    test_result[safe_skewed_columns] = np.log1p(
        test_result[safe_skewed_columns]
    )

    # 合并只是为了保证独热编码后的列完全一致
    all_features = pd.concat(
        [
            train_result,
            test_result,
        ],
        axis=0,
        ignore_index=True,
    )

    all_encoded = pd.get_dummies(
        all_features,
        dummy_na=False,
        dtype=np.float32,
    )

    train_encoded = (
        all_encoded
        .iloc[:len(train_result)]
        .copy()
        .reset_index(drop=True)
    )

    test_encoded = (
        all_encoded
        .iloc[len(train_result):]
        .copy()
        .reset_index(drop=True)
    )

    return (
        train_encoded,
        test_encoded,
        safe_skewed_columns,
    )


# ==========================================
# 6. 主程序
# ==========================================

def main() -> None:

    train = pd.read_csv(
        RAW_DATA_DIR / "train.csv"
    )

    test = pd.read_csv(
        RAW_DATA_DIR / "test.csv"
    )

    print("原始训练集：", train.shape)
    print("原始测试集：", test.shape)

    # 删除两个已经确认的异常点
    outlier_condition = (
        (train["GrLivArea"] > 4000)
        & (train["SalePrice"] < 300000)
    )

    print(
        "删除异常样本：",
        int(outlier_condition.sum()),
    )

    train = (
        train.loc[~outlier_condition]
        .copy()
        .reset_index(drop=True)
    )

    test_ids = test["Id"].copy()

    y = np.log1p(
        train["SalePrice"]
    ).reset_index(drop=True)

    X = train.drop(
        columns=[
            "Id",
            "SalePrice",
        ]
    )

    X_test = test.drop(
        columns=["Id"]
    )

    X = add_engineered_features(X)
    X_test = add_engineered_features(X_test)

    print("特征工程后训练集：", X.shape)
    print("特征工程后测试集：", X_test.shape)

    # CatBoost版本数据
    (
        X_cat,
        X_test_cat,
        categorical_columns,
    ) = prepare_catboost_data(
        X,
        X_test,
    )

    # 独热编码版本数据
    (
        X_encoded,
        X_test_encoded,
        skewed_columns,
    ) = prepare_one_hot_data(
        X,
        X_test,
    )

    print("独热编码后特征数量：", X_encoded.shape[1])
    print("log1p转换数值特征数量：", len(skewed_columns))

    # --------------------------------------
    # 两组五折，共训练10个划分
    # --------------------------------------

    random_seeds = [
        42,
        2026,
    ]

    model_names = [
        "Ridge",
        "XGBoost",
        "LightGBM",
        "CatBoost",
    ]

    oof_predictions = {
        name: np.zeros(len(X))
        for name in model_names
    }

    test_predictions = {
        name: np.zeros(len(X_test))
        for name in model_names
    }

    validation_counts = np.zeros(
        len(X),
        dtype=int,
    )

    fold_results = []

    total_fold_count = (
        len(random_seeds)
        * 5
    )

    current_fold = 0

    for seed in random_seeds:

        cross_validation = KFold(
            n_splits=5,
            shuffle=True,
            random_state=seed,
        )

        for fold_number, (
            train_indices,
            validation_indices,
        ) in enumerate(
            cross_validation.split(X),
            start=1,
        ):

            current_fold += 1

            print(
                f"\n========== Seed {seed} "
                f"Fold {fold_number} "
                f"({current_fold}/{total_fold_count}) =========="
            )

            y_train = y.iloc[train_indices]
            y_valid = y.iloc[validation_indices]

            X_train_encoded = X_encoded.iloc[
                train_indices
            ]

            X_valid_encoded = X_encoded.iloc[
                validation_indices
            ]

            X_train_cat = X_cat.iloc[
                train_indices
            ]

            X_valid_cat = X_cat.iloc[
                validation_indices
            ]

            validation_counts[
                validation_indices
            ] += 1

            # ==================================
            # Ridge
            # ==================================

            ridge_model = Pipeline(
                steps=[
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "model",
                        RidgeCV(
                            alphas=[
                                5.0,
                                10.0,
                                15.0,
                                20.0,
                                30.0,
                                50.0,
                            ],
                            cv=5,
                            scoring="neg_mean_squared_error",
                        ),
                    ),
                ]
            )

            ridge_model.fit(
                X_train_encoded,
                y_train,
            )

            ridge_valid_predictions = (
                ridge_model.predict(
                    X_valid_encoded
                )
            )

            ridge_test_predictions = (
                ridge_model.predict(
                    X_test_encoded
                )
            )

            # ==================================
            # XGBoost
            # ==================================

            xgb_model = XGBRegressor(
                n_estimators=6000,
                learning_rate=0.015,
                max_depth=3,
                min_child_weight=1,
                subsample=0.80,
                colsample_bytree=0.80,
                reg_alpha=0.001,
                reg_lambda=1.0,
                gamma=0,
                objective="reg:squarederror",
                eval_metric="rmse",
                early_stopping_rounds=250,
                tree_method="hist",
                random_state=seed + fold_number,
                n_jobs=-1,
            )

            xgb_model.fit(
                X_train_encoded,
                y_train,
                eval_set=[
                    (
                        X_valid_encoded,
                        y_valid,
                    )
                ],
                verbose=False,
            )

            xgb_valid_predictions = (
                xgb_model.predict(
                    X_valid_encoded
                )
            )

            xgb_test_predictions = (
                xgb_model.predict(
                    X_test_encoded
                )
            )

            # ==================================
            # LightGBM
            # ==================================

            lgb_model = LGBMRegressor(
                objective="regression",
                n_estimators=6000,
                learning_rate=0.01,
                num_leaves=20,
                max_depth=-1,
                min_child_samples=20,
                subsample=0.80,
                subsample_freq=1,
                colsample_bytree=0.80,
                reg_alpha=0.01,
                reg_lambda=0.05,
                random_state=seed + fold_number,
                n_jobs=-1,
                verbosity=-1,
            )

            lgb_model.fit(
                X_train_encoded,
                y_train,
                eval_set=[
                    (
                        X_valid_encoded,
                        y_valid,
                    )
                ],
                eval_metric="rmse",
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=250,
                        verbose=False,
                    )
                ],
            )

            lgb_valid_predictions = (
                lgb_model.predict(
                    X_valid_encoded,
                    num_iteration=(
                        lgb_model.best_iteration_
                    ),
                )
            )

            lgb_test_predictions = (
                lgb_model.predict(
                    X_test_encoded,
                    num_iteration=(
                        lgb_model.best_iteration_
                    ),
                )
            )

            # ==================================
            # CatBoost
            # ==================================

            cat_model = CatBoostRegressor(
                iterations=6000,
                learning_rate=0.015,
                depth=6,
                loss_function="RMSE",
                eval_metric="RMSE",
                l2_leaf_reg=5,
                random_strength=0.4,
                bagging_temperature=0.2,
                random_seed=seed + fold_number,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
            )

            cat_model.fit(
                X_train_cat,
                y_train,
                cat_features=categorical_columns,
                eval_set=(
                    X_valid_cat,
                    y_valid,
                ),
                early_stopping_rounds=250,
                use_best_model=True,
                verbose=False,
            )

            cat_valid_predictions = (
                cat_model.predict(
                    X_valid_cat
                )
            )

            cat_test_predictions = (
                cat_model.predict(
                    X_test_cat
                )
            )

            # ==================================
            # 保存本折预测
            # ==================================

            fold_model_predictions = {
                "Ridge": (
                    ridge_valid_predictions,
                    ridge_test_predictions,
                ),
                "XGBoost": (
                    xgb_valid_predictions,
                    xgb_test_predictions,
                ),
                "LightGBM": (
                    lgb_valid_predictions,
                    lgb_test_predictions,
                ),
                "CatBoost": (
                    cat_valid_predictions,
                    cat_test_predictions,
                ),
            }

            for model_name, (
                valid_predictions,
                current_test_predictions,
            ) in fold_model_predictions.items():

                oof_predictions[
                    model_name
                ][validation_indices] += (
                    valid_predictions
                )

                test_predictions[
                    model_name
                ] += (
                    current_test_predictions
                    / total_fold_count
                )

                fold_rmse = calculate_rmse(
                    y_valid,
                    valid_predictions,
                )

                print(
                    f"{model_name:<10} "
                    f"RMSE：{fold_rmse:.6f}"
                )

                fold_results.append(
                    {
                        "seed": seed,
                        "fold": fold_number,
                        "model": model_name,
                        "rmse": fold_rmse,
                    }
                )

    # 每个训练样本被验证两次，所以取平均
    for model_name in model_names:
        oof_predictions[model_name] = (
            oof_predictions[model_name]
            / validation_counts
        )

    # ==========================================
    # 7. 单模型OOF结果
    # ==========================================

    comparison_rows = []

    print("\n========== 重复五折OOF结果 ==========")

    for model_name in model_names:
        current_rmse = calculate_rmse(
            y,
            oof_predictions[model_name],
        )

        print(
            f"{model_name:<10} "
            f"OOF RMSE：{current_rmse:.6f}"
        )

        comparison_rows.append(
            {
                "model": model_name,
                "oof_rmse": current_rmse,
            }
        )

    # ==========================================
    # 8. 搜索四模型融合权重
    # ==========================================

    prediction_matrix = np.column_stack(
        [
            oof_predictions["Ridge"],
            oof_predictions["XGBoost"],
            oof_predictions["LightGBM"],
            oof_predictions["CatBoost"],
        ]
    )

    test_prediction_matrix = np.column_stack(
        [
            test_predictions["Ridge"],
            test_predictions["XGBoost"],
            test_predictions["LightGBM"],
            test_predictions["CatBoost"],
        ]
    )

    weight_values = np.arange(
        0,
        1.01,
        0.10,
    )

    best_rmse = float("inf")
    best_weights = None

    for ridge_weight, xgb_weight, lgb_weight in product(
        weight_values,
        repeat=3,
    ):
        cat_weight = (
            1.0
            - ridge_weight
            - xgb_weight
            - lgb_weight
        )

        if cat_weight < -1e-9:
            continue

        if cat_weight > 1.0 + 1e-9:
            continue

        weights = np.array(
            [
                ridge_weight,
                xgb_weight,
                lgb_weight,
                cat_weight,
            ]
        )

        blended_predictions = (
            prediction_matrix
            @ weights
        )

        current_rmse = calculate_rmse(
            y,
            blended_predictions,
        )

        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_weights = weights.copy()

    if best_weights is None:
        raise RuntimeError(
            "没有找到有效融合权重。"
        )

    print("\n========== 最佳融合权重 ==========")

    for model_name, weight in zip(
        model_names,
        best_weights,
    ):
        print(
            f"{model_name:<10}：{weight:.2f}"
        )

    print(
        f"最佳重复OOF融合RMSE：{best_rmse:.6f}"
    )

    # ==========================================
    # 9. 更保守的融合权重
    # ==========================================

    # 将最优权重向四模型等权重收缩20%，
    # 减少权重对本地OOF的过拟合。
    equal_weights = np.full(
        4,
        0.25,
    )

    conservative_weights = (
        0.80 * best_weights
        + 0.20 * equal_weights
    )

    conservative_oof_predictions = (
        prediction_matrix
        @ conservative_weights
    )

    conservative_rmse = calculate_rmse(
        y,
        conservative_oof_predictions,
    )

    print("\n========== 保守融合权重 ==========")

    for model_name, weight in zip(
        model_names,
        conservative_weights,
    ):
        print(
            f"{model_name:<10}：{weight:.3f}"
        )

    print(
        f"保守融合OOF RMSE：{conservative_rmse:.6f}"
    )

    # ==========================================
    # 10. 生成测试集预测
    # ==========================================

    optimized_test_log_predictions = (
        test_prediction_matrix
        @ best_weights
    )

    conservative_test_log_predictions = (
        test_prediction_matrix
        @ conservative_weights
    )

    optimized_price_predictions = np.expm1(
        optimized_test_log_predictions
    )

    conservative_price_predictions = np.expm1(
        conservative_test_log_predictions
    )

    optimized_price_predictions = np.maximum(
        optimized_price_predictions,
        0,
    )

    conservative_price_predictions = np.maximum(
        conservative_price_predictions,
        0,
    )

    # ==========================================
    # 11. 保存提交文件
    # ==========================================

    optimized_submission = pd.DataFrame(
        {
            "Id": test_ids,
            "SalePrice": optimized_price_predictions,
        }
    )

    conservative_submission = pd.DataFrame(
        {
            "Id": test_ids,
            "SalePrice": conservative_price_predictions,
        }
    )

    optimized_path = (
        SUBMISSION_DIR
        / "advanced_four_model_blend.csv"
    )

    conservative_path = (
        SUBMISSION_DIR
        / "advanced_conservative_blend.csv"
    )

    optimized_submission.to_csv(
        optimized_path,
        index=False,
    )

    conservative_submission.to_csv(
        conservative_path,
        index=False,
    )

    # ==========================================
    # 12. 保存结果表
    # ==========================================

    comparison_rows.extend(
        [
            {
                "model": "OptimizedFourModelBlend",
                "oof_rmse": best_rmse,
            },
            {
                "model": "ConservativeFourModelBlend",
                "oof_rmse": conservative_rmse,
            },
        ]
    )

    pd.DataFrame(
        comparison_rows
    ).sort_values(
        by="oof_rmse"
    ).to_csv(
        TABLE_DIR
        / "advanced_model_comparison.csv",
        index=False,
    )

    pd.DataFrame(
        fold_results
    ).to_csv(
        TABLE_DIR
        / "advanced_fold_results.csv",
        index=False,
    )

    weights_table = pd.DataFrame(
        {
            "model": model_names,
            "optimized_weight": best_weights,
            "conservative_weight": conservative_weights,
        }
    )

    weights_table.to_csv(
        TABLE_DIR
        / "advanced_blend_weights.csv",
        index=False,
    )

    print("\n========== 输出文件 ==========")
    print("最优OOF融合：", optimized_path)
    print("保守融合：", conservative_path)

    print(
        "\n最优提交形状：",
        optimized_submission.shape,
    )

    print(
        "最优提交缺失值：",
        int(
            optimized_submission
            .isna()
            .sum()
            .sum()
        ),
    )

    print("\n高级融合训练完成。")


if __name__ == "__main__":
    main()