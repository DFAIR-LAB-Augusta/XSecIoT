# src.FIRE.models

import argparse
import logging
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from lime.lime_tabular import LimeTabularExplainer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow import random as tfr
from tensorflow.keras.layers import Dense, Dropout, Input  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Input  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore

logger = logging.getLogger(__name__)
np.random.seed(42)
tfr.set_seed(42)


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Model training pipeline for FIRE (Step 2)")
    parser.add_argument("aggregated_file", type=str,
                        help="Path to the aggregated data CSV file")
    parser.add_argument(
        "--unsw",
        action="store_true",
        help="Use multiclass labels"
    )
    parser.add_argument(
        "--pca",
        action="store_true",
        help="Enable PCA for feature reduction"
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Enable SHAP for XAI"
    )
    parser.add_argument(
        "--lime",
        action="store_true",
        help="Enable LIME for XAI"
    )
    return parser.parse_args()


def _explain_with_lime(model, X_train, X_test, feature_names, class_names, outputPath, output_prefix="lime_plot") -> None:  # noqa: E501
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    exp = explainer.explain_instance(X_test[0], model.predict_proba)
    exp.save_to_file(os.path.join(
        outputPath, f"{output_prefix}_instance.html"))
    print(
        f"LIME explanation saved to {os.path.join(outputPath, f'{output_prefix}_instance.html')}")


def _explain_with_shap(model, X_sample, outputPath, feature_names, model_type="tree", output_prefix="shap_plot") -> None:  # noqa: E501
    """
    Generate and save SHAP explanations for a given model and data sample.
    model_type can be "tree" for XGBoost/RandomForest or "kernel" for others.
    """
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    else:
        explainer = shap.KernelExplainer(lambda x: model.predict_proba(x), X_sample)  # Nparray wrapped to avoid TypeErrors  # noqa: E501
        shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list) and len(shap_values) > 1:
        for i, class_shap_values in enumerate(shap_values):
            plt.figure()
            shap.summary_plot(class_shap_values, X_sample,
                              feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(
                outputPath, f"{output_prefix}_class{i}summary.png"))
            plt.close()
            print(
                f"SHAP summary for class {i} {os.path.join(outputPath, f'{output_prefix}_class{i}_summary.png')}")
    else:
        plt.figure()
        shap.summary_plot(shap_values, X_sample,
                          feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(outputPath, f"{output_prefix}_summary.png"))
        plt.close()
        print(
            f"SHAP summary saved to {os.path.join(outputPath, f'{output_prefix}_summary.png')}")


def run_feature_engineering(aggregated_file: str) -> tuple[StandardScaler, PCA, np.ndarray]:
    """
    Performs scaling and PCA on the aggregated data.
    Saves the scaler and PCA objects to the 'feature_engineering' folder.
    Returns the scaler, PCA object, and the PCA-transformed features.
    """
    data = pd.read_csv(aggregated_file)
    X = data.drop(columns=['BinLabel', 'Label', 'src_ip', 'dst_ip', 'start_time',
                        'end_time_x', 'end_time_y', 'time_diff', 'time_diff_seconds', 'Attack', 'start_time_x', 'start_time_y'], errors='ignore')  # noqa: E501

    if X.isna().any().any():
        X = X.fillna(X.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    dataset_name = os.path.basename(os.path.dirname(aggregated_file))
    fe_dir = os.path.join(os.getcwd(), "feature_engineering", dataset_name)
    os.makedirs(fe_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(fe_dir, 'scaler.pkl'))
    joblib.dump(pca, os.path.join(fe_dir, 'pca.pkl'))
    print("Scaler and PCA objects saved in", fe_dir)
    return scaler, pca, X_pca


def run_binary_classification(aggregated_file: str, isUNSW: bool, isPCA: bool) -> None:
    """
    Loads aggregated data, performs scaling and PCA, then trains and evaluates
    multiple binary classifiers using the 'BinLabel' column as the target.
    Trained models and transformation objects are saved in the 'binary_models' folder.
    """
    print(f"Agg Data Path: {aggregated_file}")  # caffeinate python3 -m FIRE.main datasets/CIC_UNSW/NF-CICIDS2018-v3.csv --unsw --window_size 3s --step_size 5s >> output/WS3_SS5/CICtest.txt  # noqa: E501
    data = pd.read_csv(aggregated_file)
    print(f"IsUNSW: {isUNSW}", file=sys.stderr, flush=True)

    # For UNSW runs, the dataset already has a binary 'Label' (Benign vs Attack)
    if isUNSW:
        data['BinLabel'] = data['Label']

    # Otherwise derive BinLabel if needed
    if 'BinLabel' not in data.columns and 'Label' in data.columns:
        data['BinLabel'] = data['Label'].apply(
            lambda x: 0 if x == 'Benign' else 1)
    X = data.drop(columns=['BinLabel', 'Label', 'src_ip', 'dst_ip', 'start_time',
                        'end_time_x', 'end_time_y', 'time_diff', 'time_diff_seconds', 'Attack', 'start_time_x', 'start_time_y'], errors='ignore')  # noqa: E501
    y = data['BinLabel']

    print("Checking for NaN values in features:", file=sys.stderr, flush=True)
    print(X.isna().sum())
    print("Any NaN in X:", X.isna().any().any(), file=sys.stderr, flush=True)
    if X.isna().any().any():
        print("Found NaN values, filling with column means.")
        X = X.fillna(X.mean())

    if isPCA:
        print("Starting PCA", file=sys.stderr, flush=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_scaled)

        print("PCA Explained Variance Ratio:", pca.explained_variance_ratio_)

        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y, test_size=0.2, random_state=42)
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]
    else:
        print("Starting Scaling (No PCA)", file=sys.stderr, flush=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42)
        feature_names = X.columns.tolist()

    # ------------------
    # Random Forest
    print("Starting Random Forest", file=sys.stderr, flush=True)
    rf = RandomForestClassifier(random_state=42)
    if isPCA:
        cv_scores = cross_val_score(
            rf, X_pca, y, cv=5, scoring='accuracy')  # type: ignore
    else:
        cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='accuracy')
    print(
        f"Random Forest CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("Random Forest Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=['Benign', 'Attack']))

    # ------------------
    # K-Nearest Neighbors
    print("Starting KNN", file=sys.stderr, flush=True)
    knn = KNeighborsClassifier(n_neighbors=5)
    if isPCA:
        cv_scores = cross_val_score(
            knn, X_pca, y, cv=5, scoring='accuracy')  # type: ignore
    else:
        cv_scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
    print(
        f"KNN CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    print("KNN Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_knn))
    print("KNN Classification Report:")
    print(classification_report(y_test, y_pred_knn,
          target_names=['Benign', 'Attack']))

    # ------------------
    # Decision Tree
    print("Starting DT", file=sys.stderr, flush=True)
    dt = DecisionTreeClassifier(random_state=42)
    if isPCA:
        cv_scores = cross_val_score(
            dt, X_pca, y, cv=5, scoring='accuracy')  # type: ignore
    else:
        cv_scores = cross_val_score(dt, X_scaled, y, cv=5, scoring='accuracy')
    print(
        f"Decision Tree CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    print("Decision Tree Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_dt))
    print("Decision Tree Classification Report:")
    print(classification_report(y_test, y_pred_dt,
          target_names=['Benign', 'Attack']))

    # ------------------
    # Support Vector Machine
    print("Starting SVM", file=sys.stderr, flush=True)
    svm = SVC(kernel='poly', C=1, random_state=0, probability=True)
    # svm = SVC(kernel='linear', C=1, random_state=0, probability=True) # Testing w/ liberal
    if isPCA:
        cv_scores = cross_val_score(
            svm, X_pca, y, cv=5, scoring='accuracy')  # type: ignore
    else:
        cv_scores = cross_val_score(svm, X_scaled, y, cv=5, scoring='accuracy')
    print(
        f"SVM CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    print("SVM Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_svm))
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred_svm,
          target_names=['Benign', 'Attack']))

    # ------------------
    # XGBoost
    print("Starting XGBoost", file=sys.stderr, flush=True)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=[
                         f"f_{i}" for i in range(X_train.shape[1])])
    params = {
        'objective': 'binary:logistic',
        'learning_rate': 0.1,
        'max_depth': 8,
        'random_state': 42
    }
    xgb_model = xgb.train(params, dtrain=dtrain)
    dtest = xgb.DMatrix(X_test, feature_names=[
                        f"f_{i}" for i in range(X_test.shape[1])])
    y_pred_prob = xgb_model.predict(dtest)
    y_pred_xgb = (y_pred_prob > 0.5).astype(int)
    print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
    print("XGBoost Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_xgb))
    print("XGBoost Classification Report:")
    print(classification_report(y_test, y_pred_xgb,
          target_names=['Benign', 'Attack']))

    # ------------------
    # Feedforward Neural Network
    print("Starting FNN", file=sys.stderr, flush=True)
    feedforward_model_bin = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    feedforward_model_bin.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    feedforward_model_bin.fit(
        X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    loss, acc = feedforward_model_bin.evaluate(X_test, y_test, verbose=0)
    print(f"Feedforward NN - Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    dataset_name = os.path.basename(os.path.dirname(aggregated_file))
    binary_models_dir = os.path.join(
        os.getcwd(), "binary_models", dataset_name)
    if not os.path.exists(binary_models_dir):
        os.makedirs(binary_models_dir)

    '''XAI Additions'''

    xai_output_dir = os.path.join(binary_models_dir, "xai")
    os.makedirs(xai_output_dir, exist_ok=True)

    class_names = ['Benign', 'Attack']

    _explain_with_shap(rf, X_test[:100], outputPath=xai_output_dir, model_type="tree", output_prefix="rf_binary", feature_names=feature_names)  # noqa: E501
    _explain_with_lime(rf, X_train, X_test, feature_names, class_names, outputPath=xai_output_dir, output_prefix="rf_binary")  # noqa: E501

    xgb_sklearn_model = xgb.XGBClassifier()
    # needed because current xgb_model is native API
    xgb_sklearn_model.fit(X_train, y_train)
    _explain_with_shap(xgb_sklearn_model, X_test[:100], outputPath=xai_output_dir, model_type="tree", output_prefix="xgb_binary", feature_names=feature_names)  # noqa: E501
    _explain_with_lime(xgb_sklearn_model, X_train, X_test, feature_names, class_names, outputPath=xai_output_dir, output_prefix="xgb_binary")  # noqa: E501

    _explain_with_shap(dt, X_test[:100], outputPath=xai_output_dir, model_type="tree", output_prefix="dt_binary", feature_names=feature_names)  # noqa: E501
    _explain_with_lime(dt, X_train, X_test, feature_names, class_names, outputPath=xai_output_dir, output_prefix="dt_binary")  # noqa: E501

    _explain_with_shap(knn, X_test[:50], outputPath=xai_output_dir, model_type="kernel", output_prefix="knn_binary", feature_names=feature_names)  # noqa: E501
    _explain_with_lime(knn, X_train, X_test, feature_names, class_names, outputPath=xai_output_dir, output_prefix="knn_binary")  # noqa: E501

    # try: # SHAP for Neural Net (DeepExplainer requires raw input)
    #     import shap
    #     from shap import DeepExplainer
    #     explainer = DeepExplainer(feedforward_model_bin, X_train[:100])
    #     shap_values = explainer.shap_values(X_test[:10])
    #     shap.summary_plot(shap_values, X_test[:10], show=False, feature_names=X.columns.tolist())
    #     plt.savefig(os.path.join(xai_output_dir, "fnn_binary_summary.png"))
    # except Exception as e:
    #     print("FNN SHAP explanation failed:", e)

    joblib.dump(rf, os.path.join(binary_models_dir, 'rf_model_binary.pkl'))
    joblib.dump(knn, os.path.join(binary_models_dir, 'knn_model_binary.pkl'))
    joblib.dump(dt, os.path.join(binary_models_dir, 'dt_model_binary.pkl'))
    joblib.dump(svm, os.path.join(binary_models_dir, 'svm_model_binary.pkl'))
    joblib.dump(xgb_model, os.path.join(
        binary_models_dir, 'xgb_model_binary.pkl'))
    joblib.dump(feedforward_model_bin, os.path.join(
        binary_models_dir, 'feedforward_model_binary.pkl'))
    joblib.dump(scaler, os.path.join(binary_models_dir, 'scaler_binary.pkl'))
    if isPCA:
        joblib.dump(pca, os.path.join(binary_models_dir,
                    'pca_binary.pkl'))  # type: ignore
    print("Binary models and transformation objects saved in", binary_models_dir)


def run_multiclass_classification(aggregated_file: str, isUNSW: bool, isPCA: bool) -> None:
    """
    Loads aggregated data, performs scaling and PCA, then trains and evaluates
    several multi-class classifiers using the 'Label' column as the target (or
    'Attack' for UNSW). Trained models and transformation objects are saved in the
    'multi_class_models' folder.
    """
    print("Starting Multiclass", file=sys.stderr, flush=True)
    data = pd.read_csv(aggregated_file)

    if 'BinLabel' not in data.columns and 'Label' in data.columns:
        data['BinLabel'] = data['Label'].apply(
            lambda x: 0 if x == 'Benign' else 1)

    X1 = data.drop(
        columns=['BinLabel', 'Label', 'src_ip', 'dst_ip', 'start_time',
                 'end_time_x', 'end_time_y', 'time_diff', 'time_diff_seconds', 'Attack', 'start_time_x', 'start_time_y'],  # noqa: E501
        errors='ignore'
    )

    y1 = data['Label'] if not isUNSW else data['Attack']

    multiclassLabels = sorted(y1.unique().tolist())
    print("Unique classes in target:", multiclassLabels)

    print("Checking for NaN values in features:")
    print(X1.isna().sum())
    print("Any NaN in X:", X1.isna().any().any())
    if X1.isna().any().any():
        print("Found NaN values, filling with column means.")
        X1 = X1.fillna(X1.mean())

    scaler1 = StandardScaler()
    X1_scaled = scaler1.fit_transform(X1)
    if isPCA:
        pca1 = PCA(n_components=0.95)
        X1_pca = pca1.fit_transform(X1_scaled)
        X1_final = X1_pca
    else:
        X1_final = X1_scaled

    # ----- Random Forest (Multi) -----
    print("Starting RF Multiclass", file=sys.stderr, flush=True)
    rf_multi = RandomForestClassifier(random_state=42)
    if isPCA:
        cv_scores = cross_val_score(
            rf_multi, X1_pca, y1, cv=5, scoring='accuracy')  # type: ignore
    else:
        cv_scores = cross_val_score(
            rf_multi, X1_final, y1, cv=5, scoring='accuracy')
    print(
        f"Random Forest (multi) CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    if isPCA:
        X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
            X1_pca, y1, test_size=0.2, random_state=42  # type: ignore
        )
    else:
        X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
            X1_final, y1, test_size=0.2, random_state=42
        )
    rf_multi.fit(X_train_multi, y_train_multi)
    y_pred_rf = rf_multi.predict(X_test_multi)
    print("RF (multi) Confusion Matrix:")
    print(confusion_matrix(y_test_multi, y_pred_rf))
    print("RF (multi) Classification Report:")
    print(classification_report(
        y_test_multi, y_pred_rf, labels=multiclassLabels, target_names=multiclassLabels
    ))

    # ----- K-Nearest Neighbors (Multi) -----
    print("Starting KNN Multiclass", file=sys.stderr, flush=True)
    knn_multi = KNeighborsClassifier(n_neighbors=4)
    if isPCA:
        cv_scores = cross_val_score(
            knn_multi, X1_pca, y1, cv=5, scoring='accuracy')  # type: ignore
    else:
        cv_scores = cross_val_score(
            knn_multi, X1_final, y1, cv=5, scoring='accuracy')
    print(
        f"KNN (multi) CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    knn_multi.fit(X_train_multi, y_train_multi)
    y_pred_knn_multi = knn_multi.predict(X_test_multi)
    print("KNN (multi) Confusion Matrix:")
    print(confusion_matrix(y_test_multi, y_pred_knn_multi))
    print("KNN (multi) Classification Report:")
    print(classification_report(
        y_test_multi, y_pred_knn_multi, labels=multiclassLabels, target_names=multiclassLabels
    ))

    # ----- Decision Tree (Multi) -----
    print("Starting DT Multiclass", file=sys.stderr, flush=True)
    dt_multi = DecisionTreeClassifier(max_depth=54)
    if isPCA:
        cv_scores = cross_val_score(
            dt_multi, X1_pca, y1, cv=5, scoring='accuracy')  # type: ignore
    else:
        cv_scores = cross_val_score(
            dt_multi, X1_final, y1, cv=5, scoring='accuracy')
    print(
        f"Decision Tree (multi) CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    dt_multi.fit(X_train_multi, y_train_multi)
    y_pred_dt = dt_multi.predict(X_test_multi)
    print("Decision Tree (multi) Confusion Matrix:")
    print(confusion_matrix(y_test_multi, y_pred_dt))
    print("Decision Tree (multi) Classification Report:")
    print(classification_report(
        y_test_multi, y_pred_dt, labels=multiclassLabels, target_names=multiclassLabels
    ))

    # ----- SVM (Multi) -----
    print("Starting SVM Multiclass", file=sys.stderr, flush=True)
    svm_multi = SVC(kernel='rbf', C=1, gamma=0.1,
                    random_state=0, probability=True)
    if isPCA:
        cv_scores = cross_val_score(
            svm_multi, X1_pca, y1, cv=5, scoring='accuracy')  # type: ignore
    else:
        cv_scores = cross_val_score(
            svm_multi, X1_final, y1, cv=5, scoring='accuracy')
    print(
        f"SVM (multi) CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    svm_multi.fit(X_train_multi, y_train_multi)
    y_pred_svm_multi = svm_multi.predict(X_test_multi)
    print("SVM (multi) Confusion Matrix:")
    print(confusion_matrix(y_test_multi, y_pred_svm_multi))
    print("SVM (multi) Classification Report:")
    print(classification_report(
        y_test_multi, y_pred_svm_multi,
        labels=multiclassLabels,
        target_names=multiclassLabels
    ))

    # ----- XGBoost (Multi) -----
    print("Starting XGBoost Multiclass", file=sys.stderr, flush=True)
    label_encoder = LabelEncoder()
    y1_encoded = label_encoder.fit_transform(y1)
    if isPCA:
        X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
            X1_pca, y1_encoded, test_size=0.2, random_state=42  # type: ignore
        )
    else:
        X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
            X1_final, y1_encoded, test_size=0.2, random_state=42
        )
    xgb_multi = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(label_encoder.classes_),
        eval_metric='mlogloss',
        random_state=42
    )
    xgb_multi.fit(X_train_multi, y_train_multi)
    if isPCA:
        cv_scores_xgb = cross_val_score(
            xgb_multi, X1_pca, y1_encoded, cv=5, scoring='accuracy')  # type: ignore
    else:
        cv_scores_xgb = cross_val_score(
            xgb_multi, X1_final, y1_encoded, cv=5, scoring='accuracy')
    print(
        f"XGBoost (multi) CV Accuracy: {cv_scores_xgb.mean():.4f} (+/- {cv_scores_xgb.std():.4f})")
    y_pred_xgb = xgb_multi.predict(X_test_multi)
    y_pred_labels = label_encoder.inverse_transform(y_pred_xgb)
    y_test_labels = label_encoder.inverse_transform(y_test_multi)

    present_labels = np.unique(y_test_labels)
    print("XGBoost (multi) Classification Report:")
    print(classification_report(
        y_test_labels, y_pred_labels,
        labels=present_labels,
        target_names=[str(label) for label in present_labels]
    ))
    print("XGBoost (multi) Confusion Matrix:")
    print(confusion_matrix(y_test_labels, y_pred_labels, labels=present_labels))

    # ----- Feedforward Neural Network (Multi) -----
    print("Starting FNN Multiclass", file=sys.stderr, flush=True)
    y1_encoded = label_encoder.fit_transform(y1)
    y1_categorical = to_categorical(y1_encoded)
    X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
        X1_pca, y1_categorical, test_size=0.2, random_state=42  # type: ignore
    )
    feedforward_model = Sequential([
        Input(shape=(X_train_nn.shape[1],)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(y1_categorical.shape[1], activation='softmax')
    ])
    feedforward_model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    feedforward_model.fit(X_train_nn, y_train_nn, epochs=20, batch_size=32, validation_data=(X_test_nn, y_test_nn), verbose=0)  # noqa: E501# noqa: E501
    y_pred_proba = feedforward_model.predict(X_test_nn)
    y_pred_ff = np.argmax(y_pred_proba, axis=1)
    y_test_ff = np.argmax(y_test_nn, axis=1)
    # Dynamically compute target names from the test labels.
    target_names_ff = sorted(np.unique(y_test_ff.astype(str)))
    print("Feedforward NN (multi) Confusion Matrix:")
    print(confusion_matrix(y_test_ff, y_pred_ff))
    print("Feedforward NN (multi) Classification Report:")
    print(classification_report(
        y_test_ff, y_pred_ff,
        labels=sorted(np.unique(y_test_ff)),
        target_names=target_names_ff
    ))

    dataset_name = os.path.basename(os.path.dirname(aggregated_file))
    multi_models_dir = os.path.join(
        os.getcwd(), "multi_class_models", dataset_name)
    if not os.path.exists(multi_models_dir):
        os.makedirs(multi_models_dir)

    '''XAI Additions'''
    xai_output_dir = os.path.join(multi_models_dir, "xai")
    os.makedirs(xai_output_dir, exist_ok=True)

    feature_names_multi = X1.columns.tolist() if not isPCA else [
        f"f{i}" for i in range(X1_final.shape[1])]
    class_names_multi = list(label_encoder.classes_) if isUNSW else sorted(
        data['Label'].unique())

    _explain_with_shap(rf_multi, X_test_multi[:100], outputPath=xai_output_dir, model_type="tree", output_prefix="rf_multi", feature_names=feature_names_multi)  # noqa: E501
    _explain_with_lime(rf_multi, X_train_multi, X_test_multi, feature_names_multi, class_names_multi, outputPath=xai_output_dir, output_prefix="rf_multi")  # noqa: E501

    _explain_with_shap(xgb_multi, X_test_multi[:100], outputPath=xai_output_dir, model_type="tree", output_prefix="xgb_multi", feature_names=feature_names_multi)  # noqa: E501
    _explain_with_lime(xgb_multi, X_train_multi, X_test_multi, feature_names_multi, class_names_multi, outputPath=xai_output_dir, output_prefix="xgb_multi")  # noqa: E501

    _explain_with_shap(dt_multi, X_test_multi[:100], outputPath=xai_output_dir, model_type="tree", output_prefix="dt_multi", feature_names=feature_names_multi)  # noqa: E501
    _explain_with_lime(dt_multi, X_train_multi, X_test_multi, feature_names=feature_names_multi, class_names=class_names_multi, outputPath=xai_output_dir, output_prefix="dt_multi")  # noqa: E501

    _explain_with_shap(knn_multi, X_test_multi[:50], outputPath=xai_output_dir, model_type="kernel", output_prefix="knn_multi", feature_names=feature_names_multi)  # noqa: E501
    _explain_with_lime(knn_multi, X_train_multi, X_test_multi, feature_names=feature_names_multi, class_names=class_names_multi, outputPath=xai_output_dir, output_prefix="knn_multi")  # noqa: E501

    # try:
    #     explainer = shap.DeepExplainer(feedforward_model, X_train_nn[:100])
    #     shap_values = explainer.shap_values(X_test_nn[:10])
    #     shap.summary_plot(shap_values[0], X_test_nn[:10], show=False)
    #     plt.savefig(os.path.join(xai_output_dir, "fnn_multi_summary.png"))
    # except Exception as e:
    #     print("FNN (multi) SHAP explanation failed:", e)

    joblib.dump(rf_multi, os.path.join(
        multi_models_dir, 'random_forest_multi.pkl'))
    joblib.dump(knn_multi, os.path.join(
        multi_models_dir, 'knearest_multi.pkl'))
    joblib.dump(dt_multi, os.path.join(
        multi_models_dir, 'decision_tree_multi.pkl'))
    joblib.dump(svm_multi, os.path.join(multi_models_dir, 'svm_multi.pkl'))
    joblib.dump(xgb_multi, os.path.join(multi_models_dir, 'xgboost_multi.pkl'))
    joblib.dump(feedforward_model, os.path.join(
        multi_models_dir, 'feedforward_multi.pkl'))
    joblib.dump(scaler1, os.path.join(multi_models_dir, 'scaler_multi.pkl'))
    if isPCA:
        joblib.dump(pca1, os.path.join(multi_models_dir,
                    'pca_multi.pkl'))  # type: ignore
    print("Multi-class models and transformation objects saved in", multi_models_dir)


if __name__ == '__main__':
    args = _parse_args()

    run_binary_classification(args.aggregated_file, args.unsw, args.pca)
    run_multiclass_classification(args.aggregated_file, args.unsw, args.pca)
    if args.pca:
        run_feature_engineering(args.aggregated_file)
