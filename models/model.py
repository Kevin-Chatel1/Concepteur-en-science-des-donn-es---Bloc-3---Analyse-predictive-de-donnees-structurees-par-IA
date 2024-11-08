import pandas as pd
import numpy as np
import joblib
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_validate
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

# Configuration MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("fraud_detection_version_def_avecetl")

# Chargement des données prétraitées
data_reduc = pd.read_csv('data_post_etl.csv')

# Configuration des colonnes
column_configs = [
    ['category', 'amt', 'lat', 'long', 'job', 'unix_time', 'merch_lat', 'merch_long', 'month', 'day_of_week', 'hour', 'age', 'distance']
]

for config in column_configs:
    with mlflow.start_run():
        mlflow.set_tag("model", "RandomForestClassifier-withSMOTE-reduc50klines-V2")
        
        # Définir X et y
        X = data_reduc[config].copy()
        y = data_reduc['is_fraud'].copy()

        # Split des données
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

        # Définition des transformateurs
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))])

        # Création du préprocesseur
        preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])

        # Définition du modèle
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1)

        # Création du pipeline avec SMOTE (ImbPipeline)
        pipeline = ImbPipeline([('preprocessor', preprocessor), ('smote', SMOTE(random_state=42, sampling_strategy=0.1)), ('classifier', clf)])

        # Cross-validation avec gestion d'erreurs
        cv_results = cross_validate(pipeline, X_train, y_train, cv=5, scoring={'precision': 'precision_weighted', 'recall': 'recall_weighted', 'f1': 'f1_weighted', 'roc_auc': 'roc_auc'}, return_train_score=True, error_score='raise')

        # Entraînement final
        pipeline.fit(X_train, y_train)

        # Prédictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        # Calcul des métriques
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Log des paramètres avec MLflow
        mlflow.log_param("config", config)
        mlflow.log_param("n_estimators", clf.n_estimators)
        mlflow.log_param("max_depth", clf.max_depth)
        mlflow.log_param("smote_ratio", 0.1)

        # Log des métriques de cross-validation
        for metric in ['precision', 'recall', 'f1', 'roc_auc']:
            mlflow.log_metric(f"cv_train_{metric}_mean", cv_results[f'train_{metric}'].mean())
            mlflow.log_metric(f"cv_train_{metric}_std", cv_results[f'train_{metric}'].std())
            mlflow.log_metric(f"cv_test_{metric}_mean", cv_results[f'test_{metric}'].mean())
            mlflow.log_metric(f"cv_test_{metric}_std", cv_results[f'test_{metric}'].std())

        # Log des métriques finales
        mlflow.log_metric("final_test_precision", precision)
        mlflow.log_metric("final_test_recall", recall)
        mlflow.log_metric("final_test_f1", f1)
        mlflow.log_metric("final_test_auc_roc", auc_roc)

        # Visualisation de la matrice de confusion
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Matrice de Confusion')
        plt.ylabel('Vraie classe')
        plt.xlabel('Classe prédite')
        plt.tight_layout()

        # Sauvegarde et log de la figure
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')
        plt.close()

        # Affichage des résultats
        print("\nRésultats de la Cross-Validation:")
        for metric in ['precision', 'recall', 'f1', 'roc_auc']:
            print(f"\n{metric.upper()}:")
            print(f"Train: {cv_results[f'train_{metric}'].mean():.3f} ± {cv_results[f'train_{metric}'].std():.3f}")
            print(f"Test:  {cv_results[f'test_{metric}'].mean():.3f} ± {cv_results[f'test_{metric}'].std():.3f}")
        print("\nRésultats sur le test set final:")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-score: {f1:.3f}")
        print(f"AUC-ROC: {auc_roc:.3f}")

        # Sauvegarder le modèle
        model_filename = 'random_forest_model.pkl'
        joblib.dump(pipeline, model_filename)
        mlflow.log_artifact(model_filename)
        print("Modèle sauvegardé !")
