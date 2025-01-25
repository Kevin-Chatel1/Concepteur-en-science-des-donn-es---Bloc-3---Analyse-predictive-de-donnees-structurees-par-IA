from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
from utils.common import *
import requests
import joblib
import importlib.util
import json
import os
from sqlalchemy import create_engine, text, Boolean
from sqlalchemy.exc import SQLAlchemyError
import mlflow

# Configuration des arguments par défaut pour le DAG
default_args = {
    'owner': 'fraud_team',
    'depends_on_past': False,
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Fonction pour charger les dépendances depuis S3
def load_dependencies(**context):
    """Charge le modèle et l'ETL depuis S3"""
    try:
        logger.info("Chargement des dépendances depuis S3...")
        s3_client = get_s3_client()
        
        # Création du dossier temporaire pour stocker les fichiers téléchargés
        tmp_dir = '/tmp/fraud_detection'
        os.makedirs(tmp_dir, exist_ok=True)
        
        # Configuration des chemins S3 sans préfixes additionnels
        s3_paths = {
            'model': "models/random_forest_model.pkl",
            'etl': "etl/etl.py"
        }
        
        # Téléchargement du modèle
        model_path = f"{tmp_dir}/model.pkl"
        s3_client.download_file(S3_BUCKET, s3_paths['model'], model_path)
        logger.info("Modèle chargé avec succès")
        
        # Téléchargement de l'ETL
        etl_path = f"{tmp_dir}/etl.py"
        s3_client.download_file(S3_BUCKET, s3_paths['etl'], etl_path)
        logger.info("ETL chargé avec succès")
        
        # Stockage des chemins dans XCom
        context['task_instance'].xcom_push(key='model_path', value=model_path)
        context['task_instance'].xcom_push(key='etl_path', value=etl_path)
        
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement des dépendances: {str(e)}")
        raise

def fetch_api(**context):
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        transactions = response.json()
        logger.info("Transaction reçue avec succès")
        context['ti'].xcom_push(key='transactions', value=transactions)

    except Exception as e:
        logger.error(f"Erreur de connexion à l'API: {str(e)}")
        raise

def process_transaction(**context):
    try:
        # Récupération des données avec une meilleure gestion d'erreur
        task_instance = context['task_instance']
        raw_transactions = task_instance.xcom_pull(task_ids='fetch_api', key='transactions')
        
        logger.info(f"Données récupérées depuis XCom: {raw_transactions}")

        if not raw_transactions:
            logger.warning("Pas de données à traiter")
            return 'skip_processing'
        
        # Parsing JSON
        transactions = json.loads(raw_transactions)
        logger.info("Données JSON parsées avec succès")
            
        # Création du DataFrame
        df = pd.DataFrame(transactions['data'], columns=transactions['columns'])
        logger.info(f"DataFrame créé avec {len(df)} lignes")
        
        # Renommage de la colonne current_time en trans_date_trans_time
        df = df.rename(columns={'current_time': 'trans_date_trans_time'})
        logger.info("Colonnes renommées avec succès")
        logger.info(f"Colonnes disponibles: {df.columns.tolist()}")

        # Conversion en timestamp Unix (en secondes)
        df['unix_time'] = df['trans_date_trans_time'].astype('int64') // 10**9  # Conversion en secondes
        logger.info("Conversion de trans_date_trans_time en unix_time réussie")
        
        # Import de l'ETL
        etl_path = context['ti'].xcom_pull(task_ids='load_dependencies', key='etl_path')
        spec = importlib.util.spec_from_file_location("etl", etl_path)
        etl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(etl)
        
        # Transformation des données
        logger.info("Transformation des données...")
        df_transformed = etl.transform_data(df)
        
        # Chargement et prédiction avec le modèle
        model_path = context['ti'].xcom_pull(task_ids='load_dependencies', key='model_path')
        model = joblib.load(model_path)
        
        # Prédiction
        prediction = model.predict(df_transformed)[0]
        probability = model.predict_proba(df_transformed)[:, 1][0]

        # Ajout du logging pour la prédiction et la probabilité
        logger.info(f"Prediction de fraude: {'FRAUDE' if prediction == 1 else 'NORMAL'}, Probabilité: {probability:.2%}")
        
        # Stockage des résultats
        context['ti'].xcom_push(key='is_fraud', value=bool(prediction))
        context['ti'].xcom_push(key='fraud_probability', value=float(probability))

        logger.info(f"Prédiction: {'FRAUDE' if prediction == 1 else 'NORMAL'} (Probabilité: {probability:.2%})")
        
        # version à remettre quand email sera verif : return 'send_fraud_alert' if prediction == 1 else 'store_normal'
        return 'notify_fraud' if prediction == 1 else 'notify_normal'

    except Exception as e:
        logger.error(f"Erreur lors du traitement: {str(e)}")
        logger.error(f"Type des transactions: {type(raw_transactions)}")
        logger.error(f"Contenu des transactions: {str(raw_transactions)[:200]}...")
        return 'skip_processing' 

def send_transaction_email(**context):
    try:
        # Récupération des données
        transaction = context['task_instance'].xcom_pull(task_ids='fetch_api', key='transactions')
        probability = context['task_instance'].xcom_pull(key='fraud_probability')
        is_fraud = context['task_instance'].xcom_pull(key='is_fraud', default=False)
        
        logger.info(f"Données de transaction récupérées: {transaction}")
        logger.info(f"Probabilité de fraude: {probability}")
        logger.info(f"Est une fraude: {is_fraud}")
        
        # Vérification des données
        if transaction is None:
            logger.error("Aucune donnée de transaction trouvée dans XCom.")
            return 'store_normal' if not is_fraud else 'store_fraud'

        if probability is None:
            logger.error("Aucune probabilité de fraude trouvée dans XCom.")
            return 'store_normal' if not is_fraud else 'store_fraud'
            
        # Parser le JSON si nécessaire
        if isinstance(transaction, str):
            transaction = json.loads(transaction)
            
        # Convertir en DataFrame pour faciliter l'accès aux données
        df = pd.DataFrame(transaction['data'], columns=transaction['columns']).iloc[0]
        
        # Formatage du montant avec 2 décimales
        amount = "{:.2f}".format(float(df['amt']))
        
        # Formatage de la date
        transaction_date = pd.to_datetime(df['current_time'], unit='ms').strftime('%Y-%m-%d %H:%M:%S')

        # Construction du sujet et du contenu en fonction du type de transaction
        if is_fraud:
            subject = "🚨 ALERTE: Transaction frauduleuse détectée!"
            status_color = "Rouge"
            status_text = "FRAUDULEUSE"
        else:
            subject = "✅ INFO: Transaction normale détectée"
            status_color = "Vert"
            status_text = "NORMALE"
        
        body = f"""
        {'🚨' if is_fraud else '✅'} Transaction {status_text} détectée!
        
        Statut: {status_color}
        Probabilité de fraude: {probability:.2%}
        
        Détails de la transaction:
        --------------------------
        ID Transaction: {df['trans_num']}
        Montant: {amount}€
        Date/Heure: {transaction_date}
        
        Informations sur le marchand:
        ----------------------------
        Nom: {df['merchant']}
        Ville: {df['city']}
        État: {df['state']}
        
        Informations sur le client:
        --------------------------
        Nom: {df['first']} {df['last']}
        Ville: {df['city']}
        
        Cette notification a été générée automatiquement par le système de détection de fraude.
        """
        
        logger.info(f"Tentative d'envoi d'email {status_text.lower()}...")
        logger.info(f"Contenu de l'email:\n{body}")
        
        # Utilisation de la fonction send_email de common.py
        success = send_email(
            subject=subject, 
            body=body
            # On n'envoie pas to_email pour utiliser l'email par défaut
        )
        
        if success:
            logger.info("Email envoyé avec succès")
        else:
            logger.error("Échec de l'envoi de l'email")
            
        return 'store_normal' if not is_fraud else 'store_fraud'
        
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi de l'email: {str(e)}")
        return 'store_normal' if not is_fraud else 'store_fraud'

def store_transaction(**context):
    try:
        # Test de connexion à Neon
        db_url = os.environ.get('NEON_DATABASE_URL')
        if not db_url:
            logger.error("NEON_DATABASE_URL n'est pas définie")
            raise ValueError("NEON_DATABASE_URL manquante")
            
        logger.info("Tentative de connexion à Neon...")
        engine = create_engine(db_url)
        
        # Test simple de connexion
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("Connexion à Neon réussie!")
        
        # Récupération et parsing des données
        raw_transactions = context['task_instance'].xcom_pull(
            task_ids='fetch_api', 
            key='transactions'
        )
        logger.info(f"Données reçues pour stockage: {raw_transactions}")
        
        if not raw_transactions:
            logger.error("Aucune donnée de transaction trouvée dans XCom.")
            return
            
        # Parser le JSON si c'est une chaîne
        if isinstance(raw_transactions, str):
            raw_transactions = json.loads(raw_transactions)
            logger.info("JSON parsé avec succès")
        
        # Création du DataFrame
        df = pd.DataFrame(raw_transactions['data'], columns=raw_transactions['columns'])
        logger.info(f"DataFrame créé avec {len(df)} lignes")

        # Renommer la colonne current_time en trans_date_trans_time
        if 'current_time' in df.columns:
            df = df.rename(columns={'current_time': 'trans_date_trans_time'})
            logger.info("Colonne current_time renommée en trans_date_trans_time")
        
        # Convertir le timestamp en datetime
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], unit='ms')
        logger.info("Conversion du timestamp réussie")

        # Récupération de l'indicateur de fraude depuis XCom
        is_fraud = context['task_instance'].xcom_pull(key='is_fraud', default=False)
        logger.info(f"Indicateur de fraude récupéré: {is_fraud}")

        # Mise à jour de la colonne is_fraud dans le DataFrame
        df['is_fraud'] = is_fraud
        logger.info(f"Colonne is_fraud mise à jour avec la valeur: {is_fraud}")

        # Convertir is_fraud en boolean
        df['is_fraud'] = df['is_fraud'].astype(bool)
        logger.info("Conversion de is_fraud en boolean réussie")

        # Log des colonnes et types
        logger.info(f"Colonnes finales avant stockage: {df.columns.tolist()}")
        logger.info(f"Types des données: \n{df.dtypes}")

        # Détermination de la table cible
        table = 'fraud_transactions' if is_fraud else 'normal_transactions'
        logger.info(f"Stockage dans la table: {table}")
        
        # Stockage des données avec dtype spécifié pour is_fraud
        dtype = {'is_fraud': Boolean}
        df.to_sql(table, engine, if_exists='append', index=False, dtype=dtype)
        logger.info(f"Données stockées avec succès dans la table {table}")

    except SQLAlchemyError as e:
        logger.error(f"Erreur SQLAlchemy: {str(e)}")
        if 'df' in locals():
            logger.error(f"Colonnes du DataFrame: {df.columns.tolist()}")
            logger.error(f"Types des données: \n{df.dtypes}")
        raise
    except Exception as e:
        logger.error(f"Erreur inattendue: {str(e)}")
        logger.error(f"Type des données reçues: {type(raw_transactions)}")
        raise

def store_normal(**context):
    """Wrapper pour stocker les transactions normales"""
    context['task_instance'].xcom_push(key='is_fraud', value=False)
    return store_transaction(**context)

def store_fraud(**context):
    """Wrapper pour stocker les transactions frauduleuses"""
    context['task_instance'].xcom_push(key='is_fraud', value=True)
    return store_transaction(**context)

def configure_mlflow():
    """Configure MLflow et force l'utilisation d'une run unique."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    experiment_name = "fraud_detection_pipeline"

    # Vérifie si l'expérience existe déjà
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        # Crée l'expérience si elle n'existe pas
        mlflow.create_experiment(name=experiment_name)
    mlflow.set_experiment(experiment_name)

def log_to_mlflow(**context):
    """Log les résultats dans MLflow, en écrasant les artefacts précédents."""
    try:
        configure_mlflow()
        client = mlflow.tracking.MlflowClient()

        # Nom de l'expérience et du run
        experiment_name = "fraud_detection_pipeline"
        artifact_path = "model"
        fixed_run_name = "fraud_detection_fixed_run"

        # Vérifier ou créer l'expérience
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.info(f"Création de l'expérience {experiment_name}.")
            experiment_id = client.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        # Supprimer les artefacts existants dans S3
        s3_client = get_s3_client()
        bucket_name = os.environ.get("S3_BUCKET")
        prefix = f"mlartifacts/{artifact_path}/"
        logger.info(f"Suppression des artefacts existants dans le bucket {bucket_name} au préfixe {prefix}...")
        
        objects_to_delete = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix).get('Contents', [])
        if objects_to_delete:
            delete_request = {
                'Objects': [{'Key': obj['Key']} for obj in objects_to_delete],
                'Quiet': True
            }
            s3_client.delete_objects(Bucket=bucket_name, Delete=delete_request)
            logger.info("Artefacts existants supprimés avec succès.")
        else:
            logger.info("Aucun artefact à supprimer.")

        # Créer un nouveau run
        logger.info(f"Création d'un nouveau run avec le nom {fixed_run_name}.")
        with mlflow.start_run(experiment_id=experiment_id, run_name=fixed_run_name):
            # Récupérer les données de contexte
            is_fraud = context['task_instance'].xcom_pull(key='is_fraud')
            fraud_probability = context['task_instance'].xcom_pull(key='fraud_probability')

            # Enregistrer les métriques et paramètres
            mlflow.log_param("model_version", "RandomForest")
            mlflow.log_metric("fraud_probability", fraud_probability)
            mlflow.log_metric("is_fraud", int(is_fraud))

            # Enregistrer les artefacts
            mlflow.log_artifact("/tmp/fraud_detection/model.pkl", artifact_path="model")
            mlflow.log_artifact("/tmp/fraud_detection/etl.py", artifact_path="etl")

        logger.info("Run MLflow créé et logué avec succès.")

    except Exception as e:
        logger.error(f"Erreur lors du logging dans MLflow : {str(e)}")
        raise

with DAG(
    'fraud_detection',
    default_args=default_args,
    description='Détection de fraude en temps réel',
    schedule_interval='* * * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:
    
    # Chargement des dépendances
    load_deps = PythonOperator(
        task_id='load_dependencies',
        python_callable=load_dependencies,
    )

    # Récupération des données de l'API
    fetch_api = PythonOperator(
        task_id='fetch_api',
        python_callable=fetch_api,
        do_xcom_push=True,
    )

    # Traitement et prédiction
    process = BranchPythonOperator(
        task_id='process_transaction',
        python_callable=process_transaction,
    )

    # Email et stockage pour transactions normales
    notify_normal = PythonOperator(
        task_id='notify_normal',
        python_callable=send_transaction_email,
        provide_context=True
    )

    store_normal = PythonOperator(
        task_id='store_normal',
        python_callable=store_transaction,
        provide_context=True
    )

    # Email et stockage pour transactions frauduleuses
    notify_fraud = PythonOperator(
        task_id='notify_fraud',
        python_callable=send_transaction_email,
        provide_context=True
    )

    store_fraud = PythonOperator(
        task_id='store_fraud',
        python_callable=store_transaction,
        provide_context=True
    )

    skip = PythonOperator(
        task_id='skip_processing',
        python_callable=lambda: logger.info("Pas de données")
    )

    log_metrics = PythonOperator(
    task_id='log_to_mlflow',
    python_callable=log_to_mlflow,
    provide_context=True,
    trigger_rule='none_failed_min_one_success'  # Exécution si une tâche réussit
)


    # Définition des dépendances
    load_deps >> fetch_api >> process >> [notify_normal, notify_fraud, skip]
    notify_normal >> store_normal >> log_metrics
    notify_fraud >> store_fraud >> log_metrics

    # Définition des dépendances
    load_deps >> fetch_api >> process >> [notify_normal, notify_fraud, skip]
    notify_normal >> store_normal
    notify_fraud >> store_fraud

    # L'étape log_to_mlflow
    [store_normal, store_fraud] >> log_metrics

