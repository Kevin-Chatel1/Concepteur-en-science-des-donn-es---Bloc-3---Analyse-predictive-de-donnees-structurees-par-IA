from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
from utils.common import *

# Ajout de la configuration du préfixe
S3_PREFIX = ''

def check_system_health(**context):
    try:
        status = {
            'api': False,
            'database': False,
            'storage': False
        }
        
        # Vérification API
        try:
            response = requests.get(API_URL, timeout=10)
            status['api'] = response.status_code == 200
        except Exception as e:
            logger.error(f"Erreur API: {str(e)}")
            
        # Vérification Base de données
        try:
            engine = get_db_engine()
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            status['database'] = True
        except Exception as e:
            logger.error(f"Erreur DB: {str(e)}")
            
        # Vérification S3
        try:
            s3 = get_s3_client()
            # Vérifie d'abord si le bucket existe
            s3.head_bucket(Bucket=S3_BUCKET)
            
            # Vérifie ensuite si le préfixe existe
            response = s3.list_objects_v2(
                Bucket=S3_BUCKET,
                Prefix=S3_PREFIX,
                MaxKeys=1
            )
            
            status['storage'] = True
            logger.info(f"Accès S3 vérifié: bucket={S3_BUCKET}, prefix={S3_PREFIX}")
            
        except Exception as e:
            logger.error(f"Erreur S3: {str(e)}")
            
        # Rapport
        body = "Rapport de santé du système:\n\n"
        body += f"API URL: {API_URL}\n"
        body += f"S3 Path: s3://{S3_BUCKET}/{S3_PREFIX}\n\n"
        body += "Statut des services:\n"
        for service, is_healthy in status.items():
            body += f"{service}: {'✅' if is_healthy else '❌'}\n"
            
        if not all(status.values()):
            body += "\n⚠️ Des actions sont requises!"
            
        send_email("Rapport quotidien - Système de détection de fraude", body)
        
    except Exception as e:
        logger.error(f"Erreur monitoring: {str(e)}")
        raise

def test_s3_connection(**context):
    """Fonction de test pour vérifier l'accès à S3"""
    try:
        s3 = get_s3_client()
        
        # Liste tous les buckets
        buckets = s3.list_buckets()['Buckets']
        logger.info("Buckets disponibles:")
        for bucket in buckets:
            logger.info(f"- {bucket['Name']}")
            
        # Vérifie le bucket cible
        logger.info(f"\nTest du bucket cible: {S3_BUCKET}")
        s3.head_bucket(Bucket=S3_BUCKET)
        
        # Liste le contenu du préfixe
        logger.info(f"Test du préfixe: {S3_PREFIX}")
        response = s3.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=S3_PREFIX,
            MaxKeys=10
        )
        
        if 'Contents' in response:
            logger.info("Objets trouvés dans le préfixe:")
            for obj in response['Contents']:
                logger.info(f"- {obj['Key']}")
        else:
            logger.info("Préfixe vide ou inexistant")
            
    except Exception as e:
        logger.error(f"Erreur test S3: {str(e)}")
        raise

# Configuration du DAG
default_args = {
    'owner': 'fraud_team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    'fraud_monitoring',
    default_args=default_args,
    description='Monitoring quotidien',
    schedule_interval='0 0 * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:

    test_s3 = PythonOperator(
        task_id='test_s3_connection',
        python_callable=test_s3_connection,
        provide_context=True
    )

    health_check = PythonOperator(
        task_id='system_health_check',
        python_callable=check_system_health,
        provide_context=True
    )

    test_s3 >> health_check