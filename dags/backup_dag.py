from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess
import os
import pandas as pd
from utils.common import *

def perform_neon_backup(**context):
    """Sauvegarde des tables depuis Neon DB"""
    try:
        logger.info("Démarrage du backup Neon...")
        
        # Connexion à la base pour vérifier les données
        engine = get_db_engine()
        
        # Création du dossier de travail
        work_dir = '/opt/airflow/backups'
        os.makedirs(work_dir, exist_ok=True)
        
        # Récupération de l'URL de la base
        db_url = os.environ.get('NEON_DATABASE_URL')
        if not db_url:
            raise ValueError("NEON_DATABASE_URL n'est pas définie")
        
        # Vérification des données avant backup
        tables_data = {}
        with engine.connect() as conn:
            for table in ['fraud_transactions', 'normal_transactions']:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").scalar()
                logger.info(f"Nombre de lignes dans {table}: {count}")
                tables_data[table] = count
        
        current_date = datetime.now().strftime('%Y%m')
        backup_files = []
        
        # Sauvegarde au format CSV
        for table in tables_data.keys():
            if tables_data[table] > 0:
                logger.info(f"Sauvegarde de {table} ({tables_data[table]} lignes)...")
                
                # Lecture des données
                df = pd.read_sql(f"SELECT * FROM {table}", engine)
                
                # Sauvegarde en CSV
                csv_file = os.path.join(work_dir, f"{table}_{current_date}.csv")
                df.to_csv(csv_file, index=False)
                
                # Vérification du fichier
                if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
                    backup_files.append(csv_file)
                    logger.info(f"Fichier créé: {csv_file} ({os.path.getsize(csv_file)/1024/1024:.2f} MB)")
                else:
                    logger.warning(f"Le fichier {csv_file} est vide ou n'existe pas")
            else:
                logger.info(f"Table {table} est vide, pas de backup nécessaire")
        
        if not backup_files:
            logger.warning("Aucun fichier de backup créé - toutes les tables sont vides")
            return
        
        # Upload vers S3
        s3_client = get_s3_client()
        uploaded_files = []
        
        for backup_file in backup_files:
            try:
                file_name = os.path.basename(backup_file)
                s3_key = f"backups/{current_date}/{file_name}"  # Chemin S3 sans S3_PREFIX
                
                logger.info(f"Upload vers S3: {s3_key}")
                s3_client.upload_file(backup_file, S3_BUCKET, s3_key)
                uploaded_files.append(s3_key)
                
                # Vérification de l'upload
                s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
                logger.info(f"Upload vérifié pour {s3_key}")
                
            except Exception as e:
                logger.error(f"Erreur lors de l'upload de {backup_file}: {str(e)}")
                raise
            finally:
                if os.path.exists(backup_file):
                    os.remove(backup_file)
                    logger.info(f"Fichier local supprimé: {backup_file}")
        
        # Notification email
        files_info = "\n".join([
            f"- {table}: {count:,} lignes"
            for table, count in tables_data.items()
            if count > 0
        ])
        
        email_body = f"""
        Backup Neon DB effectué avec succès!
        
        Résumé des données sauvegardées:
        {files_info}
        
        Fichiers créés:
        {chr(10).join([f"- s3://{S3_BUCKET}/{f}" for f in uploaded_files])}
        
        Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        send_email(
            subject=f"✅ Backup Neon DB - {current_date}",
            body=email_body
        )
        
        logger.info("Backup Neon terminé avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors du backup Neon: {str(e)}")
        raise


default_args = {
    'owner': 'fraud_team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    'neon_backup_v2',  # Nouveau nom pour éviter les conflits
    default_args=default_args,
    description='Backup des tables Neon DB en CSV',
    schedule_interval='0 0 1 * *',
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:

    backup = PythonOperator(
        task_id='perform_neon_backup',
        python_callable=perform_neon_backup,
        provide_context=True
    )