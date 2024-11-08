# dags/utils/common.py
import os
import logging
import boto3
from sqlalchemy import create_engine
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

# Configuration du logging
logger = logging.getLogger(__name__)

# Configuration des variables d'environnement
API_URL = os.environ.get('FRAUD_API_URL', 'https://real-time-payments-api.herokuapp.com/current-transactions')
NEON_URL = os.environ.get('NEON_DATABASE_URL')
S3_BUCKET = os.environ.get('S3_BUCKET')
S3_PREFIX = ''

def get_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
    )

def get_db_engine():
    return create_engine(NEON_URL)

def send_email(subject, body, to_email=None):
    try:
        msg = MIMEMultipart()
        msg['From'] = os.environ.get('EMAIL_USER')
        msg['To'] = to_email or os.environ.get('EMAIL_TO', 'chatelain.kev@gmail.com')
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(
            os.environ.get('SMTP_SERVER', 'smtp.gmail.com'),
            int(os.environ.get('SMTP_PORT', 587))
        ) as server:
            server.starttls()
            server.login(
                os.environ.get('EMAIL_USER'),
                os.environ.get('EMAIL_PASSWORD')
            )
            server.send_message(msg)
            
        return True
    except Exception as e:
        logger.error(f"Erreur d'envoi d'email: {str(e)}")
        return False