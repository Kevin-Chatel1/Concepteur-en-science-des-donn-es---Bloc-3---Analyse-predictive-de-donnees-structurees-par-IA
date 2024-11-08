import pandas as pd
import numpy as np
from datetime import datetime
from geopy.distance import geodesic

def transform_data(dataset):
    """Transforme les données pour la détection de fraude"""
    data_reduc = dataset.copy()
    
    # Features temporelles
    data_reduc['trans_date_trans_time'] = pd.to_datetime(data_reduc['trans_date_trans_time'])
    data_reduc['month'] = data_reduc['trans_date_trans_time'].dt.month
    data_reduc['day_of_week'] = data_reduc['trans_date_trans_time'].dt.dayofweek + 1
    data_reduc['hour'] = data_reduc['trans_date_trans_time'].dt.hour
    
    # Calcul de l'âge
    data_reduc['dob'] = pd.to_datetime(data_reduc['dob'])
    def calcule_age(born):
        today = datetime.today()
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    data_reduc['age'] = data_reduc['dob'].apply(calcule_age)
    
    # Calcul de la distance
    data_reduc['distance'] = data_reduc.apply(lambda row: geodesic((row['lat'], row['long']), (row['merch_lat'], row['merch_long'])).km, axis=1)
    
    return data_reduc

def load_data(data_reduc):
    """Retourne les données transformées"""
    return data_reduc

if __name__ == "__main__":
    print("Module de transformation des données chargé.")