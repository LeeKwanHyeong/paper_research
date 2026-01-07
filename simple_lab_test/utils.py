import polars as pl
import torch
import os, sys
sys.path.insert(0, os.path.abspath(".."))

MAC_DIR = '/Users/igwanhyeong/PycharmProjects/paper_research/'
WINDOW_DIR = 'C:/Users/USER/PycharmProjects/research/raw_data/'

if sys.platform == 'win32':
    DIR = WINDOW_DIR
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.version.cuda)
    print(torch.__version__)
    print(torch.cuda.get_device_name(0))
    print(torch.__version__)
else:
    DIR = MAC_DIR

def get_taxi_data(vendor: int = 0):
    select_schema = [
        'tpep_pickup_datetime', 'tpep_dropoff_datetime',
        'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude',
    ]
    rename_schema = {
        'tpep_pickup_datetime': 'pick_dt', 'tpep_dropoff_datetime': 'drop_dt',
        'pickup_longitude': 'pick_lon', 'pickup_latitude': 'pick_lat',
        'dropoff_longitude': 'drop_lon', 'dropoff_latitude': 'drop_lat'
    }

    vendor1_data = (
        pl.read_parquet(DIR + 'sample_data/yellow_trip.parquet')
        .filter(pl.col('VendorID') == 1)  # VendorID = 1 | 2
        .select(select_schema)
        .rename(rename_schema)
    )

    vendor2_data = (
        pl.read_parquet(DIR + 'sample_data/yellow_trip.parquet')
        .filter(pl.col('VendorID') == 2)
        .select(select_schema)
        .rename(rename_schema)
    )

    if vendor == 1:
        return vendor1_data
    elif vendor == 2:
        return vendor2_data
    else:
        return vendor1_data + vendor2_data