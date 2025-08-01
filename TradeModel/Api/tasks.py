from celery import shared_task
from .utils import train_model_with_latest_data

@shared_task
def retrain_model_task(interval):
    train_model_with_latest_data(interval)
    return f"Model retrained for interval: {interval}"