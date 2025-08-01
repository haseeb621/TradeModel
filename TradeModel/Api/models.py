from django.db import models

class CandleData(models.Model):
    interval = models.CharField(max_length=10)
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    volume = models.FloatField()
    timestamp = models.BigIntegerField(unique=True)

class PredictionLog(models.Model):
    interval = models.CharField(max_length=10)
    predicted_close = models.FloatField()
    real_close = models.FloatField()
    timestamp = models.BigIntegerField()
    error = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
