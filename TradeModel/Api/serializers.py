from rest_framework import serializers
from .models import PredictionLog

class PredictionLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionLog
        fields = '__all__'
