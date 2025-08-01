from rest_framework.response import Response
from rest_framework.decorators import api_view
from .utils import predict_next
from .tasks import retrain_model_task
from .models import PredictionLog
from .serializers import PredictionLogSerializer

@api_view(['GET'])
def predict_view(request, interval):
    result = predict_next(interval)
    return Response(result)

@api_view(['POST'])
def train_view(request, interval):
    retrain_model_task.delay(interval)
    return Response({"message": f"Training started for {interval}"})

@api_view(['GET'])
def log_view(request):
    logs = PredictionLog.objects.all().order_by('-created_at')[:50]
    serializer = PredictionLogSerializer(logs, many=True)
    return Response(serializer.data)
