from django.urls import path
from . import views

urlpatterns = [
    path('predict/<str:interval>/', views.predict_view),
    path('train/<str:interval>/', views.train_view),
    path('logs/', views.log_view),
]