from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='main-home'),
    path('results/', views.results, name='main-results')
]