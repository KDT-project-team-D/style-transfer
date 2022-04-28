from . import views
from django.contrib import admin
from django.urls import path

app_name = 'content'

urlpatterns = [
    path('predict/', views.predict, name='predict'),
    path('cezanne_predict/', views.cezanne_predict, name='cezanne_predict'),
    path('monet_predict/', views.monet_predict, name='monet_predict'),

    path('vangogh/', views.index, name='vangogh'),
    path('cezanne/', views.cezanne, name='cezanne'),
    path('monet/', views.monet, name='monet'),
]


