from django.urls import path
from . import views

app_name = 'chat_app'

urlpatterns = [
    path('', views.chat_home, name='home'),
    path('api/ask/', views.ask_question, name='ask_question'),
] 