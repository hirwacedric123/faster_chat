from django.urls import path
from . import views

app_name = 'documents'

urlpatterns = [
    path('', views.document_home, name='home'),
    path('upload/', views.upload_document, name='upload'),
    path('list/', views.document_list, name='list'),
    path('delete/<int:doc_id>/', views.delete_document, name='delete'),
] 