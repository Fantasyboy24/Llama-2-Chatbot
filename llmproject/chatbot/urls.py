from django.urls import path
from . import views

urlpatterns = [

    path('', views.index, name='home'),
    path('login/', views.user_login, name='login'),
    path('signup/', views.user_signup, name='signup'),
    path('logout/', views.user_logout, name='logout'),
    path('upload/', views.upload_file_to_fastapi, name='upload'),
    path('search/', views.search_content, name='search'),
    path('show_llama2_form/', views.show_llama2_form, name='show_llama2_form'),
    path('call_fastapi_llama2/', views.call_fastapi_llama2, name='call_fastapi_llama2'),
]