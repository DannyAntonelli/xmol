from django.urls import path
from django.views.generic import TemplateView

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path(
        '/xai_methods',
        TemplateView.as_view(template_name='xai_methods.html'),
        name='xai_methods',
    ),
]
