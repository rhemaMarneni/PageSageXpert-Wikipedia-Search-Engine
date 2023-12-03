"""mds_app URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# from django.contrib import admin
# from django.urls import path

# urlpatterns = [
#     path('admin/', admin.site.urls),
# ]
# from django.urls import path
# from .views import index

# urlpatterns = [
#     path('index/', index, name='index'),
#     #path('redirect/', your_redirect_view, name='your_redirect_view'),
#     # Add other URL patterns as needed
# ]
# from django.urls import path
# from .views import index,question,login_or_register,logout_view
# from django.contrib import admin


# urlpatterns = [
#     path('', index, name='index'),  # Empty path for the root URL
#     path('question/',question, name='question'),
#     path('login_or_register/', login_or_register, name='login_or_register'),
#     path('logout/', logout_view, name='logout')
#     # path('register/', register, name='register'),
#     # path('login/', user_login, name='login')

   

# ]
from django.urls import path
from .views import index, question, login_or_register, logout_view,summary_view, graph_view
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', login_or_register, name='login_or_register'),  
    path('question/', question, name='question'),
    path('logout/', logout_view, name='logout'),
    path('index/', index, name='index'), 
    path('summary/<int:page_id>/', views.summary_view, name='summary'),
    path('graph/', graph_view, name='graph'),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)