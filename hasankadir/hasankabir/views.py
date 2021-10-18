from django.shortcuts import render
from django.http import HttpResponse

def index(request):
    return HttpResponse("Расчет распределения давления по затрубному пространству")
# Create your views here.
