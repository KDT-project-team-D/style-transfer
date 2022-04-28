import os.path

from django.http import JsonResponse, HttpResponse
from django.shortcuts import render, redirect
from django.views.generic import TemplateView, ListView, CreateView
from django.core.files.storage import FileSystemStorage
from django.urls import reverse_lazy
from django.template import loader


from .forms import PhotoForm
from .models import Photo

from PIL import Image
import numpy as np
import io, base64
import mimetypes
import json
# Create your views here.
# 메인

class Main(TemplateView):
    template_name = 'main.html'


def index(request):
    template = loader.get_template('index.html')
    context = {'form': PhotoForm()}

    return HttpResponse(template.render(context, request))

def cezanne(request):
    template = loader.get_template('cezanne.html')
    context = {'form': PhotoForm()}

    return HttpResponse(template.render(context, request))

def monet(request):
    template = loader.get_template('monet.html')
    context = {'form': PhotoForm()}

    return HttpResponse(template.render(context, request))


def predict(request):
    if not request.method == 'POST':
        return redirect('content:index')

    form = PhotoForm(request.POST, request.FILES)
    if not form.is_valid():
        raise ValueError('Form error')

    file_name = form.cleaned_data['image'].name
    print("file_name=", file_name)
    photo = Photo(image=form.cleaned_data['image'])
    translated_img = photo.vangogh_predict()  # (1, 128, 128, 3)
    # photo.save()
    translated_img = np.squeeze(translated_img, 0)  # (1, 128, 128, 3) -> (128, 128, 3)
    translated_img = translated_img.astype(np.float32)
    translated_img = translated_img + 1  # 0～1로 변환
    pil_img = Image.fromarray((translated_img * 128).astype('uint8'), mode='RGB')
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue())
    img_str = str(img_str)[2:-1]
    content_type = mimetypes.guess_type(file_name)[0]
    translated_img = 'data:' + content_type + ';base64,' + img_str

    d = {
        'img_str': translated_img,
    }
    return JsonResponse(d)


def cezanne_predict(request):
    if not request.method == 'POST':
        return redirect('content:cezanne')

    form = PhotoForm(request.POST, request.FILES)
    if not form.is_valid():
        raise ValueError('Form error')

    file_name = form.cleaned_data['image'].name
    print("file_name=", file_name)
    photo = Photo(image=form.cleaned_data['image'])
    translated_img = photo.cezanne_predict()  # (1, 128, 128, 3)
    # photo.save()
    translated_img = np.squeeze(translated_img, 0)  # (1, 128, 128, 3) -> (128, 128, 3)
    translated_img = translated_img.astype(np.float32)
    translated_img = translated_img + 1  # 0～1로 변환
    pil_img = Image.fromarray((translated_img * 128).astype('uint8'), mode='RGB')
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue())
    img_str = str(img_str)[2:-1]
    content_type = mimetypes.guess_type(file_name)[0]
    translated_img = 'data:' + content_type + ';base64,' + img_str

    d = {
        'img_str': translated_img,
    }
    return JsonResponse(d)


def monet_predict(request):
    if not request.method == 'POST':
        return redirect('content:monet')

    form = PhotoForm(request.POST, request.FILES)
    if not form.is_valid():
        raise ValueError('Form error')

    file_name = form.cleaned_data['image'].name
    print("file_name=", file_name)
    photo = Photo(image=form.cleaned_data['image'])
    translated_img = photo.monet_predict()  # (1, 256, 256, 3)
    # photo.save()
    translated_img = np.squeeze(translated_img, 0)  # (1, 256, 256, 3) -> (256, 256, 3)
    translated_img = translated_img.astype(np.float32)
    translated_img = translated_img + 1  # 0～1로 변환
    pil_img = Image.fromarray((translated_img * 128).astype('uint8'), mode='RGB')
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue())
    img_str = str(img_str)[2:-1]
    content_type = mimetypes.guess_type(file_name)[0]
    translated_img = 'data:' + content_type + ';base64,' + img_str

    d = {
        'img_str': translated_img,
    }
    return JsonResponse(d)