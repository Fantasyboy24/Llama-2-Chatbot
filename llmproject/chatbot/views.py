import requests
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.http import JsonResponse,StreamingHttpResponse
from .forms import UserCreationForm, LoginForm
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django import forms
import time

# FastAPI app URL (replace with the actual URL)
fastapi_url = "http://localhost:8000"

# Create a Django form for the user to input collection_name, loader_type, and upload a file
class FileUploadForm(forms.Form):
    collection_name = forms.CharField(max_length=100)
    loader_type = forms.ChoiceField(choices=[('text', 'Text'), ('csv', 'CSV'), ('pdf', 'PDF')])
    file = forms.FileField()

class SearchUploadForm(forms.Form):
    query = forms.CharField(max_length=100)
    max_results = forms.IntegerField()

# Create your views here.
# Home page
def index(request):
    return render(request, 'index.html')

# signup page
def user_signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
        else:
            print(form.errors)

    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})



# login page
def user_login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user:
                login(request, user)    
                return redirect('home')
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})

# logout page
def user_logout(request):
    logout(request)
    return redirect('login')

@login_required
def upload_file_to_fastapi(request):
    if request.method == 'POST':
        # Get the logged-in user's ID
        user_id = request.user.id

        form = FileUploadForm(request.POST, request.FILES)  # Bind form with POST data and uploaded files
        if form.is_valid():
            collection_name = form.cleaned_data['collection_name']
            loader_type = form.cleaned_data['loader_type']
            file = form.cleaned_data['file']

            # Prepare the data to send to the FastAPI endpoint
            data = {
                'user_id': user_id,
                'collection_name': collection_name,
                'loader_type': loader_type
            }

            files = {'file': (file.name, file.read())}

            response = requests.post(fastapi_url + '/upload/', data=data, files=files)
            if response.status_code == 200:
                messages.success(request, "File uploaded successfully to FastAPI.")
            else:
                messages.error(request, "Error uploading the file to FastAPI.")

    else:
        form = FileUploadForm()  # Create a blank form for GET requests

    return render(request, 'upload.html', {'form': form})

@login_required
def search_content(request):
    if request.method == 'POST':
        form = SearchUploadForm(request.POST)
        if form.is_valid():
            query = form.cleaned_data['query']
            max_results = form.cleaned_data['max_results']  # Ensure 'max_results' is included

            # Prepare the data to send to the FastAPI endpoint
            data = {
                'user_id': request.user.id,
                'query': query,
                'max_results': max_results,  # Include 'max_results' in the data
            }

            # Send a POST request to the FastAPI endpoint
            response = requests.post(fastapi_url + '/upload/search/', data=data)
            if response.status_code == 200:
                # Parse the JSON response from FastAPI
                search_results = response.json()

                # Render the HTML template with the search results
                return render(request, 'search.html', {'form': form, 'search_results': search_results})
            else:
                messages.error(request, "Error uploading content and performing the search.")

    else:
        form = SearchUploadForm()

    return render(request, 'search.html', {'form': form})

# A dictionary to store the form settings
form_settings = {
    "collection_name": "",
    "max_new_tokens": "",
    "repetition_penalty": "",
    "temperature": "",
    "gpu_layers": "",
    "context_length": "",
    "model_name": "",
    "k": "",
}

@require_POST
@csrf_exempt
def call_fastapi_llama2(request):
    try:
        if request.method == 'POST':
            # Extract data from the HTML form input
            form_settings["collection_name"] = request.POST.get('collection_name')
            form_settings["max_new_tokens"] = request.POST.get('max_new_tokens')
            form_settings["repetition_penalty"] = request.POST.get('repetition_penalty')
            form_settings["temperature"] = request.POST.get('temperature')
            form_settings["gpu_layers"] = request.POST.get('gpu_layers')
            form_settings["context_length"] = request.POST.get('context_length')
            form_settings["model_name"] = request.POST.get('model_name')
            form_settings["k"] = request.POST.get('k')

            # Initialize an empty conversation history
            conversation = []

            while True:
                user_query = request.POST.get('user_query')
                if user_query:
                    conversation.append({"role": "user", "content": user_query})

                # Define the request data to be sent to the FastAPI endpoint
                request_data = {
                    "collection_name": form_settings["collection_name"],
                    "search_kwargs": {"k": int(form_settings["k"])},
                    "query": user_query if user_query else "",
                    "config": {
                        "max_new_tokens": int(form_settings["max_new_tokens"]),
                        "repetition_penalty": float(form_settings["repetition_penalty"]),
                        "temperature": float(form_settings["temperature"]),
                        "gpu_layers": int(form_settings["gpu_layers"]),
                        "context_length": int(form_settings["context_length"])
                    },
                    "model_name": form_settings["model_name"],
                }

                fastapi_url = "http://localhost:8000/llama2_rag/"
                response = requests.post(fastapi_url, json=request_data)

                if response.status_code == 200:
                    response_data = response.json()
                    conversation.append({"role": "assistant", "content": response_data["response"]})
                    context = {"conversation": conversation, "form_settings": form_settings}
                    return render(request, 'llama2_form.html', context)
                
                else:
                    return JsonResponse({"error": "Failed to call FastAPI endpoint"}, status=500)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

def show_llama2_form(request):
    return render(request, 'llama2_form.html', {"form_settings": form_settings})

