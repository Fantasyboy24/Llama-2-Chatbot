# Llama 2 Chatbot 

## Project Overview
This project is a web application that combines FastAPI and Django to manage document collections and perform various document-related operations. It allows users to upload documents, search for content on the web and store the results in a database, and perform similarity searches on stored documents. The project also includes a feature for leveraging language models for question answering and retrieval.

## Table of Contents
Project Overview
Prerequisites
Installation
Usage
File Upload
Web Search
Similarity Search
Llama2 Question Answering
Deployment
Environment Variables
Database Setup
Contributing
License
Prerequisites

### Before you begin, ensure you have met the following requirements:
Python 3.x installed on your system.
Docker installed if you plan to use the provided PostgreSQL database container.
FastAPI and Django knowledge.
Basic knowledge of working with PostgreSQL databases.


#### Installation
Clone the project repository to your local machine:

``` git clone <repository_url> ```
Install the required Python packages using pip:

``` pip install -r requirements.txt ```
Create a PostgreSQL database using Docker (optional):

If you don't already have a PostgreSQL database, you can create one using Docker. Simply run the following command in the project root directory:

``` docker-compose up -d ```
This will create a PostgreSQL database container with the necessary environment variables.

### Usage
The project provides several features that can be accessed through the web interface.

### File Upload
To upload a document file:

Access the "Upload Document" page.

Provide a collection name, choose the loader type (Text, CSV, or PDF), and select the file to upload.

Click the "Upload" button.

The document will be processed, and its content will be stored in the database.

### Web Search
To perform a web search and store the results:

Access the "Web Search" page.

Enter a search query and specify the maximum number of results to retrieve.

Click the "Search" button.

The search results will be displayed, and you can choose to store them in the database. The documents will be processed and stored.

### Similarity Search
To perform a similarity search on stored documents:

Access the "Similarity Search" page.

Provide the collection name and a query.

Click the "Search" button.

The system will return documents from the database that are most similar to the query.

### Llama2 Question Answering
The "LLAMA2 Question Answering" feature utilizes language models for question answering and retrieval. It provides a web form to interact with the models. You can adjust various parameters such as context length, repetition penalty, and temperature.

Access the "LLAMA2 Question Answering" page.

Set the collection name, model name, and other parameters.

Enter a user query in the input field.

Click the "Submit" button.

The system will provide responses to the user's queries using the specified language model and retrieval settings.

### Deployment
To deploy this project in a production environment, consider using a production-ready web server like Gunicorn for FastAPI and a WSGI server like uWSGI for Django. Configure the appropriate database and set environment variables.

### Environment Variables
The project uses environment variables for database and other configuration. You can set these environment variables in a .env file:

DATABASE_URL: The PostgreSQL database connection URL.
FASTAPI_HOST: The host where FastAPI will run.
FASTAPI_PORT: The port for FastAPI.
DJANGO_SECRET_KEY: Django secret key for secure sessions.
DDG_SEARCH_API_KEY: API key for DuckDuckGo search (if used).
DEBUG: Set to True for debugging mode.
Database Setup
If you are not using the provided Docker PostgreSQL container, make sure to set up your PostgreSQL database. Create a database with a name that matches the DATABASE_URL configuration in your .env file. You should also configure the username and password according to your database setup.

### Contributing
Feel free to contribute to this project by submitting pull requests and issues.

### License
This project is licensed under the MIT License - see the LICENSE file for details