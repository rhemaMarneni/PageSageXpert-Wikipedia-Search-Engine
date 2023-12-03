from django.shortcuts import render
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,lower
from pyspark.sql.types import StructType, StringType
from django.contrib.auth.forms import AuthenticationForm
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout
from .models import Question 
from collections import Counter
from pyspark.sql import functions as F
from django.contrib.auth.models import User
from django.db.models import Count
from django.db.models import Subquery, OuterRef
import nltk
#nltk.download('punkt')
import json
import pandas as pd

from nltk.tokenize import sent_tokenize
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer
import torch.optim as optim
from torch.nn.parallel import DataParallel
from django.http import JsonResponse


# Initialize Spark session and read CSV file
spark = SparkSession.builder.appName("DjangoSparkIntegration").getOrCreate()
page_df = spark.read.csv("pages_rank.csv", header=True, inferSchema=True)
link_df = spark.read.json("link_annotated_text.jsonl")
target_df = spark.read.csv("merged_subset.csv", header=True, inferSchema=True)

#all_titles = [row.title for row in page_df.collect()]

# Concatenate text from all sections for a given page_id
def concatenate_sections_text(page_id, sections):
    # print("Gathering page data of page_id, " + str(page_id))
    page_text = ""
    for section in sections:
        page_text += section["text"] + " "
    return page_text.strip()

# Tokenize the data
def tokenize_data(tokenizer, source_text, target_summary, max_length_source=512, max_length_summary=700):
    # print("Tokenizing data")
    inputs = tokenizer(source_text, return_tensors="pt", max_length=max_length_source, truncation=True)
    labels = tokenizer(target_summary, return_tensors="pt", max_length=max_length_summary, truncation=True).input_ids
    return inputs["input_ids"], inputs["attention_mask"], labels

# Generate and print the summary
def generate_and_print_summary(model, tokenizer, input_text, max_length=1000):
    # print("Generating summary")
    input_ids = tokenizer.encode("" + input_text, return_tensors="pt", max_length=max_length, truncation=True)
    summary_ids = model.generate(input_ids, max_length=max_length, length_penalty=2.0, num_beams=4, early_stopping=False)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # print("Completing sentences")
    sentences = sent_tokenize(summary)
    complete_sentences = sentences[:-2]
    final_summary = " ".join(complete_sentences)
    #print("Generated Summary:", final_summary)
    return final_summary
    
# Generate summary using parallel processing
def generate_summary_parallel(target_page_id):
#     print("Generate and print summary - parallel")
    target_page_data = link_df.filter(col("page_id") == target_page_id).collect()

    # Extract the text from all sections for the target page
    page_text = concatenate_sections_text(target_page_id, target_page_data[0]["sections"])

    # Initialize the BART model and tokenizer
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    # Tokenize the data
    input_ids, attention_mask, labels = tokenize_data(tokenizer, page_text, page_text)  # Using the same text for source and target (unsupervised)

    # Move model to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Wrap the model with DataParallel
    model = DataParallel(model)

    # Create DataLoader
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Generate and print the summary
    return generate_and_print_summary(model.module, tokenizer, page_text)  # Use model.module for DataParallel models
    

#recommendations = recommend_pages(user_search_history, all_titles)
def summary_view(request, page_id):
    # Retrieve the result based on the page_id
    #search_query = request.GET.get('q', '')
    #result = YourResultModel.objects.get(page_id=page_id)
    print("page_id is",page_id)
    summary = generate_summary_parallel(page_id)
    # print(summary)
    context = {
        'summary' : summary
    }
    # You can pass the result data to the template or perform other actions here
    
    return JsonResponse({'summary': summary})
    
def recommend_pages(user_search_history, page_df, num_recommendations=5):
    # Convert user_search_history to lowercase
    #user_search_history_lower = [query.lower() for query in user_search_history]
    user_search_history_lower = [query.lower() for query in user_search_history.split()]


    # Create a lowercase column for titles
    page_df_lower = page_df.withColumn("title_lower", F.lower(F.col("title")))

    # Filter the DataFrame based on user_search_history
    filtered_df = page_df_lower.filter(F.expr("title_lower RLIKE '(" + "|".join(user_search_history_lower) + ")'"))

    # Group by title and calculate the total similarity score
    similarity_df = filtered_df.groupBy("title").agg(
        *[F.sum(F.when(F.expr(f"title_lower RLIKE '{query}'"), 1).otherwise(0)).alias(f"similarity_score_{query}")
          for query in user_search_history_lower]
    )

    # Calculate the total similarity score
    total_similarity_cols = [F.col(f"similarity_score_{query}") for query in user_search_history_lower]
    similarity_df = similarity_df.withColumn("total_similarity", sum(total_similarity_cols))

    # Sort titles based on total similarity scores
    recommended_df = similarity_df.select("title", "total_similarity").orderBy(F.desc("total_similarity")).limit(num_recommendations)

    # Collect the recommended titles
    recommended_titles = [row.title.lower() for row in recommended_df.collect()]

    return recommended_titles


def login_or_register(request):
    login_form = AuthenticationForm(request, request.POST or None)
    register_form = UserCreationForm(request.POST or None)

    if request.method == 'POST':
        if 'login_submit' in request.POST and login_form.is_valid():
            user = login_form.get_user()
            login(request, user)
            return redirect('index')  # Replace 'home' with the URL you want to redirect to after login
        elif 'register_submit' in request.POST and register_form.is_valid():
            user = register_form.save()
            login(request, user)
            return redirect('index')  # Replace 'home' with the URL you want to redirect to after registration

    context = {
        'login_form': login_form,
        'register_form': register_form,
    }
    return render(request, 'login_or_register.html', context)
# def register(request):
#     if request.method == 'POST':
#         form = UserCreationForm(request.POST)
#         if form.is_valid():
#             user = form.save()
#             login(request, user)  # Log the user in after registration
#             return redirect('home')  # Replace 'home' with the URL you want to redirect to after registration
#     else:
#         form = UserCreationForm()

#     return render(request, 'register.html', {'form': form})

# def user_login(request):
#     if request.method == 'POST':
#         form = AuthenticationForm(request, request.POST)
#         if form.is_valid():
#             user = form.get_user()
#             login(request, user)
#             return redirect('home')  # Replace 'home' with the URL you want to redirect to after login
#     else:
#         form = AuthenticationForm()

#     return render(request, 'login.html', {'form': form})

def wikipedia_url_from_title(title):
    return 'https://en.wikipedia.org/wiki/{}'.format(title.replace(' ', '_'))
def index(request):
    # Retrieve the 'q' parameter from the URL
    # search_query = request.GET.get('q', '')

    # #Filter page_df based on the search_query
    # filtered_df = page_df.filter(col("title").like(f"%{search_query}%"))

    # # Convert the filtered DataFrame to a list of dictionaries
    # result_list = filtered_df.collect()

    # print("frwg",result_list)

    # # Render your template with the context
    # context = {
    #     'search_query': search_query,
    #     'result_list': result_list,
    # }
    

    user = User.objects.get(username=request.user.username)
    user_questions = Question.objects.filter(user=user).order_by('-timestamp')[:2]
    question_text_list = [question.question_text for question in user_questions]

    #user_questions = Question.objects.filter(user=user).values('question_text').annotate(count=Count('question_text')).order_by('-timestamp')[:5]
    # user_questions = Question.objects.filter(user=user).values('question_text').annotate(count=Count('question_text')).order_by('-timestamp')[:5]
    
    user_search_history = ["machine learning"]
    #user_questions_list = list(user_questions)
    # print("-", list(set(question_text_list)))
    user_search_history = list(set(question_text_list))

    print("history",user_search_history)
    final_reco = []
    for i in user_search_history:
        recommendations = recommend_pages(i, page_df)
        final_reco.extend(recommendations)

    
    # print("Recommended Pages:",final_reco)
    
    context = {
        'user': request.user, 
        'recommendations': final_reco
    }

    return render(request, 'index.html',context)
def logout_view(request):
    logout(request)
    return redirect('login_or_register')

def question(request):
    search_query = request.GET.get('q', '')
    if request.user.is_authenticated and search_query:
        question_instance = Question(user=request.user, question_text=search_query)
        question_instance.save()
    # Filter page_df based on the search_query
    filtered_df = page_df.filter(col("title").like(f"%{search_query}%"))

    # Check for an exact match
    filtered_df_exact = page_df.filter(lower(col("title")) == search_query)
    
    # Initialize result_list as an empty list
    result_list = []

    if filtered_df_exact.count() != 0:
        # If there is an exact match, take that match and 9 other pages from filtered_df_exact
        result_list = filtered_df_exact.collect()
        result_list.extend(filtered_df.filter(col("title") != search_query).orderBy("pagerank", ascending=False).take(9))

    else:
        # If there is no exact match, take the top 10 pages with the highest pagerank from filtered_df
        result_list = filtered_df.orderBy("pagerank", ascending=False).take(10)  ######################

    # Convert the result_list to a list of dictionaries with URLs
    result_list_with_urls = []
    for row in result_list:
        row_dict = row.asDict()
        row_dict['url'] = wikipedia_url_from_title(row_dict['title'])
        result_list_with_urls.append(row_dict)

    # Render your template with the context
    page_ids_list = [result['page_id'] for result in result_list_with_urls]
    # print("page_id list",page_ids_list)
    selected_target_rows = target_df.filter(col("source").isin(page_ids_list))
    selected_target_rows_list = [
            {'source': row['source'], 'target_id': row['target_id']} for row in selected_target_rows.collect()
        ]
    #print(selected_target_rows.collect())
    context = {
        'search_query': search_query,
        'result_list': result_list_with_urls,
        'user': request.user,
        'selected_target_rows': selected_target_rows_list
    }
    # print(context)
    return render(request, 'answer.html', context)

def graph_view(request):
    return render(request, 'graph.html')
