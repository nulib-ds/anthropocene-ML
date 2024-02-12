import logging
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

# Configure logging
logging.basicConfig(filename='clustering.log', level=logging.INFO)

# Load 20 newsgroups dataset
newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Use TfidfVectorizer for BERT embeddings (you can replace it with BERT embeddings directly if needed)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(newsgroups_data.data)

# Use BERT for clustering
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Move model to CUDA
model = model.to('cuda')

# Set a maximum sequence length for BERT embeddings
max_seq_length = 128

# Function to get BERT embeddings for a given text
def get_bert_embeddings(text):
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors="pt")
    inputs = {key: value.to('cuda') for key, value in inputs.items()}  # Move inputs to CUDA
    outputs = model(**inputs)
    embeddings = outputs['last_hidden_state'].mean(dim=1).squeeze().detach().cpu().numpy()  # Move back to CPU
    return embeddings

# Get BERT embeddings for the newsgroup dataset
newsgroup_embeddings = [get_bert_embeddings(text) for text in newsgroups_data.data]

# Flatten the embeddings for KMeans
flat_embeddings = np.array([embedding.flatten() for embedding in newsgroup_embeddings])

# Use KMeans for clustering
kmeans = KMeans(n_clusters=20, random_state=42)
clusters = kmeans.fit_predict(flat_embeddings)

# Group documents by cluster
documents_per_cluster = [[] for _ in range(20)]  # Assuming 20 clusters
for i, cluster_label in enumerate(clusters):
    documents_per_cluster[cluster_label].append(newsgroups_data.data[i])

# Extract most frequent words for each cluster
cluster_names = []
for cluster_index, documents in enumerate(documents_per_cluster):
    vectorizer = CountVectorizer(max_features=10, stop_words='english')
    cluster_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    most_frequent_words = [str(feature_names[i]) for i in cluster_matrix.sum(axis=0).argsort()[0, ::-1]]
    cluster_names.append(", ".join(most_frequent_words))

# Print predicted labels with actual cluster names
for i, cluster_label in enumerate(clusters):
    actual_cluster_name = cluster_names[cluster_label]
    print(f"Document {i + 1}: Predicted Cluster {actual_cluster_name}")

# Save cluster labels with actual names to log file
with open('clustering.log', 'a') as f:
    for i, cluster_label in enumerate(clusters):
        actual_cluster_name = cluster_names[cluster_label]
        f.write(f"Document {i + 1}: Cluster {actual_cluster_name}\n")
        logging.info(f"Document {i + 1}: Cluster {actual_cluster_name}")
