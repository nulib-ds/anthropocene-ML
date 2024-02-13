
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import numpy as np

# Load 20 newsgroups dataset
newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Use TfidfVectorizer for BERT embeddings (you can replace it with BERT embeddings directly if needed)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(newsgroups_data.data)

# Use BERT for clustering
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Set a maximum sequence length for BERT embeddings
max_seq_length = 128

# Function to get BERT embeddings for a given text
def get_bert_embeddings(text):
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs['last_hidden_state'].mean(dim=1).squeeze().detach().numpy()
    return embeddings

# Get BERT embeddings for the newsgroup dataset
newsgroup_embeddings = [get_bert_embeddings(text) for text in newsgroups_data.data]

# Flatten the embeddings for KMeans
flat_embeddings = np.array([embedding.flatten() for embedding in newsgroup_embeddings])

# Use KMeans for clustering
kmeans = KMeans(n_clusters=20, random_state=42)
clusters = kmeans.fit_predict(flat_embeddings)

# Print cluster labels for each document
for i, cluster_label in enumerate(clusters):
    print(f"Document {i + 1}: Cluster {cluster_label}")
