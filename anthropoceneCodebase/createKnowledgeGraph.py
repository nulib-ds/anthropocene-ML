import spacy
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd 

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

def get_entities(sent):
    ent1 = ""
    ent2 = ""
    relation = ""

    doc = nlp(sent)
    for tok in doc:
        if tok.dep_.find("subj") == True:
            ent1 = tok.text
        if tok.dep_.find("obj") == True:
            ent2 = tok.text
        if tok.dep_.endswith("ROOT"):
            relation = tok.text

    return [ent1.strip(), relation, ent2.strip()]

def add_edges(graph, sentence):
    entities = get_entities(sentence)
    graph.add_edge(entities[0], entities[2], label=entities[1])

# Initialize a directed graph
G = nx.DiGraph()

# List of sentences on which knowledge graph will be built
with open('/home/ysc4337/aerith/anthropocene-reconcile/anthropocene-data/results/concatenated_captions.txt', 'r') as file:
    sentences = file.readlines()
sentences = [sentence.strip() for sentence in sentences]
print(sentences)
sentences = sentences['caption']


# Add edges to the graph
for sentence in sentences:
    add_edges(G, sentence)

# Visualize the graph
plt.figure(figsize=(10,10))
pos = nx.spring_layout(G, seed = 200)  # positions for all nodes
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, font_size = 15)

edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

# Get nodes with their degree
node_degrees = dict(G.degree())

# Sort nodes by degree
sorted_node_degrees = sorted(node_degrees.items(), key=lambda item: item[1], reverse=True)

print(sorted_node_degrees)

plt.savefig('/home/ysc4337/aerith/anthropocene-reconcile/anthropocene-data/results/graph.png')