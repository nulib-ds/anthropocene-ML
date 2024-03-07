import networkx as nx
import matplotlib.pyplot as plt
import string

# Your long string of text
with open('/home/ysc4337/aerith/anthropocene-reconcile/anthropocene-data/results/concatenated_captions.txt', 'r') as file:
    text = file.read().replace('\n', '')

# Clean the text
table = str.maketrans("", "", string.punctuation)
clean_text = text.translate(table).lower().split()

# Create a graph
G = nx.Graph()

# Add nodes and edges
for i in range(len(clean_text)-1):
    if G.has_edge(clean_text[i], clean_text[i+1]):
        # we added this one before, just increase the weight by one
        G[clean_text[i]][clean_text[i+1]]['weight'] += 1
    else:
        # new edge. add with weight=1
        G.add_edge(clean_text[i], clean_text[i+1], weight=1)

# Draw the graph
plt.figure(figsize=(10,10))
pos = nx.spring_layout(G) # k regulates the distance between nodes
nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_color='white', linewidths=1, font_size=15, pos=pos)
plt.savefig('/home/ysc4337/aerith/anthropocene-reconcile/anthropocene-data/results/graph.png')