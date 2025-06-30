import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
edges = [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5), (5, 6)]
G.add_edges_from(edges)

start_node = 0
goal_node = 6
sink_node = 'sink'


G.add_edge(start_node, start_node)
G.add_edge(goal_node, goal_node)

G.add_edge(start_node, sink_node)

A = []
arc_map = {}
index = 0
for u, v in G.edges():
    A.append((u, v))
    arc_map[(u, v)] = index
    index += 1
    if u != v and (v, u) not in arc_map:
        A.append((v, u))
        arc_map[(v, u)] = index
        index += 1

arc_list = list(arc_map.keys())
num_arcs = len(arc_list)

def grover_operator(G, arc_list, arc_map):
    U = np.zeros((num_arcs, num_arcs), dtype=complex)
    for i, (u, v) in enumerate(arc_list):
        deg = G.degree[u]
        incoming_arcs = []
        for x in G.neighbors(u):
            if (x, u) in arc_map:
                incoming_arcs.append(arc_map[(x, u)])
        for j in incoming_arcs:
            U[i, j] += 2 / deg
        if (v, u) in arc_map:
            reverse = arc_map[(v, u)]
            U[i, reverse] -= 1
    return U

U = grover_operator(G, arc_list, arc_map)

P = np.identity(num_arcs, dtype=complex)
for a, (u, v) in enumerate(arc_list):
    if u == sink_node or v == sink_node:
        P[a, a] = 0

state_vector = np.zeros(num_arcs, dtype=complex)
state_vector[arc_map[(start_node, start_node)]] = 1.0

num_steps = 50
state = state_vector.copy()
prob_dist = []

for t in range(num_steps):
    state = P @ (U @ state)
    mu = {node: 0 for node in G.nodes() if node != sink_node}
    for a, (u, v) in enumerate(arc_list):
        if v != sink_node:
            mu[v] += abs(state[a]) ** 2
    prob_dist.append(mu)

final_probs = prob_dist[-1]

pos = nx.spring_layout(G, seed=42)
node_colors = [final_probs.get(n, 0) for n in G.nodes()]

# Normalize values between 0 and 1 for color mapping
color_values = [final_probs.get(n, 0) for n in G.nodes()]
max_val = max(color_values)
min_val = min(color_values)
norm = [1 - (v - min_val) / (max_val - min_val + 1e-9) for v in color_values]

# ---- Plotting ----
plt.figure(figsize=(10, 4))

# --- Graph with color-coded nodes: Red (low) → Green (high) ---
plt.subplot(1, 2, 1)
nx.draw(
    G, pos, with_labels=True,
    node_color=norm,
    cmap=plt.cm.RdYlGn_r,  # red-yellow-green reversed: green = high
    node_size=800,
    font_color='white'
)
plt.title("Final Probability Distribution on Maze")

# --- Bar chart ---
plt.subplot(1, 2, 2)
labels = list(final_probs.keys())
values = [final_probs[k] for k in labels]
colors = plt.cm.RdYlGn_r(norm)  # same colormap as graph

bars = plt.bar(labels, values, color=colors)
plt.title("Final Probabilities per Node")
plt.xlabel("Node")
plt.ylabel("Probability")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
