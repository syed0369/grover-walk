import cirq
import numpy as np
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt

num_nodes = 4
start = 0
goal = 3
steps = 50

edges = [(0, 1), (1, 2), (1, 3)]
G = nx.DiGraph()
for u, v in edges:
    G.add_edge(u, v)
    G.add_edge(v, u)
G.add_edge(start, start)
G.add_edge(goal, goal)
sink = 'sink'
G.add_edge(start, sink)

arc_list = []
arc_index = {}
idx = 0
for u, v in G.edges():
    arc_list.append((u, v))
    arc_index[(u, v)] = idx
    idx += 1
num_arcs = len(arc_list)
log_arcs = int(np.ceil(np.log2(num_arcs)))

qubits = cirq.LineQubit.range(log_arcs)

init_state = np.zeros(2**log_arcs, dtype=complex)
if (start, start) in arc_index:
    init_state[arc_index[(start, start)]] = 1.0

# --- Define Grover coin operator ---
U = np.zeros((num_arcs, num_arcs), dtype=complex)
for i, (u, v) in enumerate(arc_list):
    deg = G.out_degree(u)
    incoming = [arc_index[(x, u)] for x in G.predecessors(u) if (x, u) in arc_index]
    for j in incoming:
        U[i, j] += 2 / deg
    if (v, u) in arc_index:
        U[i, arc_index[(v, u)]] -= 1

# --- Projection operator (remove sink) ---
P = np.identity(num_arcs, dtype=complex)
for i, (u, v) in enumerate(arc_list):
    if sink in (u, v):
        P[i, i] = 0

# --- Simulate evolution ---
state = np.zeros(num_arcs, dtype=complex)
state[arc_index[(start, start)]] = 1.0
state = state / np.linalg.norm(state)

for _ in range(steps):
    state = P @ (U @ state)

# --- Collect final probabilities ---
node_probs = defaultdict(float)
for i, (u, v) in enumerate(arc_list):
    if v != sink:
        node_probs[v] += abs(state[i])**2

# --- Display results ---
print("Final node probabilities:")
for node, prob in node_probs.items():
    print(f"Node {node}: {prob:.4f}")

# --- Side-by-side plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Bar chart: Grover probabilities ---
ax1.bar(node_probs.keys(), node_probs.values(), color='skyblue')
ax1.set_title("Grover Walk Node Probabilities")
ax1.set_xlabel("Node")
ax1.set_ylabel("Probability")
ax1.set_ylim(0, 1)

# --- Graph layout ---
pos = nx.spring_layout(G, seed=42)

# Draw nodes
nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=700, node_color='lightblue')

# Draw edges as arrows
nx.draw_networkx_edges(
    G, pos, ax=ax2,
    edgelist=G.edges(),
    edge_color='gray',
    arrows=True,
    arrowstyle='->',
    arrowsize=20,
    connectionstyle='arc3,rad=0.1'
)

# Draw labels
nx.draw_networkx_labels(G, pos, ax=ax2, font_size=12)

# Highlight start and goal
nx.draw_networkx_nodes(G, pos, nodelist=[start], node_color='green', node_size=800, ax=ax2, label='Start')
nx.draw_networkx_nodes(G, pos, nodelist=[goal], node_color='red', node_size=800, ax=ax2, label='Goal')

ax2.set_title("Directed Graph (Maze Structure)")
ax2.axis('off')

plt.tight_layout()
plt.show()
