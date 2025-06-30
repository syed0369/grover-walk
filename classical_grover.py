import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import deque, defaultdict

st.set_page_config(layout="wide")
st.title("Maze Solving using Quantum Walk vs DFS vs BFS")

# --- 1. Tree Input ---
st.sidebar.header("Define Tree Structure")
num_nodes = st.sidebar.number_input("Number of Nodes", min_value=2, max_value=20, value=7)

edges = []
for i in range(num_nodes - 1):
    u = st.sidebar.number_input(f"Edge {i+1} - From", 0, num_nodes - 1, i)
    v = st.sidebar.number_input(f"Edge {i+1} - To", 0, num_nodes - 1, i + 1)
    if u != v:
        edges.append((u, v))

start = st.sidebar.selectbox("Start Node", range(num_nodes))
goal = st.sidebar.selectbox("Goal Node", range(num_nodes))

G = nx.Graph()
G.add_nodes_from(range(num_nodes))
G.add_edges_from(edges)

# --- 2. DFS Path ---
def dfs_path(graph, start, goal):
    visited = set()
    stack = [(start, [start])]
    while stack:
        node, path = stack.pop()
        if node == goal:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor in reversed(list(graph.neighbors(node))):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
    return []

# --- 3. BFS Path ---
def bfs_path(graph, start, goal):
    visited = set()
    queue = deque([(start, [start])])
    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
    return []

dfs = dfs_path(G, start, goal)
bfs = bfs_path(G, start, goal)

# --- 4. Grover Walk ---
def grover_tree_walk(graph, start, goal, steps=100):
    Gx = graph.copy()
    sink = 'sink'
    Gx.add_edge(start, sink)
    Gx.add_edge(start, start)
    Gx.add_edge(goal, goal)

    arc_map = {}
    A = []
    idx = 0
    for u, v in Gx.edges():
        A.append((u, v))
        arc_map[(u, v)] = idx
        idx += 1
        if u != v and (v, u) not in arc_map:
            A.append((v, u))
            arc_map[(v, u)] = idx
            idx += 1

    num_arcs = len(A)
    U = np.zeros((num_arcs, num_arcs), dtype=complex)
    for i, (u, v) in enumerate(A):
        deg = Gx.degree[u]
        incoming = [arc_map[(x, u)] for x in Gx.neighbors(u) if (x, u) in arc_map]
        for j in incoming:
            U[i, j] += 2 / deg
        if (v, u) in arc_map:
            U[i, arc_map[(v, u)]] -= 1

    P = np.identity(num_arcs, dtype=complex)
    for i, (u, v) in enumerate(A):
        if sink in (u, v):
            P[i, i] = 0

    ζ = np.zeros(num_arcs, dtype=complex)
    ζ[arc_map[(start, start)]] = 1.0
    ζ /= np.linalg.norm(ζ)

    state = ζ.copy()
    for _ in range(steps):
        state = P @ (U @ state)

    # Exclude sink from probabilities
    node_probs = defaultdict(float)
    for i, (u, v) in enumerate(A):
        if v != sink:
            node_probs[v] += abs(state[i])**2

    # Sink probability (shown separately in metric)
    sink_prob = sum(abs(state[i])**2 for i, (u, v) in enumerate(A) if sink in (u, v))

    total_prob = sum(node_probs.values()) + sink_prob
    print("Total probability (should be ≈ 1):", total_prob)

    return dict(node_probs), Gx, sink_prob

# --- Run Grover Walk ---
grover_probs, Gx, absorbed_prob = grover_tree_walk(G, start, goal)

# --- Normalize for color scale ---
max_prob = max(grover_probs.values()) if grover_probs else 1
grover_probs_normalized = {k: v / max_prob for k, v in grover_probs.items()}

# --- Layout ---
pos = nx.spring_layout(Gx, seed=42)

# --- 5. Plotting ---
fig, ax = plt.subplots(figsize=(10, 5))
nx.draw(Gx, pos, ax=ax, with_labels=True, node_color='lightgrey', node_size=600, edge_color='gray')

def highlight(path, color):
    if len(path) < 2:
        return
    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_edges(Gx, pos, edgelist=path_edges, edge_color=color, width=3, ax=ax)
    nx.draw_networkx_nodes(Gx, pos, nodelist=path, node_color=color, node_size=700, ax=ax)

highlight(dfs, 'blue')
highlight(bfs, 'green')

# Color nodes by Grover probability (red to green), excluding sink
node_list = [n for n in Gx.nodes() if n != 'sink']
grover_colormap = [grover_probs_normalized.get(n, 0.0) for n in node_list]
nx.draw_networkx_nodes(Gx, pos, nodelist=node_list, node_color=grover_colormap,
                       cmap=plt.cm.RdYlGn_r, node_size=800, ax=ax)

st.pyplot(fig)

# --- 6. Comparison Metrics ---
st.subheader("Path Comparisons")
col1, col2, col3 = st.columns(3)
col1.metric("DFS Path Length", len(dfs) - 1)
col2.metric("BFS Path Length", len(bfs) - 1)
col3.metric("Absorbed into Sink", f"{absorbed_prob:.4f}")

st.write("**DFS Path:**", dfs)
st.write("**BFS Path:**", bfs)
st.write("**Grover Probabilities (Raw):**", grover_probs)
