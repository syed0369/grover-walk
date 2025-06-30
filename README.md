# Maze Solving via Grover Walk vs Classical Methods
## Background
Grover's algorithm is a quantum search technique. Inspired by it, this simulation applies Grover-like evolution operators on tree graphs to highlight potential quantum speedups or behavior differences in search problems.

This project explores maze (tree) solving using both classical and quantum-inspired approaches. It visualizes and compares the performance of:

- **Depth-First Search (DFS)**
- **Breadth-First Search (BFS)**
- **Grover-inspired Quantum Walk (via matrix simulation and Cirq)**

The core aim is to illustrate the behavior of quantum walks versus classical search algorithms on graphs.

---

## File descriptions

| File | Description |
|------|-------------|
| `classical_grover.py` | Interactive Streamlit app for exploring Grover Walk vs DFS/BFS. |
| `grover.py` | Static Grover walk simulation with probability visualizations using matplotlib. |
| `quantum_simulation.py` | Cirq-based quantum evolution of Grover walk on a tree structure. |

---

## Run the project files

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/grover-walk-visualizer.git
cd grover-walk-visualizer
```
### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```
### Step 3: Launch Streamlit or graph (acyclic, connected) visualization of statically typed tree structured maze
#### For steamlit app
```bash
streamlit run classical_grover.py
```
#### For statically typed data and plot of probabilities
```bash
python grover.py
python quantum_simulation.py
```

## Features

- Tree structure input via GUI
- Visual comparison of DFS, BFS paths
- Quantum probability-based Grover walk simulation
- Probability plots using node color coding and bar charts
- Cirq quantum simulation for log-encoded arc walks

## Technologies Used

- Python
- Streamlit
- NumPy, Matplotlib, NetworkX
- Cirq (for quantum simulation)

## Author

[Syed Umair](https://github.com/syed0369)

Open to collaboration and suggestions! Feel free to fork and extend the simulation for more graph types or search comparisons.




