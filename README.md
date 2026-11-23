# ML-Based Intelligent Routing Optimization

**Computer Networks Project**

**Team members**
- B Kavin (24PD07)
- Gomathi S (24PD12)

## 1. Project overview
This project simulates a computer network and compares traditional routing with a machine-learning-assisted route selection approach. Each network link has four measurements:

- Latency (milliseconds)
- Bandwidth (Mbps)
- Congestion (0 to 1)
- Packet loss (percentage)

A Random Forest regression model learns to estimate a combined link cost. Dijkstra and A* use latency as their edge weight, while ML-based routing uses the model-predicted cost. The program reports each selected path and compares latency, estimated packet loss, estimated throughput, and average congestion.

## 2. Technologies
- Python
- NetworkX
- Scikit-learn
- Pandas and NumPy
- Matplotlib and Seaborn

## 3. Requirements
Python 3.10 or later is recommended.

Install dependencies:
```bash
pip install -r requirements.txt
```

## 4. Run
Open a terminal in this folder and run:
```bash
python main.py
```

The program trains the model, builds the simulated network, finds routes from node 1 to node 10, prints the comparison, and creates the `outputs` folder.

## 5. Project workflow
1. Generate synthetic network-link measurements.
2. Create a target cost from latency, congestion, bandwidth, and packet loss.
3. Train and test a Random Forest regression model.
4. Build a network topology using NetworkX.
5. Calculate Dijkstra, A*, and ML-based routes.
6. Calculate route-level performance metrics.
7. Save comparison tables and plots.

## 6. Algorithms
### Dijkstra
Finds a minimum-weight path. In this project, the weight is link latency.

### A* Search
Finds a minimum-weight path using a heuristic. A zero heuristic is used so that A* remains optimal without assuming geographic coordinates or inventing a distance estimate. In this setting, A* behaves like Dijkstra in path optimality.

### ML-based routing
The Random Forest model predicts a cost for each link based on all four link measurements. Dijkstra's shortest-path algorithm then finds the path with the lowest sum of predicted link costs. The model does not directly select a route; it supplies the edge weights used by the routing algorithm.

## 7. Evaluation metrics
- **Total latency:** sum of link latencies on the route.
- **Estimated packet loss:** probability of losing a packet on at least one link, assuming independent link losses.
- **Estimated throughput:** minimum bandwidth along the route, adjusted for estimated packet loss.
- **Average congestion:** mean congestion score of links in the route.
- **Hops:** number of links in the route.
- **MAE and R-squared:** measures of the model's cost-prediction accuracy on a held-out synthetic test set.

## 8. Output files
The program creates:
- `synthetic_training_data.csv`
- `route_comparison.csv`
- `network_routes.png`
- `latency_ms_comparison.png`
- `packet_loss_pct_comparison.png`
- `throughput_mbps_comparison.png`
- `avg_congestion_comparison.png`

## 9. Important limitations
- All network measurements are simulated; they are not collected from a real network.
- The training labels are generated from a designed cost formula. This makes the project a demonstration of ML-assisted routing rather than proof that ML will outperform traditional routing on a real network.
- A* uses a zero heuristic for correctness. A future extension could use a valid lower-bound heuristic if a suitable topology-distance model is added.
- The throughput value is a simplified estimate, not a packet-level measurement.
- This is a simulation and does not change operating-system routing tables or forward live packets.

## 10. Future enhancements
- Change link congestion and latency during a simulation to model changing network conditions.
- Train on measurements exported from a network simulator or testbed.
- Add packet-level simulation using a dedicated simulator such as Mininet or ns-3.
- Add multiple source/destination pairs and repeated trials.
- Compare against Bellman-Ford or other routing approaches.
- Add a simple command-line menu for selecting source and destination nodes.
