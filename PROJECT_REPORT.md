# Project Report: ML-Based Intelligent Routing Optimization

## Abstract
This project proposes an ML-Based Intelligent Routing Optimization System that combines conventional routing algorithms with machine learning to support route selection using multiple network conditions. The simulated network records latency, bandwidth, congestion, and packet loss for each link. A Random Forest regression model estimates a combined link cost from these parameters. Dijkstra and A* are evaluated using latency-based weights, while the ML-assisted approach uses predicted link costs. The selected routes are compared using total latency, estimated packet loss, estimated throughput, average congestion, and hop count. The project demonstrates how machine learning can be incorporated into a routing pipeline in a controlled simulation environment.

## 1. Introduction
Routing is the process of selecting a path for data to travel from a source to a destination. Traditional shortest-path methods commonly use a fixed metric such as link cost or latency. However, network performance depends on several factors at once. A path with low latency may contain a congested link or have a high packet-loss rate. This project demonstrates a simple ML-assisted approach that considers multiple link attributes when calculating routing cost.

## 2. Problem statement
A route chosen using only one metric may not be the most suitable route when congestion, bandwidth, and packet loss also affect network quality. The project builds a simulated network and evaluates whether using a predicted multi-metric link cost changes route selection compared with latency-only routing.

## 3. Objectives
1. Simulate a network topology with multiple paths.
2. represent link conditions using latency, bandwidth, congestion, and packet loss.
3. Train a Random Forest model to estimate a combined link cost.
4. Find routes using Dijkstra, A*, and ML-assisted routing.
5. Compare routes using consistent performance metrics.
6. Visualize the network paths and comparison results.

## 4. System design
**Input:** simulated link metrics and a source/destination pair.

**Processing:**
1. Generate synthetic training samples.
2. calculate a designed target cost for each sample.
3. train and test a Random Forest regressor.
4. apply the trained model to network links.
5. run the routing algorithms.
6. compute route-level metrics.

**Output:** route paths, model scores, CSV results, and plots.

## 5. Methodology
### 5.1 Synthetic data
Synthetic examples are generated for latency, congestion, bandwidth, and packet loss. A weighted cost formula creates the target value for supervised learning. The data is split into training and testing subsets.

### 5.2 Machine learning
A Random Forest Regressor learns the mapping from the four link measurements to the target cost. MAE and R-squared are reported on held-out samples.

### 5.3 Routing
- Dijkstra minimizes the sum of link latencies.
- A* uses latency as its edge weight and a zero heuristic to guarantee optimality without an unsupported distance estimate.
- ML-assisted routing uses the model's predicted link cost as the edge weight and finds the minimum-cost path.

### 5.4 Metrics
Total latency is the sum of link latencies. Estimated end-to-end packet loss is calculated from independent per-link loss probabilities. Estimated throughput is the bottleneck link bandwidth adjusted for packet loss. Average congestion is the mean congestion value of the route's links. These are simplified simulation metrics.

## 6. Implementation
The project is implemented in `main.py`. NetworkX represents the topology and computes shortest paths. Pandas and NumPy manage the data. Scikit-learn trains the Random Forest model. Matplotlib and Seaborn generate visual comparisons.

## 7. How to execute
1. Install Python 3.10 or later.
2. Run `pip install -r requirements.txt`.
3. Run `python main.py`.
4. Review the terminal output and the files created in the `outputs` folder.

## 8. Results
When the program is executed, it prints the route selected by each algorithm and reports its hop count, total latency, estimated packet loss, estimated throughput, and average congestion. It also saves a CSV table and charts. The actual values should be taken from the program's output rather than written in advance because they depend on the model fit and simulated data.

## 9. Conclusion
The project provides a working simulation of ML-assisted route selection. It shows how a machine-learning model can estimate a link cost from multiple network conditions and feed that estimate into a conventional shortest-path algorithm. The approach is a prototype and does not establish real-world performance gains because both the training data and the network measurements are synthetic. Real-world validation would require measurements from a network testbed or a packet-level simulator.

## 10. Future scope
- Simulate changing congestion and latency over time.
- Run experiments across many source/destination pairs.
- Use real or packet-level simulated measurements.
- Add repeated trials and statistical comparison.
- Evaluate additional routing algorithms and valid A* heuristics.
