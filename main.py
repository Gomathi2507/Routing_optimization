import os
import random
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_training_data(samples=3000):
    """Create synthetic link measurements and a target network cost."""
    latency = np.random.uniform(2, 120, samples)             # milliseconds
    congestion = np.random.uniform(0.0, 1.0, samples)        # 0 to 1
    bandwidth = np.random.uniform(5, 1000, samples)          # Mbps
    packet_loss = np.random.uniform(0.0, 8.0, samples)       # percentage

    # Lower is better. The formula creates labels for a supervised demo.
    # Congestion and packet loss increase cost; bandwidth reduces cost.
    cost = (
        0.45 * latency
        + 28.0 * congestion
        + 1.8 * packet_loss
        + 120.0 / (bandwidth + 10.0)
    )

    # Small noise makes the learning task less perfectly deterministic.
    cost += np.random.normal(0, 1.5, samples)
    cost = np.maximum(cost, 0.1)

    data = pd.DataFrame({
        "latency_ms": latency,
        "congestion": congestion,
        "bandwidth_mbps": bandwidth,
        "packet_loss_pct": packet_loss,
        "target_cost": cost
    })
    return data


def train_model():
    """Train and evaluate a Random Forest regression model."""
    data = generate_training_data()
    features = ["latency_ms", "congestion", "bandwidth_mbps", "packet_loss_pct"]
    X = data[features]
    y = data["target_cost"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        random_state=SEED,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    data.to_csv(os.path.join(OUTPUT_DIR, "synthetic_training_data.csv"), index=False)
    print("\n=== ML Model Evaluation ===")
    print(f"Training samples       : {len(X_train)}")
    print(f"Testing samples        : {len(X_test)}")
    print(f"Mean Absolute Error    : {mae:.3f}")
    print(f"R-squared              : {r2:.3f}")

    return model, features, {"MAE": mae, "R2": r2}


def build_network():
    """Create a small undirected network with realistic link metrics."""
    graph = nx.Graph()
    nodes = list(range(1, 11))
    graph.add_nodes_from(nodes)

    # Each tuple: (node A, node B, latency ms, bandwidth Mbps,
    #              congestion 0..1, packet loss %)
    links = [
        (1, 2, 12, 180, 0.20, 0.4),
        (1, 3, 18, 500, 0.12, 0.2),
        (1, 4, 10, 80, 0.62, 1.2),
        (2, 3, 8, 150, 0.25, 0.5),
        (2, 5, 16, 300, 0.18, 0.3),
        (3, 4, 7, 220, 0.30, 0.4),
        (3, 5, 11, 650, 0.10, 0.1),
        (3, 6, 20, 450, 0.22, 0.4),
        (4, 6, 9, 90, 0.58, 1.4),
        (5, 6, 6, 250, 0.16, 0.2),
        (5, 7, 14, 700, 0.08, 0.1),
        (6, 7, 8, 200, 0.35, 0.6),
        (6, 8, 17, 550, 0.18, 0.3),
        (7, 8, 5, 350, 0.14, 0.2),
        (7, 9, 13, 120, 0.45, 0.9),
        (8, 9, 9, 600, 0.12, 0.2),
        (8, 10, 15, 400, 0.20, 0.3),
        (9, 10, 7, 170, 0.28, 0.5),
        (5, 10, 28, 900, 0.06, 0.1),
    ]

    for u, v, latency, bandwidth, congestion, packet_loss in links:
        graph.add_edge(
            u, v,
            latency_ms=float(latency),
            bandwidth_mbps=float(bandwidth),
            congestion=float(congestion),
            packet_loss_pct=float(packet_loss)
        )

    return graph


def apply_link_costs(graph, model, features):
    """Attach baseline latency cost and ML-predicted cost to every link."""
    for u, v, attrs in graph.edges(data=True):
        row = pd.DataFrame([{
            "latency_ms": attrs["latency_ms"],
            "congestion": attrs["congestion"],
            "bandwidth_mbps": attrs["bandwidth_mbps"],
            "packet_loss_pct": attrs["packet_loss_pct"]
        }])[features]

        predicted_cost = float(model.predict(row)[0])
        attrs["latency_cost"] = attrs["latency_ms"]
        attrs["ml_cost"] = max(predicted_cost, 0.1)


def astar_heuristic(graph, node, destination):
    """
    A* heuristic for latency-weighted routing.
    The value is deliberately zero, making A* equivalent to Dijkstra in
    optimality while still demonstrating the A* API without unsafe guesses.
    """
    return 0


def route_metrics(graph, path):
    """Calculate combined metrics for a selected route."""
    total_latency = 0.0
    total_cost = 0.0
    loss_survival = 1.0
    bottleneck_bandwidth = float("inf")
    total_congestion = 0.0

    for i in range(len(path) - 1):
        attrs = graph[path[i]][path[i + 1]]
        total_latency += attrs["latency_ms"]
        total_cost += attrs["ml_cost"]
        loss_survival *= (1.0 - attrs["packet_loss_pct"] / 100.0)
        bottleneck_bandwidth = min(bottleneck_bandwidth, attrs["bandwidth_mbps"])
        total_congestion += attrs["congestion"]

    packet_loss = (1.0 - loss_survival) * 100.0
    average_congestion = total_congestion / max(len(path) - 1, 1)
    estimated_throughput = bottleneck_bandwidth * (1.0 - packet_loss / 100.0)
    return {
        "hops": len(path) - 1,
        "latency_ms": total_latency,
        "packet_loss_pct": packet_loss,
        "throughput_mbps": estimated_throughput,
        "avg_congestion": average_congestion,
        "ml_cost": total_cost
    }


def choose_routes(graph, source, destination):
    """Compute routes with Dijkstra, A*, and ML-based Dijkstra."""
    dijkstra_path = nx.shortest_path(
        graph, source, destination, weight="latency_cost", method="dijkstra"
    )
    astar_path = nx.astar_path(
        graph, source, destination,
        heuristic=lambda n, target: astar_heuristic(graph, n, target),
        weight="latency_cost"
    )
    ml_path = nx.shortest_path(
        graph, source, destination, weight="ml_cost", method="dijkstra"
    )
    return {
        "Dijkstra (latency)": dijkstra_path,
        "A* (latency)": astar_path,
        "ML-based routing": ml_path
    }


def draw_network(graph, routes, source, destination):
    """Save a network diagram highlighting the selected paths."""
    pos = nx.spring_layout(graph, seed=SEED)
    plt.figure(figsize=(11, 7))
    nx.draw_networkx_nodes(graph, pos, node_size=750)
    nx.draw_networkx_labels(graph, pos, font_size=10)
    nx.draw_networkx_edges(graph, pos, width=1.5, edge_color="gray", alpha=0.65)

    edge_labels = {
        (u, v): f'{attrs["latency_ms"]:.0f}ms'
        for u, v, attrs in graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7)

    styles = {
        "Dijkstra (latency)": ("blue", "solid"),
        "A* (latency)": ("green", "dashed"),
        "ML-based routing": ("red", "dashdot")
    }
    # Draw paths one by one. Overlapping routes may hide each other, but the
    # printed route listing remains the authoritative path output.
    for name, path in routes.items():
        color, style = styles[name]
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(
            graph, pos, edgelist=path_edges, width=3.5,
            edge_color=color, style=style, alpha=0.8
        )

    plt.title(f"Network Topology and Routes ({source} → {destination})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "network_routes.png"), dpi=180)
    plt.close()


def make_comparison_chart(results_df):
    """Create a chart comparing route metrics."""
    metrics = [
        ("latency_ms", "Total latency (ms)"),
        ("packet_loss_pct", "Estimated packet loss (%)"),
        ("throughput_mbps", "Estimated throughput (Mbps)"),
        ("avg_congestion", "Average congestion (0–1)")
    ]

    for column, title in metrics:
        plt.figure(figsize=(8, 5))
        plt.bar(results_df["Algorithm"], results_df[column])
        plt.title(title + " by routing algorithm")
        plt.xticks(rotation=12, ha="right")
        plt.tight_layout()
        filename = column.replace("%", "pct")
        plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}_comparison.png"), dpi=180)
        plt.close()


def main():
    print("=" * 66)
    print(" ML-BASED INTELLIGENT ROUTING OPTIMIZATION")
    print(" Computer Networks Project | Simulation")
    print("=" * 66)

    model, features, model_scores = train_model()
    graph = build_network()
    apply_link_costs(graph, model, features)

    source = 1
    destination = 10
    routes = choose_routes(graph, source, destination)

    rows = []
    print(f"\n=== Route Comparison: Source {source} → Destination {destination} ===")
    for algorithm, path in routes.items():
        metrics = route_metrics(graph, path)
        row = {"Algorithm": algorithm, "Path": " -> ".join(map(str, path)), **metrics}
        rows.append(row)
        print(f"\n{algorithm}")
        print(f"  Path                : {row['Path']}")
        print(f"  Hops                : {metrics['hops']}")
        print(f"  Total latency       : {metrics['latency_ms']:.2f} ms")
        print(f"  Estimated packet loss: {metrics['packet_loss_pct']:.2f}%")
        print(f"  Estimated throughput: {metrics['throughput_mbps']:.2f} Mbps")
        print(f"  Average congestion  : {metrics['avg_congestion']:.3f}")
        if algorithm == "ML-based routing":
            print(f"  Total predicted cost: {metrics['ml_cost']:.2f}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "route_comparison.csv"), index=False)
    draw_network(graph, routes, source, destination)
    make_comparison_chart(results_df)

    print("\n=== Files Created ===")
    print(f"- {OUTPUT_DIR}/synthetic_training_data.csv")
    print(f"- {OUTPUT_DIR}/route_comparison.csv")
    print(f"- {OUTPUT_DIR}/network_routes.png")
    print(f"- {OUTPUT_DIR}/latency_ms_comparison.png")
    print(f"- {OUTPUT_DIR}/packet_loss_pct_comparison.png")
    print(f"- {OUTPUT_DIR}/throughput_mbps_comparison.png")
    print(f"- {OUTPUT_DIR}/avg_congestion_comparison.png")
    print("\nNote: results are based on synthetic simulated network data.")
    print("The ML model learns a link-cost estimate; this is not live network routing.")


if __name__ == "__main__":
    main()
