"""
This script computes key statistics for the KG, MEDAKA including:
- Node and edge counts
- Predicate-wise triple and object counts
- Degree distribution stats
- Betweenness centrality
- Assortativity
- Top drugs based on the number of relations
"""

#------------------ Install Dependencies -------------------#
import argparse
import pandas as pd
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'

#------------------ Helper Functions -------------------#
def build_graph(df):
    """
    Constructs a directed multigraph from the input dataframe.
    Each node is labeled with a type, and each edge represents a predicate.
    Parameters:
    df (pd.DataFrame): A dataframe with columns Subject, Predicate, Object
    Returns:
    nx.MultiDiGraph: A NetworkX directed multigraph representing the KG
    """
    G = nx.MultiDiGraph()
    for _, row in df.iterrows():
        subj, pred, obj = row['Subject'], row['Predicate'], row['Object']
        G.add_node(subj, node_type='Drug')
        G.add_node(obj, node_type=pred.replace('HAS_', '').title())
        G.add_edge(subj, obj, predicate=pred)
    return G

def basic_stats(df, G):
    """
    Prints basic statistics: total triples, node count, unique drugs and predicates.
    Parameters:
    df (pd.DataFrame): Knowledge graph dataframe
    G (nx.Graph): The corresponding graph structure
    """
    print(f"Total subject-predicate-object triples : {G.number_of_edges()}")
    print(f"Total unique nodes: {G.number_of_nodes()}")
    print(f"Unique drugs : {df['Subject'].nunique()}")
    print(f"Unique predicates (edge types): {df['Predicate'].nunique()}")

def predicate_summary(df):
    """
    Prints detailed statistics for each predicate:
    number of triples, unique object count, and average per drug.
    Parameters:
    df (pd.DataFrame): Knowledge graph dataframe
    """
    print("\nPredicate-wise statistics:")
    predicate_counts = df['Predicate'].value_counts()
    for pred, count in predicate_counts.items():
        unique_objs = df[df['Predicate'] == pred]['Object'].nunique()
        avg_per_drug = count / df['Subject'].nunique()
        print(f"{pred}: {count} triples, {unique_objs} unique objects, avg per drug : {avg_per_drug:.2f}")

def degree(G_undirected):
    """
    Computes and prints degree connectivity from the undirected graph.
    Includes average, max, and min degree.
    Parameters:
    G_undirected (nx.Graph): The undirected version of the graph
    """
    degree_sequence = [d for _, d in G_undirected.degree()]
    print(f"\nAverage degree: {sum(degree_sequence) / len(degree_sequence):.2f}")
    print(f"Max degree: {max(degree_sequence)}, Min degree: {min(degree_sequence)}")

def centrality(G_undirected):
    """
    Calculates betweenness centrality and prints top 5 nodes.
    Also reports average, min, and max centrality.
    Parameters:
    G_undirected (nx.Graph): The undirected version of the graph
    """
    print("\nBetweenness centrality (top 5 nodes):")
    centrality = nx.betweenness_centrality(G_undirected)
    top = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    for node, score in top:
        print(f"  {node}: {score:.4f}")
    print(f"Avg: {sum(centrality.values())/len(centrality):.4f}, Min: {min(centrality.values()):.4f}, Max: {max(centrality.values()):.4f}")

def assortativity(G_undirected):
    """
    Attempts to compute the degree assortativity coefficient of the graph.
    Parameters:
    G_undirected (nx.Graph): The undirected version of the graph
    """
    print("\nDegree assortativity:")
    try:
        assort = nx.degree_assortativity_coefficient(G_undirected)
        print(f"Coefficient: {assort:.4f}")
    except Exception as e:
        print("Could not compute assortativity.", e)

def top_drugs(G):
    """
    Identifies the top 5 drug nodes by total degree.
    Parameters:
    G (nx.Graph): The original directed graph
    """
    drug_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'Drug']
    drug_degrees = {n: G.degree(n) for n in drug_nodes}
    top_drugs = sorted(drug_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 drugs by degree:")
    for drug, deg in top_drugs:
        print(f"  {drug}: {deg} connections")

#------------------ Main Function -------------------#
def main(input_csv):
    """
    Main function to get graph statistics.
    Parameters:
    input_csv (str): Path to the CSV file containing the dataset.
    """
    df = pd.read_csv(input_csv)
    G = build_graph(df)
    G_undirected = G.to_undirected()

    basic_stats(df, G)
    predicate_summary(df)
    degree(G_undirected)
    centrality(G_undirected)
    assortativity(G_undirected)
    top_drugs(G)

#------------------ Argparse for Reproducibility -------------------#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute structural statistics of MEDAKA.")
    parser.add_argument("--input", required=True, help="Path to input CSV file, MEDAKA.")
    args = parser.parse_args()
    main(args.input)
