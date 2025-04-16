import json
import networkx as nx
from pathlib import Path

# === Config ===
MATCHED_FILE = "modules/matched_questions.json"
GRAPH_EXPORT_FILE = "data/graphrag_graph.gexf"  # Can be visualized with Gephi or NetworkX tools


def load_matched_data():
    with open(MATCHED_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_graph(matched_data):
    """
    Builds a directed graph of questions → matched chunks.
    Each edge has a weight (similarity score).
    """
    G = nx.DiGraph()

    for item in matched_data:
        question = item["question"]
        question_id = item["question_id"]
        question_node = f"Q{question_id}: {question}"
        G.add_node(question_node, type="question")

        for match in item["matches"]:
            chunk_id = match.get("chunk_id")
            chunk_text = match.get("text")
            score = match.get("score", 0.0)

            chunk_node = f"Chunk {chunk_id}"
            G.add_node(chunk_node, type="chunk", text=chunk_text)

            # Edge: question ➝ chunk
            G.add_edge(question_node, chunk_node, weight=1 - score)

    return G


def export_graph(graph):
    """
    Exports the graph to GEXF (for visualization in Gephi or PyVis/Streamlit).
    """
    nx.write_gexf(graph, GRAPH_EXPORT_FILE)
    print(f"📦 Graph exported to: {GRAPH_EXPORT_FILE}")


def run_graphrag_engine():
    print("🔍 Loading matched question-answer data...")
    matched = load_matched_data()

    print("🔗 Building graph...")
    G = build_graph(matched)

    print("📦 Exporting graph to GEXF format...")
    export_graph(G)

    print("✅ GraphRAG engine completed successfully.")


if __name__ == "__main__":
    run_graphrag_engine()
