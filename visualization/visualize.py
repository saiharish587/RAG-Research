import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def __init__(self, output_dir="results/graphs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.titlesize': 16
        })

    def generate_all_plots(self, csv_path):
        """Loads results CSV and generates comparative analysis plots."""
        if not os.path.exists(csv_path):
            print(f"Visualization error: Results file '{csv_path}' does not exist.")
            return
            
        df = pd.read_csv(csv_path)
        if df.empty:
            print("Visualization error: Loaded CSV is empty.")
            return

        print("Generating benchmark plots...")
        self.plot_accuracy_vs_rag(df)
        self.plot_latency_vs_rag(df)
        self.plot_accuracy_vs_latency_scatter(df)
        self.plot_hallucination_rate(df)
        print(f"All plots saved to {self.output_dir}")

    def plot_accuracy_vs_rag(self, df):
        """Plots average answer accuracy across configurations."""
        plt.figure(figsize=(10, 6))
        # Group by Model and RAG configuration
        grouped = df.groupby(["model", "rag_type"])["answer_accuracy"].mean().reset_index()
        
        sns.barplot(
            data=grouped,
            x="rag_type",
            y="answer_accuracy",
            hue="model",
            palette="viridis"
        )
        plt.title("Answer Accuracy vs. RAG Pipeline Sophistication")
        plt.xlabel("RAG Pipeline Sophistication")
        plt.ylabel("Average Answer Accuracy (Semantic Similarity)")
        plt.ylim(0, 1.0)
        plt.legend(title="Model", loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "accuracy_vs_rag.png"), dpi=300)
        plt.close()

    def plot_latency_vs_rag(self, df):
        """Plots average latency across configurations."""
        plt.figure(figsize=(10, 6))
        grouped = df.groupby(["model", "rag_type"])["latency"].mean().reset_index()
        
        sns.barplot(
            data=grouped,
            x="rag_type",
            y="latency",
            hue="model",
            palette="coolwarm"
        )
        plt.title("Inference Latency vs. RAG Pipeline Sophistication")
        plt.xlabel("RAG Pipeline Sophistication")
        plt.ylabel("Average Latency (seconds)")
        plt.legend(title="Model", loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "latency_vs_rag.png"), dpi=300)
        plt.close()

    def plot_hallucination_rate(self, df):
        """Plots average hallucination rate across configurations."""
        plt.figure(figsize=(10, 6))
        grouped = df.groupby(["model", "rag_type"])["hallucination_rate"].mean().reset_index()
        
        sns.barplot(
            data=grouped,
            x="rag_type",
            y="hallucination_rate",
            hue="model",
            palette="magma"
        )
        plt.title("Hallucination Rate vs. RAG Pipeline Sophistication")
        plt.xlabel("RAG Pipeline Sophistication")
        plt.ylabel("Average Hallucination Rate")
        plt.ylim(0, 1.0)
        plt.legend(title="Model", loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "hallucination_vs_rag.png"), dpi=300)
        plt.close()

    def plot_accuracy_vs_latency_scatter(self, df):
        """Plots a scatter plot showing accuracy vs latency trade-off."""
        plt.figure(figsize=(10, 6))
        grouped = df.groupby(["model", "rag_type"])[["answer_accuracy", "latency"]].mean().reset_index()
        
        # Define marker style per RAG type
        markers = {"no_rag": "o", "naive": "s", "advanced": "D", "modular": "^"}
        
        sns.scatterplot(
            data=grouped,
            x="latency",
            y="answer_accuracy",
            hue="model",
            style="rag_type",
            markers=markers,
            s=120,
            palette="Set1"
        )
        
        # Label points
        for idx, row in grouped.iterrows():
            plt.text(
                row["latency"] + 0.1,
                row["answer_accuracy"] - 0.01,
                f"{row['rag_type']}",
                fontsize=8,
                alpha=0.8
            )
            
        plt.title("RAG Cost-Benefit: Accuracy vs. Latency Trade-Off")
        plt.xlabel("Average Latency (seconds) -> lower is better")
        plt.ylabel("Average Answer Accuracy -> higher is better")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "accuracy_vs_latency_tradeoff.png"), dpi=300)
        plt.close()
