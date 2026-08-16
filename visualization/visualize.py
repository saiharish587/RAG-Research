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
        """Plots average answer quality (Token F1) across configurations."""
        plt.figure(figsize=(10, 6))
        acc_col = "token_f1" if "token_f1" in df.columns else ("answer_accuracy" if "answer_accuracy" in df.columns else "cosine")
        grouped = df.groupby(["model", "rag_type"])[acc_col].mean().reset_index()
        
        sns.barplot(
            data=grouped,
            x="rag_type",
            y=acc_col,
            hue="model",
            palette="viridis"
        )
        plt.title("Answer Token F1 Quality vs. RAG Pipeline Sophistication")
        plt.xlabel("RAG Pipeline Sophistication")
        plt.ylabel(f"Average {acc_col}")
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
        """Plots average groundedness/hallucination rate across configurations."""
        plt.figure(figsize=(10, 6))
        h_col = "groundedness" if "groundedness" in df.columns else ("hallucination_rate" if "hallucination_rate" in df.columns else None)
        if not h_col or h_col not in df.columns:
            print("Skipping hallucination plot: column not present.")
            return
            
        grouped = df.groupby(["model", "rag_type"])[h_col].mean().reset_index()
        
        sns.barplot(
            data=grouped,
            x="rag_type",
            y=h_col,
            hue="model",
            palette="magma"
        )
        plt.title(f"{h_col.capitalize()} vs. RAG Pipeline Sophistication")
        plt.xlabel("RAG Pipeline Sophistication")
        plt.ylabel(f"Average {h_col.capitalize()}")
        plt.ylim(0, 1.0)
        plt.legend(title="Model", loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{h_col}_vs_rag.png"), dpi=300)
        plt.close()

    def plot_accuracy_vs_latency_scatter(self, df):
        """Plots a scatter plot showing quality vs latency trade-off."""
        plt.figure(figsize=(10, 6))
        acc_col = "token_f1" if "token_f1" in df.columns else ("answer_accuracy" if "answer_accuracy" in df.columns else "cosine")
        grouped = df.groupby(["model", "rag_type"])[[acc_col, "latency"]].mean().reset_index()
        
        sns.scatterplot(
            data=grouped,
            x="latency",
            y=acc_col,
            hue="model",
            style="rag_type",
            s=120,
            palette="Set1"
        )
        
        for idx, row in grouped.iterrows():
            plt.text(
                row["latency"] + 0.02,
                row[acc_col],
                f"{row['rag_type']}",
                fontsize=8,
                alpha=0.8
            )
            
        plt.title("RAG Cost-Benefit: Quality (Token F1) vs. Latency Trade-Off")
        plt.xlabel("Average Latency (seconds)")
        plt.ylabel(f"Average {acc_col}")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "accuracy_vs_latency_tradeoff.png"), dpi=300)
        plt.close()
