import json
import os
import argparse
from typing import List, Dict, Any
from pathlib import Path

def generate_markdown_report(results: dict, output_path: str):
    print(f"[AUDIT] Generating Markdown Report: {output_path}")
    
    with open(output_path, 'w') as f:
        f.write("# CondorNet v4.3 Interpretability Audit\n\n")
        f.write("## 1. Feature Importance (Permutation)\n\n")
        
        imp = results.get("permutation_importance", {})
        for head, feats in imp.items():
            f.write(f"### Head: {head}\n")
            for feat in feats:
                f.write(f"- **{feat['feature']}**: {feat['importance']:.4f}\n")
            f.write("\n")
            
        f.write("## 2. Gradient Saliency\n\n")
        sal = results.get("gradient_saliency", {})
        for head, feats in sal.items():
            f.write(f"### Head: {head}\n")
            for feat in feats:
                f.write(f"- **{feat['feature']}**: {feat['saliency']:.4f}\n")
            f.write("\n")
            
        f.write("## 3. Mutual Information\n\n")
        mi = results.get("mutual_information", {})
        for head, feats in mi.items():
            f.write(f"### Head: {head}\n")
            for feat in feats:
                f.write(f"- **{feat['feature']}**: {feat['mi_score']:.4f}\n")
            f.write("\n")
            
        f.write("## 4. Surrogate Decision Trees\n\n")
        trees = results.get("surrogate_trees", {})
        for head, info in trees.items():
            f.write(f"### Head: {head} (Fidelity R2: {info['fidelity_r2']:.4f})\n")
            f.write("```text\n")
            f.write(info['rules'])
            f.write("\n```\n\n")
            
        f.write("## 5. Output Statistics\n\n")
        stats = results.get("output_statistics", {})
        for head, s in stats.items():
            f.write(f"### {head}\n")
            f.write(f"- Mean: {s['mean']:.4f}\n")
            f.write(f"- Std: {s['std']:.4f}\n")
            f.write(f"- Skew: {s['skew']:.4f}\n")
            f.write(f"- Kurtosis: {s['kurtosis']:.4f}\n")
            f.write(f"- Min/Max: [{s['min']:.4f}, {s['max']:.4f}]\n\n")
            
        f.write("## 6. Advanced Diagnostics\n\n")
        
        fisher = results.get("fisher_information", {})
        f.write("### Fisher Information (Top Sensitive Layers)\n")
        for layer in fisher.get("top_sensitive_layers", []):
            f.write(f"- **{layer['layer']}**: {layer['fisher_trace_mean']:.6f}\n")
        f.write("\n")
        
        hessian = results.get("hessian_eigenspectrum", {})
        f.write("### Hessian Eigenspectrum Estimation\n")
        for k, v in hessian.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n")
        
        boot = results.get("bootstrap_stability", {})
        f.write("### Bootstrap Stability\n")
        for head, s in boot.items():
            f.write(f"- **{head}**: Mean={s['mean_of_means']:.4f}, Std={s['std_of_means']:.4f}\n")
        f.write("\n")
        
        f.write("## 7. Pros & Cons Evaluation\n\n")
        pc = results.get("pros_cons", {})
        f.write("### Pros\n")
        for p in pc.get("pros", []):
            f.write(f"- ✅ {p}\n")
        f.write("\n### Cons\n")
        for c in pc.get("cons", []):
            f.write(f"- ⚠️ {c}\n")
        f.write("\n")

def __main__hook(parser):
    """Extend the existing arguments with data capabilities"""
    parser.add_argument("--data", type=str, default="",
                        help="Path to preprocessed features CSV for full analytics audit")
    parser.add_argument("--samples", type=int, default=3000,
                        help="Number of continuous samples to load for analytics")
    parser.add_argument("--seq-len", type=int, default=10,
                        help="Sequence length for recurrent processing")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Directory to save full audit reports and visualizations")
                        
    return parser
