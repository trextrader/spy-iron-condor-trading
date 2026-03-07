import os
import re
from pathlib import Path

base_path = "c:/SPYOptionTrader_test/intelligence/audit_condornet_interpretability_v43.py"
part2_path = "c:/SPYOptionTrader_test/intelligence/audit_part2.py"
part3_path = "c:/SPYOptionTrader_test/intelligence/audit_part3.py"
part4_path = "c:/SPYOptionTrader_test/intelligence/audit_part4.py"

with open(base_path, 'r', encoding='utf-8') as f:
    base = f.read()
with open(part2_path, 'r', encoding='utf-8') as f:
    part2 = f.read()
with open(part3_path, 'r', encoding='utf-8') as f:
    part3 = f.read()
with open(part4_path, 'r', encoding='utf-8') as f:
    part4 = f.read()

parts = "\n\n# --- DATA ANALYTICS MODULES ---\n\n" + part2 + "\n" + part3 + "\n" + part4 + "\n"

# Add __main__hook to parser
base = base.replace(
    'args = parser.parse_args()',
    'parser = __main__hook(parser)\n    args = parser.parse_args()'
)

data_run_code = """
    # 7. Data-based Analytics
    if args.data:
        print("\\n[AUDIT] Starting Data-based Analytics...")
        from intelligence.schema_v43 import STRATEGY_TYPES, PIVOT_HORIZONS
        
        output_dir = args.output_dir or str(Path(args.output_json).parent / "v43_audit")
        os.makedirs(output_dir, exist_ok=True)
        
        # Load sequence data once
        X = load_audit_data(args.data, _FULL_FEATURE_NAMES, args.samples, args.seq_len)
        runner = CondorRunner(model, DEVICE, STRATEGY_TYPES, PIVOT_HORIZONS)
        
        # Run base permutations and trees
        data_results = {}
        data_results["output_statistics"] = analyze_output_statistics(runner, X)
        data_results["permutation_importance"] = analyze_permutation_importance(runner, X, _FULL_FEATURE_NAMES)
        data_results["gradient_saliency"] = analyze_gradient_saliency(runner, X, _FULL_FEATURE_NAMES)
        data_results["mutual_information"] = analyze_mutual_information(runner, X, _FULL_FEATURE_NAMES)
        data_results["surrogate_trees"] = analyze_surrogate_trees(runner, X, _FULL_FEATURE_NAMES)
        
        # Run extended analytics
        data_results = run_extended_analytics(
            runner=runner,
            model=model,
            X=X,
            feature_names=_FULL_FEATURE_NAMES,
            data_results=data_results
        )
        
        logic["data_analytics"] = data_results
        
        # Visuals and Markdown
        plot_comprehensive_visualizations(data_results, output_dir)
        
        md_path = os.path.join(output_dir, "audit_report.md")
        generate_markdown_report(data_results, md_path)
        
        # Re-save logic with data analytics included
        with open(args.output_json, 'w') as f:
            json.dump(logic, f, indent=2)
"""

base = base.replace('    print("\\n[AUDIT] Done.")', data_run_code + '\n    print("\\n[AUDIT] Done.")')

# Insert parts before main
main_idx = base.find("def main():")
final_code = base[:main_idx] + parts + base[main_idx:]

with open(base_path, 'w', encoding='utf-8') as f:
    f.write(final_code)
