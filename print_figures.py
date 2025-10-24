import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import warnings

# silence specific seaborn warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def analyze_feature_distributions(X, y, feature_names):
    """
    For each feature:
      - print PD vs Control stats
      - save a boxplot + stripplot to ./feature_figures/<Feature_Name>.png
      - append a LaTeX subsection with a right-aligned wrapfigure
        into ./feature_figures/feature_sections.tex
    """

    pd_mask = y == 1
    ctrl_mask = y == 0

    # output folder next to this file
    out_dir = os.path.join(os.path.dirname(__file__), "feature_figures")
    os.makedirs(out_dir, exist_ok=True)

    tex_path = os.path.join(out_dir, "feature_sections.tex")
    with open(tex_path, "w", encoding="utf-8") as tex:
        # header comment + required packages hint
        tex.write("% Auto-generated sections. Include in your main .tex:\n")
        tex.write("% \\usepackage{graphicx,wrapfig,caption}\n")
        tex.write("% Optional spacing tweaks:\n")
        tex.write("% \\setlength{\\intextsep}{6pt}  % space above/below wrapfig\n")
        tex.write("% \\setlength{\\columnsep}{12pt} % gap between text and figure\n\n")

        print("\n--- Group Statistics (PD vs Control) ---\n")

        for j, name in enumerate(feature_names):
            if j < 12:
                pd_vals = X[pd_mask, j]
                ctrl_vals = X[ctrl_mask, j]

                mean_pd, std_pd     = np.mean(pd_vals),   np.std(pd_vals, ddof=1)
                mean_ctrl, std_ctrl = np.mean(ctrl_vals), np.std(ctrl_vals, ddof=1)
                t, p = ttest_ind(pd_vals, ctrl_vals, equal_var=False)

                print(f"{name:25s} PD: {mean_pd:.3f} ± {std_pd:.3f} | "
                      f"Ctrl: {mean_ctrl:.3f} ± {std_ctrl:.3f} | p = {p:.4e}")

                # data for seaborn
                vals = np.concatenate([ctrl_vals, pd_vals])
                groups = ["Control"] * len(ctrl_vals) + ["PD"] * len(pd_vals)
                data = {"Group": groups, "Value": vals}

                # figure
                plt.figure(figsize=(5, 4))
                sns.boxplot(x="Group", y="Value", data=data,
                            hue="Group", dodge=False, legend=False,
                            palette=["#88CCEE", "#EE8866"],
                            showfliers=False, width=0.5)
                sns.stripplot(x="Group", y="Value", data=data,
                              hue="Group", dodge=False, legend=False,
                              color="black", alpha=0.6, jitter=0.1, size=4)
                plt.title(f"{name}\n(p = {p:.3e})", fontsize=11)
                plt.ylabel(name)
                plt.grid(alpha=0.3)
                plt.tight_layout()

                # save plot
                safe_name = name.replace(" ", "_").replace("/", "_")
                save_path = os.path.join(out_dir, f"{safe_name}.png")
                plt.savefig(save_path, dpi=300)
                plt.close()

                # ... inside your loop, replace your LaTeX-emitting block with:

                wrap_lines = 16  # how many lines the figure should occupy on the right
                rel_path = f"feature_figures/{safe_name}.png"

                tex.write(f"\\subsection{{{name}}}\n")
                tex.write("\\noindent% start paragraph flush-left so wrap attaches here\n")
                tex.write(f"\\begin{{wrapfigure}}[{wrap_lines}]{{r}}{{0.36\\textwidth}}\n")
                tex.write("  \\vspace{-0.8\\baselineskip}% nudge up toward subsection line\n")
                tex.write("  \\centering\n")
                tex.write(f"  \\includegraphics[width=\\linewidth]{{{rel_path}}}\n")
                tex.write(f"  \\caption{{{name} (PD vs Control)}}\n")
                tex.write("\\end{wrapfigure}\n")
                tex.write("% Your explanatory paragraph goes here; it will wrap for ~wrap_lines lines.\n")
                tex.write(
                    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed fringilla quis velit at lacinia. Ut dictum tellus sem, in viverra ante vestibulum id. Mauris quam sapien, dapibus sit amet elit vitae, maximus commodo mi. Suspendisse a neque massa. Quisque nibh quam, luctus id nisl vitae, faucibus sagittis mi. Etiam rutrum id neque quis laoreet. Aliquam facilisis lacus ac bibendum sagittis.Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed fringilla quis velit at lacinia. Ut dictum tellus sem, in viverra ante vestibulum id. Mauris quam sapien, dapibus sit amet elit vitae, maximus commodo mi. Suspendisse a neque massa. Quisque nibh quam, luctus id nisl vitae, faucibus sagittis mi. Etiam rutrum id neque quis laoreet. Aliquam facilisis lacus ac bibendum sagittis.Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed fringilla quis velit at lacinia. Ut dictum tellus sem, in viverra ante vestibulum id. Mauris quam sapien, dapibus sit amet elit vitae, maximus commodo mi. Suspendisse a neque massa. Quisque nibh quam, luctus id nisl vitae, faucibus sagittis mi. Etiam rutrum id neque quis laoreet. Aliquam Quisque nibh quam, luctus id nisl vitae, faucibus sagittis mi. Etiam rutrum id neque quis laoreet. Aliquam facilisis lacus ac bibendum sagittis.Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed fringilla quis velit at lacinia. Ut dictum tellus sem, in viverra ante vestibulum id. Mauris quam sapien, dapibus sit amet elit vitae, maximus commodo mi. Suspendisse a neque massa. Quisque nibh quam, luctus id nisl vitae, faucibus sagittis mi. Etiam rutrum id neque quis laoreet. Aliquam .\n\n")

                print(f"→ Saved: {save_path}")

    print(f"\n✅ Figures saved to: {out_dir}")
    print(f"✅ LaTeX sections written to: {tex_path}\n")
