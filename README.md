# information-bottleneck-nested-optimizers

**information-bottleneck-nested-optimizers** is an empirical research project on optimizer dynamics through the lens of the **Information Bottleneck (IB)** framework. The repository studies whether deeply nested optimization schemes—especially **Deep Momentum Gradient Descent (DMGD)**—produce the representation-compression behavior predicted by information-theoretic accounts of learning.

The project combines experiment code, logged outputs, figures, and the final write-up in a single repository. It is intended to be read as a compact research artifact rather than only a code dump.

| Repository focus | Description |
|---|---|
| Research domain | Optimization dynamics, representation learning, and information-theoretic analysis |
| Central question | Do nested optimizers induce a distinct compression phase relative to standard baselines? |
| Main experimental setting | Bottlenecked neural architectures trained on MNIST |
| Key comparison | DMGD versus optimizers such as AdamW and gradient descent with momentum |
| Portfolio value | Shows experimental ML research, careful comparative analysis, and technical reporting |

## Main finding

The central empirical result is that **DMGD exhibits a sustained deep-layer compression phase** in the bottlenecked architecture, whereas the comparison optimizers do not show the same qualitative behavior. In the IB framing, this matters because compression is interpreted as the optimizer learning to retain task-relevant structure while discarding descriptive noise.

| Optimizer | Deep-layer \(I(T;X)\) trajectory | Interpretation |
|---|---|---|
| AdamW / GDM | Rises and plateaus at a relatively high information level | Retains more descriptive noise and does not show sustained compression |
| DMGD | Rises, then enters a persistent compression phase after later epochs | Supports the nested-memory interpretation and aligns with the IB-style prediction |

The figure used in the report to highlight this contrast is available at [paper/figures/figure1.png](paper/figures/figure1.png), and the final write-up is available at [paper/nested_learning.pdf](paper/nested_learning.pdf).

## Why this repository matters

The project sits at the intersection of optimization theory and empirical representation analysis. Rather than comparing optimizers only by accuracy or loss, it studies how they shape internal representations over training time. That makes the repository useful for readers interested in learning dynamics, not just endpoint performance.

From a portfolio perspective, the value lies in the full loop: formulating a theoretical question, implementing comparative experiments, logging information-theoretic quantities, and packaging the results in a concise research report.

| Capability area | How it appears here |
|---|---|
| Experimental ML research | Controlled comparison across multiple optimizers and epoch budgets |
| Information-theoretic analysis | Use of mutual-information-style quantities to interpret learned representations |
| Reproducibility | Code, figures, and paper materials are kept together in one repository |
| Research communication | The final report and supporting figures make the result easy to inspect |

## Running the experiments

Install the project dependencies and launch the training-and-logging script.

```bash
git clone https://github.com/davidcagoh/information-bottleneck-nested-optimizers
cd information-bottleneck-nested-optimizers
pip install -r code/requirements.txt
python code/train_ib_analysis.py
```

The script runs the MLP experiments across the configured optimizer set and epoch schedule, producing the mutual-information traces used for the report figures and comparative analysis.

## Repository layout

| Path | Role |
|---|---|
| `code/` | Training and analysis scripts for the optimizer experiments |
| `paper/` | Final report and publication-style figures |
| Logged outputs | Intermediate results supporting the reported comparisons |

## Reading this repository as a portfolio piece

If you are arriving from the GitHub profile, the intended takeaway is that this repository demonstrates **research-oriented machine learning engineering**. It is not a benchmark leaderboard project and not a polished library package. Instead, it shows the ability to turn a theoretical claim about nested optimization into an executable experiment with interpretable outcomes.

In that sense, **information-bottleneck-nested-optimizers** showcases comparative experiment design, representation-level analysis, and clear technical communication in a compact format.

## Related work

This repository builds on prior work in nested learning dynamics and information-theoretic analysis of neural representations.

1. **Behrouz, A., Razaviyayn, M., Zhong, P., & Mirrokni, V.** (2025). *Nested Learning: The Illusion of Deep Learning Architectures.* arXiv:2512.24695.
2. **Cheng, P., Hao, W., Dai, S., Liu, J., Gan, Z., & Carin, L.** (2020). *CLUB: A Contrastive Log-ratio Upper Bound of Mutual Information.* arXiv:2006.12013.
3. **Lyu, Z., Aminian, G., & Rodrigues, M. R. D.** (2023). *On Neural Networks Fitting, Compression, and Generalization Behavior via Information-Bottleneck-like Approaches.* *Entropy*, 25(7), 1063.

## License

This project is released under the MIT License. See `LICENSE` for details.
