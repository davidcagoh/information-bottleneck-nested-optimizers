# Deep Learning Nested Optimization: Information Bottleneck Analysis of DMGD

This repository contains the source code, data logs, and the final academic report for an empirical investigation into the dynamics of deeply nested optimizers (specifically Deep Momentum Gradient Descent, DMGD) using the Information Bottleneck (IB) framework.

The project validates the theory that optimizers can be interpreted as multi-timescale associative memory systems, demonstrating that **DMGD uniquely induces the representation compression phase** predicted by IB theory.

## 📄 Final Report

The complete two-page report, including the paper summary and the novel empirical investigation, is available here:
- **[Download Final PDF Report](paper/nested_learning.pdf)**

## 📊 Key Finding: The Nested Compression Phase

The central finding is the qualitative difference in the deep-layer compression trajectory observed in the bottlenecked architecture ($784 \to 4$).

| Optimizer | Deep Layer $I(T;X)$ Trajectory | Interpretation |
| :--- | :--- | :--- |
| **AdamW / GDM** | Rises asymptotically, plateaus above 10 bits. | Retains descriptive noise; lacks sustained compression. |
| **DMGD** | Rises, then enters a **sustained decrease** (compression phase) post-Epoch 150, approaching 0 bits. | Nested memory successfully filters noise, achieving near-perfect compression. |

The figure used in the report, highlighting this comparison, is available in the paper directory: [Figure 1: IB Trajectory Comparison](paper/figures/figure1.png)

## 🚀 Reproduction Instructions

### Prerequisites

1.  Clone this repository:
    ```bash
    git clone https://github.com/davidcagoh/information-bottleneck-nested-optimizers
    cd information-bottleneck-nested-optimizers
    ```
2.  Install all required Python dependencies:
    ```bash
    pip install -r code/requirements.txt
    ```

### Running the Experiment

To train the MLP on MNIST using the four optimizers and log the mutual information data:

```bash
python code/train_ib_analysis.py 
```

## Licensing

This project is released under the MIT License. See the `LICENSE` file for details.