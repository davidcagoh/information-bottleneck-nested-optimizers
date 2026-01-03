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

## 📚 Related Work

This project builds on prior work in nested learning dynamics and information-theoretic analysis of neural representations.

1. **Behrouz, A., Razaviyayn, M., Zhong, P., & Mirrokni, V.** (2025).  
   *Nested Learning: The Illusion of Deep Learning Architectures.*  
   arXiv:2512.24695. https://doi.org/10.48550/arXiv.2512.24695

2. **Cheng, P.** (2025).  
   *Linear95/CLUB* (Jupyter Notebook implementation).  
   https://github.com/Linear95/CLUB  
   (Original work published 2020)

3. **Cheng, P., Hao, W., Dai, S., Liu, J., Gan, Z., & Carin, L.** (2020).  
   *CLUB: A Contrastive Log-ratio Upper Bound of Mutual Information.*  
   arXiv:2006.12013. https://doi.org/10.48550/arXiv.2006.12013

4. **Lyu, Z., Aminian, G., & Rodrigues, M. R. D.** (2023).  
   *On Neural Networks Fitting, Compression, and Generalization Behavior via Information-Bottleneck-like Approaches.*  
   *Entropy*, 25(7), 1063. https://doi.org/10.3390/e25071063

5. **McCleary, K.** (2026).  
   *kmccleary3301/nested_learning* (Python implementation).  
   https://github.com/kmccleary3301/nested_learning  
   (Original work published 2025)

## Licensing

This project is released under the MIT License. See the `LICENSE` file for details.