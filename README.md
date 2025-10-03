# DeltaNet: A Lagrangian Perspective

This repository contains a concise research-style implementation and explanation of **DeltaNet**, an efficient linear attention variant with adaptive forgetting. We derive DeltaNet from a quadratic Lagrangian formulation and provide a PyTorch reference implementation and demo notebook.

---

## Overview
- **Vanilla Linear Attention** can be derived from a quadratic Lagrangian with an identity penalty, leading to additive accumulation of memory terms.
- **DeltaNet** introduces an adaptive forgetting factor, projecting away redundant directions and retaining a compressed, more efficient memory representation.
- We show that DeltaNet naturally arises when the quadratic penalty matrix is replaced by a dynamic form that enforces "don’t overuse already-represented directions."

---

## Key Equations
- **Vanilla:**  
  $S_t = S_{t-1} + v_t k_t^T$

- **DeltaNet:**  
  $S_t = S_{t-1} - \beta_t S_{t-1} k_t k_t^T + \beta_t v_t k_t^T$
- Coefficients emerge as survival factors: earlier memories decay if later keys align with them.

---

## Repository Contents
- `deltanet.py` → PyTorch implementation of DeltaNet update.
- `notebooks/demo.ipynb` → Colab-ready demo showing Vanilla vs DeltaNet on toy data.
- `report.pdf` → Two-page technical note (Lagrangian derivation + intuition).
- `figures/figure1.png` → Schematic diagram of Vanilla vs DeltaNet memory dynamics.

---

## Getting Started
```bash
# Clone repo
git clone https://github.com/Vishal-sys-code/deltanet-lagrangian
cd deltanet-lagrangian

# Install dependencies
pip install torch matplotlib jupyter

# Run demo
jupyter notebook notebooks/demo.ipynb
```

---

## Citation
If you use this work, please cite:
```
@misc{pandey2025deltanet,
  title={A Lagrangian Interpretation of DeltaNet's Update Mechanism},
  author={Pandey, Vishal},
  year={2025},
  note={Technical Report}
}
```

---

## License
MIT License. See `LICENSE` for details.
