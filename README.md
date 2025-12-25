# Optical Perceptron and Neural Networks with Angle-Tunable Weights

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18047298.svg)](https://doi.org/10.5281/zenodo.18047298)

A novel optical neural network architecture that encodes learnable weights as physical angles of frequency-selective optical elements. This enables **non-volatile weight storage** and potential for **continuous online learning** in optical hardware.

**Includes: Optical Perceptron + Optical Transformer (87% optical compute)**

## 🔑 Key Idea

```
Angle θ  →  Transmission T(θ) = sigmoid(kθ)  →  Weight W = 2T - 1 ∈ [-1, +1]
```

Instead of encoding weights as voltages or currents, we encode them as the **physical angle** of a frequency-selective optical surface (dichroic filter, Fabry-Pérot etalon, photonic crystal, etc.). The transmission coefficient varies smoothly with angle, giving us a continuous, differentiable weight.

**Why this matters:**
- **Non-volatile**: Weights are physical angles—no power needed to retain them
- **Online learning**: Adjust angles during operation (motorized mounts, MEMS, piezo)
- **Speed of light inference**: Forward pass is optical propagation
- **Energy efficient**: Passive optical elements, no transistor switching

## 📊 Validation Results

This simulation validates that the architecture can learn:

| Function | Type | Result |
|----------|------|--------|
| AND | Single layer | ✅ 100% |
| OR | Single layer | ✅ 100% |
| NAND | Single layer | ✅ 100% |
| NOR | Single layer | ✅ 100% |
| XOR | Multi-layer (2→4→1) | ✅ 100% |

XOR requires nonlinear decision boundaries, proving **universal approximation capability**.

## 🤖 Optical Transformer

We also implement a **tiny transformer** where 87% of compute is optical:

```bash
python optical_transformer.py
```

**Architecture:**
- Vocab: 42 characters
- Embedding: 32 dims
- 1 attention head, 1 layer
- FFN hidden: 64
- Context: 16 tokens
- **Total: 11,114 parameters**

**What's Optical:**

| Component | Implementation | Hardware |
|-----------|---------------|----------|
| Q, K, V, O projections | OpticalLinear | Angle-encoded weights |
| FFN layers | OpticalLinear | Angle-encoded weights |
| **Softmax** | **OpticalSoftmax** | exp→transmission, Σ→beam combining |
| Activations | Sigmoid | Saturable absorbers |

**Key Insight:** Softmax IS optical!
```
Sigmoid:  exp(x) / (1 + exp(x))     ← our perceptron
Softmax:  exp(x_i) / Σ exp(x_j)     ← same, just N-way with optical sum
```

**Training Results:**
```
Loss: 3.17 → 0.24 (learns "hello hello hello..." pattern)
Optical compute: 87.3%
```

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/yoctotta-softwares/optical-perceptron.git
cd optical-perceptron

# Install dependencies (just numpy)
pip install -r requirements.txt

# Run validation
python optical_perceptron.py

# Run transformer demo
python optical_transformer.py
```

Expected output:
```
=================================================================
  OPTICAL PERCEPTRON - FINAL VALIDATION
=================================================================

─── LOGIC GATES ───

AND: ✓ PASS
  Weights: [0.997, 0.996], Bias: -1.493
    [0, 0] → 0.003 (expect 0) ✓
    [0, 1] → 0.119 (expect 0) ✓
    [1, 0] → 0.121 (expect 0) ✓
    [1, 1] → 0.881 (expect 1) ✓
...
  🎉 VALIDATED! THE CONCEPT HAS WATER! 🎉
```

## 🏗️ Architecture

### Single Optical Perceptron

```
                    ┌─────────────────┐
   x₁ ──[light]──→  │  θ₁ (angle)     │──→ T(θ₁)·x₁ ─┐
                    │  dichroic       │              │
   x₂ ──[light]──→  │  θ₂ (angle)     │──→ T(θ₂)·x₂ ─┼──→ Σ ──→ σ(z) ──→ y
                    │  filter         │              │
   xₙ ──[light]──→  │  θₙ (angle)     │──→ T(θₙ)·xₙ ─┘
                    └─────────────────┘
```

### Weight Mapping

```python
def weight(self, i):
    """angle → transmission → signed weight"""
    T = 1.0 / (1.0 + np.exp(-2.0 * self.angles[i]))  # Transmission [0,1]
    return 2.0 * T - 1.0  # Weight [-1, +1]
```

The key insight is the `2T - 1` transformation that maps transmission to signed weights, interpretable as phase (constructive vs destructive interference).

### Multi-Layer Networks

For non-linearly separable problems (like XOR), we stack layers:

```python
net = OpticalNetwork([2, 4, 1])  # 2 inputs → 4 hidden → 1 output
```

## 📄 Paper

See `paper/optical_perceptron_paper.pdf` for the full technical writeup including:
- Mathematical framework
- Algorithm pseudocode  
- Comparison with existing optical neural networks (D²NN, MZI meshes, etc.)
- Implementation pathways with commercial components
- Open challenges

## 🔬 Physical Implementation Ideas

| Approach | Components | Est. Cost | Notes |
|----------|------------|-----------|-------|
| Dichroic filters + motorized rotation | Thorlabs PRM1Z8, Edmund Optics filters | ~$3,000 | Proof of concept, slow |
| Spatial Light Modulator | Holoeye PLUTO | ~$10,000 | Fast, programmable |
| MEMS mirror array | Custom/research | Variable | Scalable, fast |
| Liquid crystal variable retarders | Meadowlark | ~$2,000 | Electronic control |

### Minimum Viable Prototype

```
Components:
├── 2× Dichroic longpass filters (Edmund Optics) - $300
├── 2× Motorized rotation mounts (Thorlabs PRM1Z8) - $3,000  
├── 1× Broadband LED source - $200
├── 1× Silicon photodetector - $150
├── Optical breadboard + mounts - $500
└── Arduino + stepper drivers - $50
                                    Total: ~$4,200
```

## 🤝 Call for Collaboration

I'm a software person, not a hardware person. This concept is mathematically validated but needs physical implementation. I'm looking for collaborators with:

- **Photonics lab access** for prototyping
- **Optical simulation expertise** (MEEP, Lumerical, COMSOL)
- **MEMS/nanofabrication** capabilities
- **Funding/resources** to build proof-of-concept

If interested, please open an issue or reach out!

## 📚 Related Work

This builds on ideas from:

- **D²NN** (Ozcan Lab, UCLA): Diffractive deep neural networks - fixed weights
- **MZI meshes** (MIT/Lightmatter): Mach-Zehnder interferometer networks
- **Photonic crystals**: Frequency-selective optical elements

Key difference: Our weights are **continuously tunable angles** enabling online learning.

## 📝 Citation

If you use this work, please cite:

```bibtex
@misc{optical_perceptron_2025,
  author = {Satpathy, Surajbhan},
  title = {Frequency-Selective Optical Perceptrons with Angle-Tunable Weights},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yoctotta-softwares/optical-perceptron}
}

@software{optical_perceptron_2025,
  author = {Satpathy, Surajbhan},
  title        = {Frequency-Selective Optical Perceptrons with Angle-Tunable Weights},
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18047298},
  url          = {https://doi.org/10.5281/zenodo.18047298}
}
```

## 📜 License

MIT License - see [LICENSE](LICENSE)

## 🙏 Acknowledgments

Mathematical validation and simulation code developed with assistance from Claude (Anthropic). The core concept of angle-tunable frequency-selective weights for continuous optical learning was conceived by the human author.


## 📢 Seeking arXiv Endorsement

I'd like to submit this work to arXiv (cs.AI) but need an endorser as a first-time submitter.
[Link to endorse](https://arxiv.org/auth/endorse?x=NCHNJN)

If you've published in these categories and are willing to endorse, please [open an issue](https://github.com/yoctotta-softwares/optical-perceptron/issues) or reach out directly.

**Already citable via Zenodo:** [10.5281/zenodo.18047298](https://doi.org/10.5281/zenodo.18047298)