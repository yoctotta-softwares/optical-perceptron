# Optical Perceptron and Neural Networks with Angle-Tunable Weights

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18047298.svg)](https://doi.org/10.5281/zenodo.18047298)

A novel optical neural network architecture that encodes learnable weights as physical angles of frequency-selective optical elements. This enables **non-volatile weight storage** and potential for **continuous online learning** in optical hardware.

---

## Plain English Summary

**What if computers could think using light instead of electricity?**

Today's AI systems like ChatGPT run on computer chips that use electricity flowing through billions of tiny switches. This works, but it's slow (electrons move sluggishly compared to light) and uses enormous amounts of energy (those switches get hot).

This project proposes a different approach: **build AI that runs on light**.

### The Core Idea

Imagine a piece of colored glass. Tilt it one way, and red light passes through easily. Tilt it another way, and the red light gets blocked. This simple physical fact—that tilting a filter changes how much light gets through—is the foundation of our entire system.

In a normal AI, the "brain" is made of millions of numbers called "weights" that determine how the system responds to inputs. These weights are stored as electrical signals that vanish the moment you cut the power.

In our optical AI, **the weights are physical angles**. Each piece of optical glass is mounted on a tiny motor. The angle of each piece IS the weight. Turn off the power, and the angles stay exactly where they are—the AI remembers everything without using any electricity.

### Why This Matters

1. **Speed**: Light travels at 300,000 kilometers per second. Once you set up the optical system, the AI's thinking happens literally at the speed of light.

2. **Energy**: Regular AI chips burn through electricity to flip billions of switches. Our optical approach uses passive glass and filters—they just sit there letting light through. No switching, no wasted energy.

3. **Learning on the fly**: Because the weights are just angles, the AI can learn new things by simply tilting its optical components. No need to retrain everything from scratch on a supercomputer.

### What We Built

We created a computer simulation that proves this concept works. Our simulated optical system successfully learned basic logic (AND, OR, NOT operations) and even solved the classic "XOR problem" that stumped early AI researchers in the 1960s.

We also built a simple version of the same technology that powers modern AI assistants—a "transformer" architecture—where 87% of the computation happens optically.

### Shrinking It Down: No Motors Needed

The tabletop version uses motorized mounts to tilt filters, but the real potential is at the **chip scale**. Several technologies can adjust optical properties electronically—no moving parts:

- **Liquid crystals**: The same technology in your phone screen. Apply a small voltage, and the crystal molecules rotate, changing how light passes through. Already mass-produced and cheap.

- **MEMS mirrors**: Microscopic mirrors etched into silicon that tilt using tiny electrical forces. Thousands fit on a fingernail. Used in projectors and telecom switches today.

- **Electro-optic materials**: Certain crystals (like lithium niobate) change their optical properties when voltage is applied. No physical movement at all—the electrons rearrange and light behaves differently.

- **Phase-change materials**: Materials that switch between glassy and crystalline states, changing how they transmit light. Already used in rewritable DVDs.

The same principle applies: changing a voltage is equivalent to changing an angle. The weight is still stored physically (in the material's state), but now you can adjust millions of weights in microseconds, on a chip the size of your thumbnail.

**This is how optical AI could eventually fit inside your phone.**

### What's Next

The simulation works. Now someone needs to build the real thing. We're looking for collaborators with optics labs, laser equipment, and photonics expertise to take this from simulation to reality.

**This is open source and freely available for anyone to build upon.**

---

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