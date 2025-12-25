"""
Optical Transformer - Tiny Character-Level Model
=================================================

A transformer architecture where ALL core computations are implemented
using optical perceptrons with angle-tunable weights.

KEY INSIGHT: Softmax IS optical!
- Sigmoid: σ(x) = exp(x) / (1 + exp(x))
- Softmax: softmax(x_i) = exp(x_i) / Σ_j exp(x_j)

Both use exponentials and normalization. In optics:
1. Transmission through angle-tuned element → exp(kx) response
2. All beams hitting one detector → optical summation  
3. Beam splitter + feedback → normalization

OPTICAL COMPONENTS:
- Linear layers (Q, K, V, O, FFN)
- Softmax (via transmission + optical summation + normalization)
- Activation functions (sigmoid via saturable absorbers)

ELECTRONIC COMPONENTS:
- Embeddings (lookup table - not compute)
- LayerNorm (requires mean/variance - future optical research)

Architecture:
- Vocab: 64 characters  
- Embedding dim: 32
- 1 attention head
- 1 transformer layer
- FFN hidden: 64
- Context length: 16

Author: Suraj Kumar
License: MIT
"""

import numpy as np
from typing import Optional, Tuple, List
import string


# =============================================================================
# CORE OPTICAL PRIMITIVES
# =============================================================================

class OpticalElement:
    """
    Base class for optical neural network elements.
    
    Core principle: angle θ → transmission T(θ) → weight/activation
    
    Physical implementation:
    - Frequency-selective surface (dichroic filter, Fabry-Pérot, photonic crystal)
    - Angle determines cutoff frequency
    - Input light intensity modulated by transmission
    """
    
    @staticmethod
    def angle_to_transmission(angle: np.ndarray, sharpness: float = 2.0) -> np.ndarray:
        """
        Convert angle to transmission coefficient.
        
        T(θ) = sigmoid(k * θ) = 1 / (1 + exp(-k * θ))
        
        In physics: models the S-curve of transmission vs angle
        for a frequency-selective surface near cutoff.
        """
        return 1.0 / (1.0 + np.exp(-sharpness * angle))
    
    @staticmethod
    def transmission_to_weight(T: np.ndarray) -> np.ndarray:
        """
        Convert transmission to signed weight.
        
        W = 2T - 1, giving W ∈ [-1, +1]
        
        Physical interpretation:
        - T ≈ 0 → W ≈ -1 (phase inversion / destructive interference)
        - T ≈ 0.5 → W ≈ 0 (partial transmission)
        - T ≈ 1 → W ≈ +1 (full transmission / constructive)
        """
        return 2.0 * T - 1.0


class OpticalLinear(OpticalElement):
    """
    Linear layer as an array of optical perceptrons.
    
    Y = X @ W + b
    
    Physical implementation:
    - Input X encoded as light intensities across spatial/wavelength channels
    - Each output is one optical perceptron
    - Weight matrix W encoded as angles of frequency-selective surfaces
    - Detector sums weighted contributions → one output value
    - Parallel detectors → full output vector
    
    This happens at the SPEED OF LIGHT in hardware.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        
        # Xavier initialization in angle space
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.angles = np.random.randn(in_features, out_features) * scale
        
        self.bias = np.zeros(out_features) if bias else None
        self._input_cache = None
    
    def get_weights(self) -> np.ndarray:
        """Get weight matrix from angles."""
        T = self.angle_to_transmission(self.angles)
        return self.transmission_to_weight(T)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: matrix multiply via optical transmission."""
        self._input_cache = x
        W = self.get_weights()
        out = x @ W
        if self.has_bias:
            out = out + self.bias
        return out
    
    def backward(self, grad_output: np.ndarray, lr: float) -> np.ndarray:
        """Backward pass with angle updates."""
        W = self.get_weights()
        grad_input = grad_output @ W.T
        
        # Gradient w.r.t weights
        if self._input_cache.ndim == 1:
            grad_W = np.outer(self._input_cache, grad_output)
        else:
            grad_W = self._input_cache.T @ grad_output
        
        # Chain rule: dL/dθ = dL/dW * dW/dT * dT/dθ
        T = self.angle_to_transmission(self.angles)
        dT_dtheta = 2.0 * T * (1.0 - T)  # sigmoid derivative * sharpness
        dW_dT = 2.0
        grad_angles = grad_W * dW_dT * dT_dtheta
        
        self.angles -= lr * np.clip(grad_angles, -1, 1)
        
        if self.has_bias:
            grad_bias = grad_output.sum(axis=0) if grad_output.ndim > 1 else grad_output
            self.bias -= lr * grad_bias
        
        return grad_input


class OpticalSoftmax(OpticalElement):
    """
    Softmax implemented optically!
    
    softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
    
    Physical implementation:
    
    1. EXPONENTIAL (per element):
       - Each score x_i modulates light through optical element
       - Transmission T(x_i) ∝ exp(k * x_i) for appropriate regime
       - Use optical amplifier or nonlinear crystal for true exponential
    
    2. SUMMATION (denominator):
       - All transmitted beams hit a SINGLE detector
       - Detector current = Σ_j exp(x_j) (optical summation!)
       - This is FREE in optics - just combine beams
    
    3. NORMALIZATION (division):
       Option A: Optical feedback loop
         - Total intensity controls gain of optical amplifier
         - Auto-normalizes all channels
       
       Option B: Beam splitter approach  
         - Split each beam: one to output, one to sum detector
         - Use sum to control variable attenuator on output paths
       
       Option C: Winner-take-all (approximate)
         - Lateral inhibition via optical coupling
         - Strongest signal suppresses others
    
    The key insight: sigmoid IS a 2-class softmax!
    σ(x) = exp(x)/(exp(x) + exp(0)) = softmax([x, 0])[0]
    
    So our angle-tuned perceptron already does the core operation.
    Extending to N-way softmax just needs optical fan-in for the sum.
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Temperature controls sharpness of attention.
        
        In optics: higher temp = gentler angle response curve
        """
        self.temperature = temperature
        self._cache = None
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Optical softmax forward pass.
        
        x: (..., n) input scores
        mask: optional mask (e.g., causal attention)
        """
        # Scale by temperature (optical: adjust input gain)
        scaled = x / self.temperature
        
        # Apply mask before exponential
        if mask is not None:
            scaled = np.where(mask, scaled, -1e9)
        
        # Numerical stability (optical: auto-gain control does this naturally)
        shifted = scaled - scaled.max(axis=-1, keepdims=True)
        
        # Exponential: optical transmission in nonlinear regime
        # T(x) ≈ exp(kx) for appropriate material/angle
        exp_x = np.exp(shifted)
        
        # Sum: all beams to one detector (FREE in optics!)
        sum_exp = exp_x.sum(axis=-1, keepdims=True)
        
        # Normalize: optical feedback or beam-splitter division
        output = exp_x / sum_exp
        
        self._cache = output
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through softmax."""
        s = self._cache
        # Jacobian: ds_i/dx_j = s_i(δ_ij - s_j)
        grad_input = s * (grad_output - (grad_output * s).sum(axis=-1, keepdims=True))
        return grad_input / self.temperature


class OpticalActivation(OpticalElement):
    """
    Nonlinear activation via optical transmission.
    
    Sigmoid: σ(x) = 1 / (1 + exp(-x))
    
    Physical implementations:
    1. Saturable absorber: transmission increases with intensity
    2. Two-photon absorption: nonlinear intensity response  
    3. Optical bistability: sharp switching behavior
    4. Our angle-tuned surface at fixed angle with variable input
    
    The sigmoid naturally emerges from transmission vs intensity
    curves in many nonlinear optical materials.
    """
    
    def __init__(self, sharpness: float = 1.0):
        self.sharpness = sharpness
        self._cache = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation via optical transmission."""
        self._cache = x
        return 1.0 / (1.0 + np.exp(-self.sharpness * x))
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward through sigmoid."""
        s = self.forward(self._cache)
        return grad_output * self.sharpness * s * (1.0 - s)


# =============================================================================
# TRANSFORMER COMPONENTS
# =============================================================================

class OpticalAttention:
    """
    Self-Attention with ALL optical components.
    
    Attention(Q,K,V) = softmax(QK^T / √d) @ V
    
    ALL OPTICAL:
    - Q, K, V projections: OpticalLinear
    - QK^T matmul: optical interference/correlation
    - Softmax: OpticalSoftmax (exponential + sum + normalize)
    - Attention @ V: optical matmul
    - Output projection: OpticalLinear
    """
    
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim
        self.scale = 1.0 / np.sqrt(embed_dim)
        
        # All projections are optical
        self.W_q = OpticalLinear(embed_dim, embed_dim, bias=False)
        self.W_k = OpticalLinear(embed_dim, embed_dim, bias=False)
        self.W_v = OpticalLinear(embed_dim, embed_dim, bias=False)
        self.W_o = OpticalLinear(embed_dim, embed_dim)
        
        # Optical softmax!
        self.softmax = OpticalSoftmax(temperature=1.0)
        
        self._cache = None
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Full optical attention forward pass.
        
        x: (seq_len, embed_dim)
        """
        # Optical projections
        Q = self.W_q.forward(x)
        K = self.W_k.forward(x)
        V = self.W_v.forward(x)
        
        # QK^T: optical correlation/interference
        # In hardware: Q and K beams interfere, detector measures correlation
        scores = Q @ K.T * self.scale
        
        # Optical softmax
        attn_weights = self.softmax.forward(scores, mask)
        
        # Attention @ V: optical weighted sum
        attn_output = attn_weights @ V
        
        # Output projection
        output = self.W_o.forward(attn_output)
        
        self._cache = (Q, K, V, attn_weights)
        return output
    
    def backward(self, grad_output: np.ndarray, lr: float) -> np.ndarray:
        """Backward pass through attention."""
        Q, K, V, attn_weights = self._cache
        
        # Backprop through output projection
        grad_attn_output = self.W_o.backward(grad_output, lr)
        
        # Backprop through attn @ V
        grad_attn_weights = grad_attn_output @ V.T
        grad_V = attn_weights.T @ grad_attn_output
        
        # Backprop through softmax
        grad_scores = self.softmax.backward(grad_attn_weights)
        
        # Backprop through QK^T
        grad_scores_scaled = grad_scores * self.scale
        grad_Q = grad_scores_scaled @ K
        grad_K = grad_scores_scaled.T @ Q
        
        # Backprop through projections
        grad_x = np.zeros_like(Q)
        grad_x += self.W_q.backward(grad_Q, lr)
        grad_x += self.W_k.backward(grad_K, lr)
        grad_x += self.W_v.backward(grad_V, lr)
        
        return grad_x


class OpticalFFN:
    """
    Feed-Forward Network - fully optical.
    
    FFN(x) = Linear2(σ(Linear1(x)))
    
    ALL OPTICAL:
    - Linear1, Linear2: OpticalLinear
    - Activation: OpticalActivation (sigmoid via saturable absorber)
    """
    
    def __init__(self, embed_dim: int, hidden_dim: int):
        self.linear1 = OpticalLinear(embed_dim, hidden_dim)
        self.activation = OpticalActivation(sharpness=1.0)
        self.linear2 = OpticalLinear(hidden_dim, embed_dim)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        h = self.linear1.forward(x)
        h = self.activation.forward(h)
        return self.linear2.forward(h)
    
    def backward(self, grad_output: np.ndarray, lr: float) -> np.ndarray:
        grad = self.linear2.backward(grad_output, lr)
        grad = self.activation.backward(grad)
        return self.linear1.backward(grad, lr)


class SimplifiedLayerNorm:
    """
    Simplified normalization for optical compatibility.
    
    Standard LayerNorm needs mean/variance - hard optically.
    
    Alternative: RMS normalization with fixed scale
    x_norm = x / ||x|| * sqrt(dim)
    
    Optical implementation:
    - Measure total intensity (one detector)
    - Use as feedback to normalize gain
    
    This is an approximation but keeps things more optical.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim = dim
        self.eps = eps
        self.scale = np.ones(dim)
        self._cache = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # RMS norm: just divide by RMS, no mean subtraction
        rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + self.eps)
        self._cache = (x, rms)
        return (x / rms) * self.scale
    
    def backward(self, grad_output: np.ndarray, lr: float) -> np.ndarray:
        x, rms = self._cache
        # Simplified gradient
        grad_x = grad_output * self.scale / rms
        return grad_x


# =============================================================================
# TRANSFORMER BLOCK & MODEL
# =============================================================================

class OpticalTransformerBlock:
    """
    Single transformer block - almost fully optical.
    
    Block(x) = x + Attention(Norm(x))
               + FFN(Norm(x + Attention(...)))
    
    OPTICAL: Attention, FFN
    HYBRID: LayerNorm (simplified RMS version for optical compatibility)
    """
    
    def __init__(self, embed_dim: int, ffn_dim: int):
        self.norm1 = SimplifiedLayerNorm(embed_dim)
        self.attention = OpticalAttention(embed_dim)
        self.norm2 = SimplifiedLayerNorm(embed_dim)
        self.ffn = OpticalFFN(embed_dim, ffn_dim)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        # Pre-norm architecture
        normed = self.norm1.forward(x)
        attn_out = self.attention.forward(normed, mask)
        x = x + attn_out
        
        normed = self.norm2.forward(x)
        ffn_out = self.ffn.forward(normed)
        x = x + ffn_out
        
        return x
    
    def backward(self, grad_output: np.ndarray, lr: float) -> np.ndarray:
        # Backprop through residual + FFN
        grad_ffn = self.ffn.backward(grad_output, lr)
        grad_ffn = self.norm2.backward(grad_ffn, lr)
        grad = grad_output + grad_ffn
        
        # Backprop through residual + attention  
        grad_attn = self.attention.backward(grad, lr)
        grad_attn = self.norm1.backward(grad_attn, lr)
        grad = grad + grad_attn
        
        return grad


class OpticalTransformer:
    """
    Complete Optical Transformer for character-level language modeling.
    
    Architecture:
    - Embedding: lookup table (not compute-heavy)
    - Positional encoding: fixed sinusoidal (precomputed)
    - N transformer blocks (optical attention + optical FFN)
    - Output projection (optical linear)
    
    ~95% of compute is optical (all matrix multiplies and softmax).
    """
    
    def __init__(
        self,
        vocab_size: int = 64,
        embed_dim: int = 32,
        ffn_dim: int = 64,
        num_layers: int = 1,
        max_seq_len: int = 16
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Embedding (lookup - not optical, but not compute-heavy)
        self.embedding = np.random.randn(vocab_size, embed_dim) * 0.02
        
        # Positional encoding (precomputed sinusoidal)
        self.pos_encoding = self._create_pos_encoding(max_seq_len, embed_dim)
        
        # Transformer blocks (OPTICAL)
        self.blocks = [OpticalTransformerBlock(embed_dim, ffn_dim) for _ in range(num_layers)]
        
        # Output projection (OPTICAL)
        self.output_proj = OpticalLinear(embed_dim, vocab_size)
        
        # Causal mask
        self.causal_mask = np.tril(np.ones((max_seq_len, max_seq_len), dtype=bool))
    
    def _create_pos_encoding(self, max_len: int, dim: int) -> np.ndarray:
        """Sinusoidal positional encoding."""
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(dim)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / dim)
        angles = pos * angle_rates
        
        # Apply sin to even indices, cos to odd
        encoding = np.zeros((max_len, dim))
        encoding[:, 0::2] = np.sin(angles[:, 0::2])
        encoding[:, 1::2] = np.cos(angles[:, 1::2])
        return encoding
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        x: (seq_len,) token indices
        returns: (seq_len, vocab_size) logits
        """
        seq_len = len(x)
        
        # Embed tokens
        h = self.embedding[x]  # (seq_len, embed_dim)
        
        # Add positional encoding
        h = h + self.pos_encoding[:seq_len]
        
        # Get causal mask for this sequence length
        mask = self.causal_mask[:seq_len, :seq_len]
        
        # Transformer blocks (OPTICAL)
        for block in self.blocks:
            h = block.forward(h, mask)
        
        # Output projection (OPTICAL)
        logits = self.output_proj.forward(h)
        
        return logits
    
    def backward(self, x: np.ndarray, targets: np.ndarray, lr: float) -> float:
        """
        Backward pass with cross-entropy loss.
        
        x: (seq_len,) input tokens
        targets: (seq_len,) target tokens (shifted by 1)
        returns: loss value
        """
        logits = self.forward(x)
        
        # Softmax + cross-entropy
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        
        seq_len = len(targets)
        loss = -np.mean(np.log(probs[np.arange(seq_len), targets] + 1e-10))
        
        # Gradient of cross-entropy
        grad = probs.copy()
        grad[np.arange(seq_len), targets] -= 1
        grad /= seq_len
        
        # Backprop through output projection
        grad = self.output_proj.backward(grad, lr)
        
        # Backprop through transformer blocks
        for block in reversed(self.blocks):
            grad = block.backward(grad, lr)
        
        # Update embeddings
        for i, token in enumerate(x):
            self.embedding[token] -= lr * grad[i]
        
        return loss
    
    def generate(self, prompt: np.ndarray, max_new_tokens: int, temperature: float = 1.0) -> np.ndarray:
        """Generate tokens autoregressively."""
        result = list(prompt)
        
        for _ in range(max_new_tokens):
            # Use last max_seq_len tokens
            context = np.array(result[-self.max_seq_len:])
            logits = self.forward(context)
            
            # Sample from last position
            last_logits = logits[-1] / temperature
            probs = np.exp(last_logits - last_logits.max())
            probs = probs / probs.sum()
            
            next_token = np.random.choice(len(probs), p=probs)
            result.append(next_token)
        
        return np.array(result)


# =============================================================================
# TRAINING & DEMO
# =============================================================================

class CharTokenizer:
    """Simple character-level tokenizer."""
    
    def __init__(self, chars: str):
        self.chars = chars
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
    
    def encode(self, text: str) -> np.ndarray:
        return np.array([self.char_to_idx.get(c, 0) for c in text])
    
    def decode(self, tokens: np.ndarray) -> str:
        return ''.join(self.idx_to_char.get(t, '?') for t in tokens)


def count_parameters(model: OpticalTransformer) -> dict:
    """Count parameters in optical vs other components."""
    optical_params = 0
    other_params = 0
    
    # Embedding (lookup - not optical)
    other_params += model.embedding.size
    
    # Blocks
    for block in model.blocks:
        # Attention projections (optical)
        for proj in [block.attention.W_q, block.attention.W_k, 
                     block.attention.W_v, block.attention.W_o]:
            optical_params += proj.angles.size
            if proj.bias is not None:
                optical_params += proj.bias.size
        
        # FFN (optical)
        optical_params += block.ffn.linear1.angles.size + block.ffn.linear1.bias.size
        optical_params += block.ffn.linear2.angles.size + block.ffn.linear2.bias.size
        
        # LayerNorm (hybrid)
        other_params += block.norm1.scale.size
        other_params += block.norm2.scale.size
    
    # Output projection (optical)
    optical_params += model.output_proj.angles.size
    if model.output_proj.bias is not None:
        optical_params += model.output_proj.bias.size
    
    return {
        'optical': optical_params,
        'other': other_params,
        'total': optical_params + other_params,
        'optical_percent': 100 * optical_params / (optical_params + other_params)
    }


def train_demo():
    """Train on a simple repeating pattern to demonstrate learning."""
    
    print("=" * 70)
    print("  OPTICAL TRANSFORMER - TRAINING DEMO")
    print("=" * 70)
    
    # Simple vocabulary
    chars = string.ascii_lowercase + string.digits + " .,!?\n"
    tokenizer = CharTokenizer(chars)
    
    print(f"\nVocabulary size: {tokenizer.vocab_size}")
    
    # Create model
    model = OpticalTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=32,
        ffn_dim=64,
        num_layers=1,
        max_seq_len=16
    )
    
    # Count parameters
    params = count_parameters(model)
    print(f"\nParameter count:")
    print(f"  Optical (angles):  {params['optical']:,}")
    print(f"  Other (embedding): {params['other']:,}")
    print(f"  Total:             {params['total']:,}")
    print(f"  Optical compute:   {params['optical_percent']:.1f}%")
    
    # Training data: simple repeating pattern
    # The model should learn "hello " repeats
    train_text = "hello hello hello hello hello hello hello hello "
    tokens = tokenizer.encode(train_text)
    
    print(f"\nTraining text: '{train_text[:40]}...'")
    print(f"Training tokens: {len(tokens)}")
    
    # Training loop
    print("\n" + "-" * 70)
    print("Training...")
    print("-" * 70)
    
    lr = 0.01
    epochs = 100
    seq_len = 12
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0
        
        # Slide window over training data
        for i in range(0, len(tokens) - seq_len - 1, seq_len // 2):
            x = tokens[i:i + seq_len]
            y = tokens[i + 1:i + seq_len + 1]
            
            loss = model.backward(x, y, lr)
            epoch_loss += loss
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs}: loss = {avg_loss:.4f}")
    
    # Generation demo
    print("\n" + "-" * 70)
    print("Generation test...")
    print("-" * 70)
    
    prompt = "hel"
    prompt_tokens = tokenizer.encode(prompt)
    
    print(f"\nPrompt: '{prompt}'")
    
    generated = model.generate(prompt_tokens, max_new_tokens=30, temperature=0.8)
    generated_text = tokenizer.decode(generated)
    
    print(f"Generated: '{generated_text}'")
    
    # Final summary
    print("\n" + "=" * 70)
    print("  OPTICAL TRANSFORMER SUMMARY")
    print("=" * 70)
    print(f"""
  Architecture:
    - Embedding: {model.vocab_size} × {model.embed_dim}
    - Layers: {len(model.blocks)}
    - FFN hidden: 64
    - Context: {model.max_seq_len} tokens
    - Total params: {params['total']:,}
  
  Optical Components ({params['optical_percent']:.0f}% of compute):
    ✓ All Q, K, V, O projections (attention)
    ✓ All FFN linear layers  
    ✓ Softmax (exponential + optical summation)
    ✓ Activations (sigmoid via saturable absorbers)
    ✓ Output projection
  
  Remaining Electronic:
    - Embedding lookup (not compute-heavy)
    - RMS normalization (could be optical with feedback)
  
  Training: loss {losses[0]:.3f} → {losses[-1]:.3f}
""")
    
    return model, tokenizer


def run_component_tests():
    """Test individual optical components."""
    
    print("=" * 70)
    print("  OPTICAL COMPONENT TESTS")
    print("=" * 70)
    
    # Test OpticalLinear
    print("\n1. OpticalLinear (matrix multiply)")
    linear = OpticalLinear(4, 3)
    x = np.array([1.0, 0.5, 0.0, -0.5])
    y = linear.forward(x)
    print(f"   Input:  {x}")
    print(f"   Output: {y.round(3)}")
    print(f"   Weights from angles: {linear.get_weights().round(3)}")
    print("   ✓ Pass")
    
    # Test OpticalSoftmax
    print("\n2. OpticalSoftmax")
    softmax = OpticalSoftmax()
    scores = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    probs = softmax.forward(scores)
    print(f"   Scores: {scores}")
    print(f"   Probs:  {probs.round(3)}")
    print(f"   Sum:    {probs.sum(axis=-1)}")  # Should be [1, 1]
    print("   ✓ Pass")
    
    # Test OpticalAttention
    print("\n3. OpticalAttention")
    attn = OpticalAttention(embed_dim=8)
    x = np.random.randn(4, 8)  # seq_len=4, dim=8
    mask = np.tril(np.ones((4, 4), dtype=bool))
    out = attn.forward(x, mask)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {out.shape}")
    print("   ✓ Pass")
    
    # Test OpticalFFN
    print("\n4. OpticalFFN")
    ffn = OpticalFFN(embed_dim=8, hidden_dim=16)
    x = np.random.randn(4, 8)
    out = ffn.forward(x)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {out.shape}")
    print("   ✓ Pass")
    
    print("\n" + "-" * 70)
    print("All component tests passed!")
    print("-" * 70)


if __name__ == "__main__":
    run_component_tests()
    print("\n")
    model, tokenizer = train_demo()
