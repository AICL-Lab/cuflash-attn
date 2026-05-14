# Architecture Overview

This page provides a comprehensive architectural view of CuFlash-Attn, designed for researchers and engineers who need to understand the system design.

---

## System Architecture

```mermaid
graph TB
    subgraph API["API Layer"]
        CPP["C++ Namespace API<br/>cuflash::flash_attention_*"]
        CABI["C ABI<br/>cuflash_attention_*"]
        PY["Python Binding<br/>ctypes interface"]
    end

    subgraph DISPATCH["Dispatch Layer"]
        VALID["Parameter Validation"]
        SELECT["Kernel Selection<br/>by dtype & arch"]
        LAUNCH["Kernel Launch"]
    end

    subgraph KERNEL["Kernel Layer"]
        FWD["Forward Kernels<br/>flash_attention_forward_*"]
        BWD["Backward Kernels<br/>flash_attention_backward_*"]
        UTIL["Utilities<br/>softmax, reduction"]
    end

    subgraph HW["Hardware Layer"]
        SM["Shared Memory<br/>SRAM Tiling"]
        REG["Register File<br/>Accumulators"]
        HBM["HBM<br/>Global Memory"]
    end

    CPP --> VALID
    CABI --> VALID
    PY --> VALID

    VALID --> SELECT
    SELECT --> LAUNCH
    LAUNCH --> FWD
    LAUNCH --> BWD

    FWD --> SM
    FWD --> REG
    BWD --> SM
    BWD --> REG
    UTIL --> SM

    SM --> HBM
    REG --> HBM
```

---

## Data Flow

### Forward Pass

```mermaid
sequenceDiagram
    participant Host as Host
    participant GPU as GPU Kernel
    participant SRAM as Shared Memory
    participant REG as Registers

    Host->>GPU: Launch kernel (Q, K, V)
    
    loop For each tile
        GPU->>SRAM: Load Q_tile, K_tile, V_tile
        SRAM->>REG: Compute QK^T (partial)
        REG->>REG: Online softmax update
        REG->>REG: Accumulate O_partial
    end
    
    REG->>GPU: Write O, L (log-sum-exp)
    GPU->>Host: Return
```

### Backward Pass

```mermaid
sequenceDiagram
    participant Host as Host
    participant GPU as GPU Kernel
    participant SRAM as Shared Memory
    participant REG as Registers

    Host->>GPU: Launch kernel (Q, K, V, O, L, dO)
    
    Note over GPU: Recompute attention if needed
    
    loop For each tile
        GPU->>SRAM: Load tiles
        SRAM->>REG: Compute dQ, dK, dV partials
        REG->>REG: Accumulate gradients
    end
    
    REG->>GPU: Write dQ, dK, dV
    GPU->>Host: Return
```

---

## Memory Layout

```mermaid
graph LR
    subgraph Input["Input Tensors"]
        Q["Q: [B, H, N, D]"]
        K["K: [B, H, N, D]"]
        V["V: [B, H, N, D]"]
    end

    subgraph Output["Output Tensors"]
        O["O: [B, H, N, D]"]
        L["L: [B, H, N]"]
    end

    subgraph Gradients["Gradient Tensors"]
        dO["dO: [B, H, N, D]"]
        dQ["dQ: [B, H, N, D]"]
        dK["dK: [B, H, N, D]"]
        dV["dV: [B, H, N, D]"]
    end

    Q --> O
    K --> O
    V --> O
    Q --> L
    K --> L

    O --> dQ
    O --> dK
    O --> dV
    dO --> dQ
    dO --> dK
    dO --> dV
```

---

## Kernel Tiling Strategy

### Tile Dimensions

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `B_r` | Query tile size | 128 |
| `B_c` | Key/Value tile size | 64 |
| `D` | Head dimension | 64, 128 |
| `T_r` | Threads per query tile | 128 |

### Memory Complexity

$$
\text{SRAM} = O(B_r \times D + B_c \times D + B_r \times B_c)
$$

For typical values ($B_r=128, B_c=64, D=128$):

$$
\text{SRAM} = 128 \times 128 + 64 \times 128 + 128 \times 64 = 32\text{KB}
$$

---

## Directory Structure

```
cuflash-attn/
├── include/cuflash/          # Public API headers
│   ├── flash_attention.h     # C++ namespace API
│   └── flash_attention_c.h   # C ABI
├── src/
│   ├── api/                  # API dispatch layer
│   │   └── flash_attention_api.cu
│   ├── forward/              # Forward kernels
│   │   ├── forward_kernel_f32.cu
│   │   └── forward_kernel_f16.cu
│   ├── backward/             # Backward kernels
│   │   ├── backward_kernel_f32.cu
│   │   └── backward_kernel_f16.cu
│   └── kernels/              # Shared utilities
│       ├── softmax.cuh
│       └── memory.cuh
└── tests/
    ├── unit/                  # Unit tests
    └── integration/           # Integration tests
```

---

## Error Handling Flow

```mermaid
graph TD
    INPUT[API Call] --> VALID{Validate Params}
    VALID -->|Invalid| ERR[Return Error Code]
    VALID -->|Valid| ALLOC{Allocate Memory}
    ALLOC -->|Fail| ERR
    ALLOC -->|Success| LAUNCH[Launch Kernel]
    LAUNCH -->|CUDA Error| ERR
    LAUNCH -->|Success| SYNC[Synchronize Stream]
    SYNC -->|CUDA Error| ERR
    SYNC -->|Success| SUCCESS[Return SUCCESS]
```

---

## Performance Characteristics

| Operation | Memory | Compute | Bandwidth Bound |
|-----------|--------|---------|-----------------|
| Forward | $O(N)$ | $O(N^2)$ | Yes (low D) |
| Backward | $O(N)$ | $O(N^2)$ | Yes (low D) |
| Recompute | $O(1)$ | $O(N^2)$ | Yes |

::: tip Key Insight
FlashAttention reduces memory from $O(N^2)$ to $O(N)$ by never materializing the full attention matrix. The trade-off is recomputing attention scores during the backward pass, which is compute-bound and thus efficient on modern GPUs.
:::
