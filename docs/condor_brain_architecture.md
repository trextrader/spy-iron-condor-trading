# CondorBrain Architecture (v2.2)

This document visualizes the **CondorBrain v2.2** model architecture currently being trained on Kaggle. It features **Mamba-2 State Space Models**, **Volatility-Gated Attention**, and a **Sparse Top-K Mixture-of-Experts** head.

## Architecture Flowchart

```mermaid
graph TD
    subgraph Inputs
    Input[Input Tensor<br/>(Batch, Seq=240, Feat=24)]
    Norm[Scaling & Normalization]
    Proj[Input Projection<br/>Linear(24 -> 1024)]
    end

    subgraph "Deep Sequence Modeling (Mamba-2)"
    M1[Mamba Block 0..7]
    V1[<b>VolGatedAttn</b><br/>(Volatility Awareness)]
    M2[Mamba Block 8..15]
    V2[<b>VolGatedAttn</b><br/>(Volatility Awareness)]
    M3[Mamba Block 16..23]
    V3[<b>VolGatedAttn</b><br/>(Volatility Awareness)]
    
    FinalNorm[RMSNorm]
    LastState[Last Hidden State<br/>(Batch, 1024)]
    end

    subgraph "Intelligent Output Head"
    Regime[Regime Detector<br/>(Bull/Bear/Crash)]
    
    subgraph "Top-K Mixture of Experts"
    Router[<b>Router Gate</b><br/>Select Top-1 Expert]
    Experts[<b>Sparse Experts</b><br/>(Linear Projections)]
    end
    
    Output[<b>Final Output</b><br/>(8 Iron Condor Params)]
    end

    %% Flow Connections
    Input --> Norm --> Proj
    Proj --> M1
    M1 --> V1
    V1 --> M2
    M2 --> V2
    V2 --> M3
    M3 --> V3
    V3 --> FinalNorm
    FinalNorm --> LastState
    
    LastState --> Regime
    LastState --> Router
    Router -- "Routing Weights" --> Experts
    Experts --> Output
    
    %% Styling
    style V1 fill:#ff9900,stroke:#333,color:black
    style V2 fill:#ff9900,stroke:#333,color:black
    style V3 fill:#ff9900,stroke:#333,color:black
    style Router fill:#00ccff,stroke:#333,color:black
    style Output fill:#66ff66,stroke:#333,color:black
```

## Key Components

### 1. Volatility-Gated Attention (`VolGatedAttn`)
Inserted after every 8 Mamba layers, this module allows the model to "pause" and attend to specific high-volatility events globally across the sequence, fixing Mamba's potential "forgetfulness" of distant spikes.

### 2. Top-K Mixture of Experts (`TopKMoE`)
Instead of a single dense output layer, we use a sparse router that selects the single best "Expert" network for the current market condition. This allows distinct sub-networks to specialize (e.g., one for "Calm Bull", one for "Crash").

### 3. Regime Detector
A parallel auxiliary head that explicitly classifies the market state (Low Vol, High Vol, Trend). This provides interpretability and can guide the MoE router.
