# Well-being Simulation Model

An agent-based model (ABM) for simulating the dynamics of well-being in networked populations. This model captures how individual well-being evolves over time through the interplay of capital accumulation, social comparison, adaptation, and network structure.

## Overview

The `WellbeingSim` class implements a sophisticated simulation of well-being dynamics that incorporates key insights from behavioral economics, social psychology, and network science:

- **Adaptation/Habituation**: People adapt to their circumstances over time, with both personal history and social comparisons influencing their reference points
- **Social Comparison**: Well-being depends not just on absolute resources but on how one compares to others in their social network
- **Loss Aversion**: Negative changes have a larger impact on well-being than equivalent positive changes
- **Network Effects**: Social network structure influences who people compare themselves to and how well-being spreads through populations
- **Feedback Loops**: Well-being can feed back into capital accumulation, creating positive or negative spirals

## Key Features

### Network Types
- **Random Networks** (Erdős-Rényi, `h=-1`): Uniformly random connections
- **Preferential Attachment** (Barabási-Albert): Scale-free networks where well-connected nodes attract more connections
  - `h=-2`: Normal BA network (unrelated to capital)
  - `h=-3`: Low capital agents have high degree
  - `h=-4`: High capital agents have high degree
- **Spatially Distributed Attachment (SDA, `0<h<2`)**: Homophily-based networks where similar agents are more likely to connect
  - Note: Values `0<h<0.2` may cause optimization issues; use `h=-1` for fully random networks instead

### Social Comparison Mechanisms
- **Quantile Comparison**: Compare to a specific percentile of your network (e.g., median, 75th percentile)

### Well-being Functions
- **Sigmoid Perception** (default): S-shaped utility function with asymmetric loss aversion and capital-dependent sensitivity
- **Concave Perception**: Square-root based concave utility with diminishing sensitivity to large changes

### Shock Mechanisms
- **Fixed Interventions**: Regular deterministic shocks to random subsets of agents
- **Random Shocks**: Continuous stochastic fluctuations (Normal, GEV, or Student-t distributions)
- **Combined**: Both types of shocks simultaneously (e.g., `'fixed+random'`)

## Parameter Reference

### Core Simulation Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `N` | int | Number of agents in the simulation | Yes |
| `n_steps` | int | Number of simulation time steps | Yes |
| `seed` | int | Random seed for reproducibility | Optional |

### Network Configuration

| Parameter | Type | Description | Default/Options |
|-----------|------|-------------|-----------------|
| `network_type` | str | Network topology | `'random'`, `'preferential'`, `'sda'` |
| `homophily` | float | Homophily strength (see details below) | `-4` to `2` |
| `avg_degree` | int | Average connections per node | Positive integer |
| `p_rewire` | float | Probability of rewiring edges per step | `0.0` to `1.0` |
| `wb_in_dist` | bool | Include well-being in distance calculations | `True`/`False` |

**Homophily Parameter (`h`) Values:**
- **`h = -1`**: Random Erdős-Rényi network
- **`h = -2`**: Barabási-Albert (preferential attachment), no capital-degree correlation
- **`h = -3`**: Barabási-Albert with poor agents having low degree
- **`h = -4`**: Barabási-Albert with poor agents having high degree
- **`0.2 < h < 2`**: True SDA network with homophily (similar agents connect)
  - **Note**: Values `0 < h < 0.2` may cause optimization issues in SDA
  - Use `h = -1` for fully random networks instead of low positive values

### Social Comparison Parameters

| Parameter | Type | Description | Options/Range |
|-----------|------|-------------|---------------|
| `soc_comparison` | bool | Enable social comparison | `True`/`False` |
| `comparison_kind` | str | Comparison mechanism | `'quantile'` |
| `quantile` | float | Reference percentile for comparison | `0.0` to `1.0` (e.g., `0.5` = median, `0.75` = 75th percentile) |
| `soc_comp_w` | float or np.array | Weight of social comparison | `0.0` (none) to `1.0` (only social), can be agent-specific array |

### Adaptation Parameters

| Parameter | Type | Description | Range |
|-----------|------|-------------|-------|
| `alpha` | float or np.array | Adaptation rate to changes | `0.0` (no adaptation) to `1.0` (instant), can be agent-specific array |
| `soc_to_wb` | bool or float | Social capital benefit to well-being | `True`/`False` or float value |

### Capital Dynamics

| Parameter | Type | Description | Options/Range |
|-----------|------|-------------|---------------|
| `capital_dist` | str | Initial capital distribution | `'lognormal'`, `'beta'`, `'homogeneous'` |
| `mu` | float | Distribution parameter 1 | For lognormal: mean of log; for beta: alpha parameter |
| `sigma` | float | Distribution parameter 2 | For lognormal: std of log; for beta: beta parameter |
| `growth_rate` | float or np.array | Base capital growth rate | Typically `0.0` to `0.05`, can be agent-specific array |
| `wb_to_cap` | float or np.array | Well-being feedback strength | Any positive float, can be agent-specific array |
| `shock_sigma` | float | Standard deviation of random shocks | Any positive float |

### Well-being Perception Function

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `perception` | str | Type of well-being function | `'sigmoid'` (default), `'concave'` |
| `slope_sig` | float | Sigmoid slope parameter | Typically negative, e.g., `-0.0001` |

**Note**: When using sigmoid perception with `N > 10`, the model automatically adjusts slope based on capital distribution to create wealth-dependent sensitivity.

### Shock/Intervention Settings

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `event_type` | str | Type of shocks to apply | `'fixed'`, `'random'`, `'fixed+random'` |
| `p` | float | Probability of receiving fixed intervention | `0.0` to `1.0` |
| `int_freq` | int | Fixed intervention frequency (time steps) | Positive integer (e.g., `50` = every 50 steps) |
| `int_size` | float | Absolute size of intervention | Any float (negative for adverse shocks) |
| `rel_size` | float | Relative intervention size | `0.0` to `1.0` (fraction of current capital) |

**Intervention Logic**:
- If `rel_size > 0`: At step `int_freq`, treated agents' capital is multiplied by `(1 - rel_size)`
- If `int_size != 0`: At regular intervals, `int_size` is added to treated agents' capital
- Both can be used simultaneously


**Note**: This README describes the `WellbeingSim` class implementation. For questions about specific theoretical assumptions or modeling choices, please refer to the paper.