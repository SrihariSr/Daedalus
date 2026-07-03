# Daedalus

A physics-informed differentiable vehicle-dynamics model for Formula 1.

## Overview

Daedalus learns a model of an F1 car's dynamics from publicly available telemetry, and uses that model as the foundation for optimal-lap analysis and setup sensitivity studies. The learned dynamics function maps (current state, driver inputs) to next state, with physical priors (energy conservation, kinematic consistency, tyre friction limits) enforced as training constraints rather than left to the network to discover.

Because the dynamics are end-to-end, the resulting simulator supports gradient-based optimisation on top, including finding the theoretical optimal lap and computing sensitivities of lap time to setup parameters.

## Approach

Development is staged in tiers, each producing a shippable artefact:

**Tier 1**: Lap simulator with quantitative validation against observed lap times.

**Tier 2**: Optimal-lap solver that computes the theoretical fastest lap for a given car and circuit, and quantifies where each driver leaves time on the table.

**Tier 3**: Setup sensitivity analysis and multi-circuit generalisation.

The initial build targets Silverstone using 2024 British GP data. Once the methodology is confirmed on one circuit, it will be extended to arbitrary circuits and cars.

## Current status

Centreline reconstruction with track-local coordinate projection is complete and passes synthetic verification. Real Silverstone verification is pending. Everything else is under active construction.

## Getting started

Dependencies: `numpy`, `scipy`, `pandas`, `fastf1`.

Build the Silverstone centreline (requires internet on first run for FastF1 to fetch session data):

```bash
python scripts/build_silverstone_centreline.py
```

Run the synthetic verification of the centreline algorithm (no FastF1 required):

```bash
python tests/test_centreline_synthetic.py
```

## Data source

All telemetry from FastF1, which pulls from Formula 1's public timing data. No proprietary data is used.

## Author

Srihari Srinivasan, 2026