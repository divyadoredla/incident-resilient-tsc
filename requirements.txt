# Incident-Resilient Traffic Signal Control

An OpenEnv-compliant reinforcement learning environment for traffic signal control that emphasizes robustness under real-world disruptions.

---

## Overview

Standard traffic signal RL agents fail when unexpected events occur — lane closures, demand spikes, sensor failures. This environment trains agents to be *resilient*, not just optimal.

The core innovation is the `DisruptionWrapper`, which injects real-world disturbances during training so agents learn adaptive recovery strategies.

---

## Action Space

Each action is a dictionary mapping intersection IDs to signal phases:

```json
{"intersection_0": 1, "intersection_1": 0, "intersection_2": 2}
```

- Phase 0/2: Green for north-south traffic
- Phase 1/3: Green for east-west traffic
- Range: 0–3 per intersection

## Observation Space

```json
{
  "vehicle_counts":  {"intersection_0_lane_0": 12, ...},
  "signal_phases":   {"intersection_0": 1, ...},
  "waiting_times":   {"intersection_0_lane_0": 23.5, ...},
  "disruptions":     {"intersection_0_lane_0": false, ...},
  "throughput":      0.72
}
```

---

## Tasks

| Task | Difficulty | Intersections | Disruption Rate | Description |
|------|-----------|---------------|-----------------|-------------|
| `basic_intersection` | Easy | 1 | 10% | Single intersection, minimal disruptions |
| `multi_intersection` | Medium | 3 | 20% | Coordinate 3 intersections with moderate disruptions |
| `city_network` | Hard | 6 | 30% | City-wide network with severe, frequent disruptions |

---

## Disruption Types

- Lane Blockage: A lane is fully blocked, forcing rerouting
- Demand Spike: Vehicle arrival rate multiplies 2–4x suddenly
- Sensor Failure: Observation noise corrupts vehicle count readings

---

## Reward Function

Reward is computed at every step (range: -1.0 to 1.0):

```
reward = 0.4 * throughput_score
       + 0.4 * waiting_time_score
       + disruption_penalty        # -0.1 per active disruption
       + disruption_bonus          # +0.1 per disruption handled well
```

Incremental feedback is provided throughout the episode, not just at completion.

---

## Grading Criteria

Each task grader returns a score in [0.0, 1.0]:

- `BasicIntersectionGrader`: 50% avg reward + 30% waiting time + 20% stability
- `MultiIntersectionGrader`: 40% avg reward + 30% disruption score + 30% recovery rate
- `CityNetworkGrader`: 30% avg reward + 40% stress performance + 20% adaptation + 10% stability

Grading is deterministic and reproducible given the same random seed.

---

## Setup & Usage

### Local

```bash
pip install -r requirements.txt
export HF_TOKEN=your_token_here
python inference.py
```

### Docker

```bash
docker build -t traffic-signal-env .
docker run -e HF_TOKEN=your_token_here traffic-signal-env
```

### Custom model/endpoint

```bash
docker run \
  -e HF_TOKEN=your_token \
  -e API_BASE_URL=https://your-endpoint/v1 \
  -e MODEL_NAME=your-model \
  traffic-signal-env
```

---

## Baseline Performance

| Task | Score | Difficulty |
|------|-------|-----------|
| basic_intersection | ~0.62 | Easy |
| multi_intersection | ~0.48 | Medium |
| city_network | ~0.35 | Hard |

Scores obtained using `gpt-4.1-mini` with greedy action selection.

---

## Tags

`openenv` `traffic` `reinforcement-learning` `resilience` `smart-city`
