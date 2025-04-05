# Motion Planners

This module provides motion planning algorithms and dynamics models for trajectory generation within bounded environments. It enables planning paths for agents while respecting specified boundaries and constraints.

## Overview

The motion_planners module consists of two main components:

- **dynamics.py**: Defines environment boundaries and collision detection
- **planner.py**: Implements various motion planning strategies

## Dynamics Module

The `Dynamics` class handles boundary management and collision detection with the following key features:

- Convex polygon boundary representation
- Point containment checking
- Boundary vector calculation
- Distance calculations from points to boundary segments
- Binary search for edge detection

### Helper Functions

- `chord_length_to_angle`: Converts chord length to angle for circular paths
- `a_to_b_in_stepsize`: Generates points between two positions with consistent step sizes

## Planner Module

The module provides an abstract `Planner` base class and several concrete implementations:

### Base Planner

The abstract `Planner` class defines the common interface with methods:
- `yield_points`: Generate trajectory points (abstract)
- `get_bounce_pos_and_new_direction`: Calculate position and direction after boundary bounce
- `random_direction`: Generate a random direction vector
- `get_planner_start_position`: Get the planner's initial position
- `stop`: Stop the planning process

### Implemented Planners

1. **BouncePlanner**: Generates trajectories that bounce off environment boundaries
   - Handles bounce physics with slight randomization
   - Prevents stalling by detecting and resolving stuck points

2. **StationaryPlanner**: Moves to a specified point and remains stationary
   - First navigates to the target point
   - Then yields the same point repeatedly

3. **PointCycle**: Cycles through a sequence of predefined points
   - Takes an arbitrary list of points
   - Generates a trajectory visiting each point in sequence

4. **CirclePlanner**: Generates a circular trajectory
   - Creates points along a circle with specified diameter and center
   - Controls direction (clockwise/counter-clockwise)
   - Maintains consistent step size along the circle

## Usage Example

```python
import numpy as np
from spf.motion_planners.dynamics import Dynamics
from spf.motion_planners.planner import CirclePlanner

# Define a square boundary
boundary = np.array([
    [1, 1],
    [1, -1],
    [-1, -1],
    [-1, 1]
])

# Create dynamics with the boundary
dynamics = Dynamics(boundary)

# Create a circular planner
start_point = np.array([0, 0.8])
circle_planner = CirclePlanner(
    dynamics=dynamics,
    start_point=start_point,
    step_size=0.1,
    circle_diameter=1.5
)

# Generate and use trajectory points
for point in circle_planner.yield_points():
    # Process each point in the trajectory
    print(point)
    
    # Stop planning when needed
    if some_condition:
        circle_planner.stop()
        break
```

## Dependencies

- numpy: For numerical operations
- matplotlib: For path operations
- scipy: For convex hull calculations 