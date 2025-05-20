Battery-Swap Routing Optimization Solution
Assumptions

Geographic Area: Riders operate within Pune, India's approximate bounds (18.4-18.6°N, 73.7-73.9°E)
Vehicle Speed: 20 km/h average speed in urban areas
Battery Profile: Linear discharge rate of 4% SOC per km
Distance Calculation: Haversine formula for direct distance, multiplied by 1.3 to simulate road networks
Rider Priority: Riders with lower SOC and those on active gigs receive higher priority
Queue Management: First-come-first-served at swap stations, with live updates
Time Windows: 60-minute planning horizon, with minute-by-minute simulation
Rider Behavior: Riders follow assigned swap instructions and travel at consistent speeds
Planning Trigger: Riders are assigned to swaps when SOC drops below 25% or is predicted to reach critical levels during their gig

Algorithm Description
The solution employs a greedy heuristic with time-based simulation to optimize battery swaps:
1. Initialization

Create snapshot of all riders, stations, and system state
Compute distances between all riders and stations
Identify riders needing immediate swaps (SOC < 25%)
Initialize event queue with rider gig completion times

2. Time-Step Simulation (minute by minute for 60 minutes)

For each time step:

Update rider positions and SOC based on their status
Complete any scheduled swaps
Update station queue lengths
Check for riders finishing gigs
Identify and schedule new riders needing swaps



3. Swap Assignment Logic
For each rider needing a swap:

Calculate reachable stations (where rider can arrive with SOC > 10%)
For each reachable station:

Calculate detour distance from current path
Estimate queue time upon arrival
Calculate total cost (weighted sum of detour distance and wait time)


Assign rider to the station with minimum cost
Update station queue projections

4. Queue Management

Maintain a time-projected queue length for each station
When assigning riders to stations, update the projected queue length for their arrival time
If all stations exceed queue capacity (5), select least congested station

Meeting Objectives
1. SOC never below 10%

Riders are scheduled for swaps when SOC reaches 25%
Algorithm calculates if riders can reach stations before dropping below 10%
Riders unreachable to any station are flagged for emergency assistance

2. Minimize Total Detour Kilometers

For each rider, the algorithm calculates the detour distance to each station
Station assignment optimizes for minimum detour while considering queue times
Incorporates rider's current gig path when scheduling swaps

3. Station Queues Never Exceed 5 Riders

Maintains projected queue lengths at each time step
Uses load balancing to distribute riders across stations when queues grow
If all stations approach capacity, prioritizes riders with lowest SOC

Scalability Considerations
Current Performance

Algorithm complexity: O(r × s × t) where:

r = number of riders
s = number of stations
t = number of time steps



Scaling Improvements

Geographic Partitioning: Divide city into zones, process independently
Rider Filtering: Only consider riders likely to need swaps in next 30 minutes
Parallel Processing: Station assignments can be calculated in parallel
Pre-computation: Cache distance matrices between common locations
Event-Driven Simulation: Switch from time-step to event-driven simulation for larger fleets
Dynamic Reoptimization: Recalculate plan every 5-10 minutes as conditions change

Limitations

Optimality not guaranteed due to greedy approach
No rider route optimization beyond station assignments
Queue predictions may deviate as real-world conditions change
