import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import math
import heapq
from typing import List, Dict, Tuple, Any
import random

# Constants
BATTERY_CONSUMPTION_PCT_PER_KM = 4.0  # % SOC per km
SWAP_TIME_MINUTES = 4  # minutes
MAX_QUEUE_LENGTH = 5  # riders
SIM_DURATION_MINUTES = 60  # 60-minute planning horizon
CRITICAL_SOC_PCT = 10  # Minimum allowed SOC percentage
SWAP_TRIGGER_SOC_PCT = 25  # SOC percentage that triggers swap planning
AVG_SPEED_KM_PER_HOUR = 20  # Average speed in urban areas
ROAD_NETWORK_FACTOR = 1.3  # Multiplier to convert direct distance to road distance

# Pune city bounds for random generation
PUNE_LAT_BOUNDS = (18.45, 18.6)
PUNE_LNG_BOUNDS = (73.75, 73.9)

# Helper functions
def haversine_distance(lat1, lng1, lat2, lng2):
    """Calculate the great circle distance between two points in kilometers."""
    # Convert decimal degrees to radians
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r

def get_road_distance(lat1, lng1, lat2, lng2):
    """Calculate road distance as a factor of haversine distance."""
    return haversine_distance(lat1, lng1, lat2, lng2) * ROAD_NETWORK_FACTOR

def time_to_travel(distance_km):
    """Calculate time in minutes to travel a given distance."""
    return (distance_km / AVG_SPEED_KM_PER_HOUR) * 60

def soc_after_travel(current_soc, distance_km):
    """Calculate SOC after traveling a given distance."""
    return current_soc - (distance_km * BATTERY_CONSUMPTION_PCT_PER_KM)

def can_reach_with_safe_soc(rider, station):
    """Check if rider can reach station with SOC > CRITICAL_SOC_PCT."""
    distance = get_road_distance(
        rider['lat'], rider['lng'], 
        station['lat'], station['lng']
    )
    expected_soc = soc_after_travel(rider['soc_pct'], distance)
    return expected_soc > CRITICAL_SOC_PCT, distance, expected_soc

def generate_mock_data(num_riders=100, num_stations=3):
    """Generate mock data for riders and stations."""
    # Generate stations
    stations = []
    # Create stations in different parts of Pune
    station_locs = [
        (18.52, 73.85),  # Central Pune
        (18.56, 73.78),  # Northwest Pune
        (18.48, 73.89)   # Southeast Pune
    ]
    
    for i in range(num_stations):
        lat, lng = station_locs[i] if i < len(station_locs) else (
            random.uniform(*PUNE_LAT_BOUNDS),
            random.uniform(*PUNE_LNG_BOUNDS)
        )
        stations.append({
            'station_id': f"S_{chr(65+i)}",  # S_A, S_B, S_C, ...
            'lat': lat,
            'lng': lng,
            'queue_len': random.randint(0, 2),  # Initial queue length
            'queue_projected': []  # Will store [timestamp, projected_queue_length] pairs
        })
    
    # Generate riders
    riders = []
    current_time = datetime.now().replace(microsecond=0)
    
    for i in range(num_riders):
        rider_id = f"R{i:03d}"
        status = random.choice(["idle", "on_gig"]) 
        
        # Random location in Pune
        lat = random.uniform(*PUNE_LAT_BOUNDS)
        lng = random.uniform(*PUNE_LNG_BOUNDS)
        
        # SOC between 15% and 90%
        soc_pct = random.uniform(15, 90)
        
        rider_data = {
            'rider_id': rider_id,
            'lat': lat,
            'lng': lng,
            'soc_pct': soc_pct,
            'status': status,
        }
        
        if status == "on_gig":
            # Random destination within 1-5 km
            dest_angle = random.uniform(0, 2 * math.pi)
            dest_distance = random.uniform(1, 5)  # km
            
            # Calculate destination coordinates
            # Approximation: 1 degree lat ≈ 111 km, 1 degree lng ≈ 111*cos(lat) km
            dest_lat = lat + (dest_distance * math.cos(dest_angle) / 111)
            dest_lng = lng + (dest_distance * math.sin(dest_angle) / (111 * math.cos(math.radians(lat))))
            
            # Keep destination within Pune bounds
            dest_lat = max(PUNE_LAT_BOUNDS[0], min(PUNE_LAT_BOUNDS[1], dest_lat))
            dest_lng = max(PUNE_LNG_BOUNDS[0], min(PUNE_LNG_BOUNDS[1], dest_lng))
            
            # Calculate km to finish and estimated finish time
            km_to_finish = get_road_distance(lat, lng, dest_lat, dest_lng)
            minutes_to_finish = time_to_travel(km_to_finish)
            est_finish_ts = current_time + timedelta(minutes=minutes_to_finish)
            
            rider_data.update({
                'dest_lat': dest_lat,
                'dest_lng': dest_lng,
                'km_to_finish': km_to_finish,
                'est_finish_ts': est_finish_ts.isoformat()
            })
        
        riders.append(rider_data)
    
    return riders, stations, current_time

class BatterySwapOptimizer:
    def __init__(self, riders, stations, current_time):
        self.riders = {r['rider_id']: r for r in riders}
        self.stations = {s['station_id']: s for s in stations}
        self.current_time = current_time
        self.start_time = current_time
        self.end_time = current_time + timedelta(minutes=SIM_DURATION_MINUTES)
        
        # Initialize swap plan
        self.swap_plan = []
        
        # Initialize queue projections for each station
        for station_id, station in self.stations.items():
            station['queue_projected'] = [(current_time, station['queue_len'])]
        
        # Initialize event queue for simulation
        self.event_queue = []
        
        # Add gig completion events
        for rider_id, rider in self.riders.items():
            if rider['status'] == 'on_gig':
                finish_time = datetime.fromisoformat(rider['est_finish_ts'])
                if finish_time <= self.end_time:
                    heapq.heappush(
                        self.event_queue, 
                        (finish_time, 'gig_completion', rider_id)
                    )
    
    def get_projected_queue_length(self, station_id, arrival_time):
        """Get projected queue length at a station at a specific time."""
        station = self.stations[station_id]
        queue_projected = station['queue_projected']
        
        # Find the queue length projection for the given time
        for i in range(len(queue_projected) - 1):
            if queue_projected[i][0] <= arrival_time < queue_projected[i + 1][0]:
                return queue_projected[i][1]
        
        # If arrival time is after all projections, use the last projection
        return queue_projected[-1][1]
    
    def update_queue_projection(self, station_id, arrival_time, departure_time):
        """Update queue projection for a station based on a new swap."""
        station = self.stations[station_id]
        queue_projected = station['queue_projected']
        
        # Find or create entry for arrival time
        arrival_idx = None
        for i, (time, _) in enumerate(queue_projected):
            if time == arrival_time:
                arrival_idx = i
                break
            if time > arrival_time:
                # Insert new entry at this position
                arrival_idx = i
                queue_projected.insert(i, (arrival_time, queue_projected[i-1][1] + 1))
                break
        
        if arrival_idx is None:
            # Arrival time is after all existing projections
            prev_queue_len = queue_projected[-1][1]
            queue_projected.append((arrival_time, prev_queue_len + 1))
            arrival_idx = len(queue_projected) - 1
        else:
            # Update all subsequent queue lengths until departure time
            for i in range(arrival_idx, len(queue_projected)):
                if queue_projected[i][0] < departure_time:
                    time, count = queue_projected[i]
                    queue_projected[i] = (time, count + 1)
        
        # Add or update entry for departure time
        departure_idx = None
        for i, (time, _) in enumerate(queue_projected):
            if time == departure_time:
                departure_idx = i
                break
            if time > departure_time:
                # Insert new entry at this position
                prev_count = queue_projected[i-1][1]
                queue_projected.insert(i, (departure_time, prev_count - 1))
                departure_idx = i
                break
        
        if departure_idx is None:
            # Departure time is after all existing projections
            prev_queue_len = queue_projected[-1][1]
            queue_projected.append((departure_time, prev_queue_len - 1))
            departure_idx = len(queue_projected) - 1
        else:
            # Update all subsequent queue lengths
            for i in range(departure_idx, len(queue_projected)):
                time, count = queue_projected[i]
                queue_projected[i] = (time, count - 1)
        
        # Sort projections by time (just in case)
        queue_projected.sort(key=lambda x: x[0])
        station['queue_projected'] = queue_projected
    
    def assign_station(self, rider_id, current_time):
        """Assign rider to best battery swap station."""
        rider = self.riders[rider_id]
        best_station = None
        min_cost = float('inf')
        best_details = None
        
        # If rider is on a gig, calculate destination coordinates
        dest_lat, dest_lng = None, None
        if rider['status'] == 'on_gig':
            dest_lat, dest_lng = rider['dest_lat'], rider['dest_lng']
        
        for station_id, station in self.stations.items():
            # Check if rider can reach station with safe SOC
            can_reach, distance_to_station, expected_soc = can_reach_with_safe_soc(rider, station)
            if not can_reach:
                continue
            
            # Calculate travel time to station
            travel_time_min = time_to_travel(distance_to_station)
            arrival_time = current_time + timedelta(minutes=travel_time_min)
            
            # Get projected queue length upon arrival
            queue_length = self.get_projected_queue_length(station_id, arrival_time)
            
            # Skip if queue would exceed maximum
            if queue_length >= MAX_QUEUE_LENGTH:
                continue
            
            # Calculate wait time
            wait_time_min = queue_length * SWAP_TIME_MINUTES
            
            # Calculate swap completion time
            swap_start_time = arrival_time + timedelta(minutes=wait_time_min)
            swap_end_time = swap_start_time + timedelta(minutes=SWAP_TIME_MINUTES)
            
            # Calculate detour distance
            detour_distance = distance_to_station
            if rider['status'] == 'on_gig':
                # If on a gig, calculate additional distance compared to direct path
                direct_distance = get_road_distance(rider['lat'], rider['lng'], dest_lat, dest_lng)
                distance_from_station_to_dest = get_road_distance(
                    station['lat'], station['lng'], dest_lat, dest_lng
                )
                total_distance = distance_to_station + distance_from_station_to_dest
                detour_distance = total_distance - direct_distance
            
            # Calculate cost as weighted sum of detour distance and wait time
            distance_weight = 1.0
            wait_time_weight = 0.2  # minutes are weighted less than kilometers
            cost = (distance_weight * detour_distance) + (wait_time_weight * wait_time_min)
            
            if cost < min_cost:
                min_cost = cost
                best_station = station_id
                best_details = {
                    'distance_to_station': distance_to_station,
                    'arrival_time': arrival_time,
                    'wait_time_min': wait_time_min,
                    'swap_start_time': swap_start_time,
                    'swap_end_time': swap_end_time,
                    'detour_distance': detour_distance,
                    'expected_soc': expected_soc
                }
        
        # If no suitable station found, return None
        if best_station is None:
            return None, None
        
        return best_station, best_details
    
    def schedule_swap(self, rider_id, station_id, details):
        """Schedule a battery swap for a rider."""
        rider = self.riders[rider_id]
        station = self.stations[station_id]
        
        # Calculate final location after swap
        rider_lat_after = station['lat']
        rider_lng_after = station['lng']
        
        # If rider is on a gig, they'll continue to their destination
        if rider['status'] == 'on_gig':
            eta_back_lat = rider['dest_lat']
            eta_back_lng = rider['dest_lng']
            
            # Calculate time to reach destination from station
            distance_to_dest = get_road_distance(
                station['lat'], station['lng'],
                rider['dest_lat'], rider['dest_lng']
            )
            time_to_dest = time_to_travel(distance_to_dest)
            eta_back_time = details['swap_end_time'] + timedelta(minutes=time_to_dest)
        else:
            # If idle, they'll stay at the station location after swap
            eta_back_lat = station['lat']
            eta_back_lng = station['lng']
            eta_back_time = details['swap_end_time']
        
        # Update queue projections
        self.update_queue_projection(
            station_id, 
            details['arrival_time'], 
            details['swap_end_time']
        )
        
        # Create swap plan entry
        swap_plan_entry = {
            'rider_id': rider_id,
            'station_id': station_id,
            'depart_ts': self.current_time.isoformat(),
            'arrive_ts': details['arrival_time'].isoformat(),
            'swap_start_ts': details['swap_start_time'].isoformat(),
            'swap_end_ts': details['swap_end_time'].isoformat(),
            'eta_back_lat': eta_back_lat,
            'eta_back_lng': eta_back_lng,
            'eta_back_ts': eta_back_time.isoformat(),
            'initial_soc': rider['soc_pct'],
            'arrival_soc': details['expected_soc'],
            'detour_km': details['detour_distance']
        }
        
        self.swap_plan.append(swap_plan_entry)
        
        # Add swap completion event to event queue
        heapq.heappush(
            self.event_queue,
            (details['swap_end_time'], 'swap_completion', rider_id)
        )
        
        # Update rider status
        rider['status'] = 'en_route_to_swap'
        rider['assigned_station'] = station_id
        rider['swap_details'] = details
    
    def process_swap_completion(self, rider_id, completion_time):
        """Process swap completion event."""
        rider = self.riders[rider_id]
        
        # Update rider SOC and location
        rider['soc_pct'] = 100.0
        station_id = rider['assigned_station']
        station = self.stations[station_id]
        rider['lat'] = station['lat']
        rider['lng'] = station['lng']
        
        # If rider was on a gig, add a gig completion event
        if 'dest_lat' in rider and 'dest_lng' in rider:
            # Calculate remaining distance and time to destination
            remaining_distance = get_road_distance(
                rider['lat'], rider['lng'],
                rider['dest_lat'], rider['dest_lng']
            )
            remaining_time = time_to_travel(remaining_distance)
            
            # Update rider status and gig details
            rider['status'] = 'on_gig'
            rider['km_to_finish'] = remaining_distance
            gig_completion_time = completion_time + timedelta(minutes=remaining_time)
            rider['est_finish_ts'] = gig_completion_time.isoformat()
            
            # Add gig completion event
            if gig_completion_time <= self.end_time:
                heapq.heappush(
                    self.event_queue,
                    (gig_completion_time, 'gig_completion', rider_id)
                )
        else:
            # If rider was idle, they remain idle at the station
            rider['status'] = 'idle'
        
        # Clean up swap-related fields
        rider.pop('assigned_station', None)
        rider.pop('swap_details', None)
    
    def process_gig_completion(self, rider_id, completion_time):
        """Process gig completion event."""
        rider = self.riders[rider_id]
        
        # Update rider location to destination
        rider['lat'] = rider['dest_lat']
        rider['lng'] = rider['dest_lng']
        
        # Update rider status
        rider['status'] = 'idle'
        
        # Clean up gig-related fields
        rider.pop('dest_lat', None)
        rider.pop('dest_lng', None)
        rider.pop('km_to_finish', None)
        rider.pop('est_finish_ts', None)
        
        # Check if rider needs a battery swap
        if rider['soc_pct'] < SWAP_TRIGGER_SOC_PCT:
            self.check_and_schedule_swap(rider_id, completion_time)
    
    def check_and_schedule_swap(self, rider_id, current_time):
        """Check if rider needs a swap and schedule if needed."""
        rider = self.riders[rider_id]
        
        # Skip if rider is already scheduled for a swap
        if rider['status'] in ['en_route_to_swap']:
            return
        
        # Check if SOC is below trigger threshold
        if rider['soc_pct'] < SWAP_TRIGGER_SOC_PCT:
            # Assign best station
            station_id, details = self.assign_station(rider_id, current_time)
            
            if station_id is not None:
                self.schedule_swap(rider_id, station_id, details)
            else:
                # No suitable station found
                # In a real system, we might flag this rider for special handling
                rider['status'] = 'critical_battery'
    
    def simulate(self):
        """Simulate the battery swap system for the given duration."""
        # Identify riders needing immediate swaps
        for rider_id, rider in self.riders.items():
            if rider['soc_pct'] < SWAP_TRIGGER_SOC_PCT:
                self.check_and_schedule_swap(rider_id, self.current_time)
        
        # Simulate minute by minute
        while self.current_time < self.end_time:
            # Process events for the current minute
            while self.event_queue and self.event_queue[0][0] <= self.current_time:
                event_time, event_type, rider_id = heapq.heappop(self.event_queue)
                
                if event_type == 'swap_completion':
                    self.process_swap_completion(rider_id, event_time)
                elif event_type == 'gig_completion':
                    self.process_gig_completion(rider_id, event_time)
            
            # Update simulation time
            self.current_time += timedelta(minutes=1)
            
            # Check for riders needing swaps
            for rider_id, rider in self.riders.items():
                if rider['status'] == 'on_gig':
                    # Update SOC based on distance traveled
                    speed_km_per_min = AVG_SPEED_KM_PER_HOUR / 60
                    km_traveled = speed_km_per_min
                    rider['km_to_finish'] -= km_traveled
                    rider['soc_pct'] -= km_traveled * BATTERY_CONSUMPTION_PCT_PER_KM
                    
                    # Check if rider will need a swap before reaching destination
                    remaining_km = rider['km_to_finish']
                    expected_soc_at_dest = rider['soc_pct'] - (remaining_km * BATTERY_CONSUMPTION_PCT_PER_KM)
                    
                    if expected_soc_at_dest < CRITICAL_SOC_PCT and rider['status'] not in ['en_route_to_swap', 'critical_battery']:
                        self.check_and_schedule_swap(rider_id, self.current_time)
                
                elif rider['status'] == 'idle':
                    # Idle riders with low battery should also be scheduled for swaps
                    if rider['soc_pct'] < SWAP_TRIGGER_SOC_PCT and rider['status'] not in ['en_route_to_swap', 'critical_battery']:
                        self.check_and_schedule_swap(rider_id, self.current_time)
    
    def get_plan_output(self):
        """Get the final swap plan output."""
        # Sort plan by departure time
        sorted_plan = sorted(self.swap_plan, key=lambda x: x['depart_ts'])
        return sorted_plan

# Main function to run the optimizer
def run_battery_swap_optimization(num_riders=100, num_stations=3):
    """Run the battery swap optimization."""
    # Generate mock data
    riders, stations, current_time = generate_mock_data(num_riders, num_stations)
    
    # Create and run optimizer
    optimizer = BatterySwapOptimizer(riders, stations, current_time)
    optimizer.simulate()
    
    # Get optimization plan
    plan_output = optimizer.get_plan_output()
    
    # Calculate statistics
    total_riders_swapped = len(plan_output)
    total_detour_km = sum(entry['detour_km'] for entry in plan_output)
    
    # Print summary
    print(f"Optimization complete!")
    print(f"Total riders scheduled for swaps: {total_riders_swapped}")
    print(f"Total detour kilometers: {total_detour_km:.2f} km")
    
    return {
        'riders': riders,
        'stations': stations,
        'plan_output': plan_output
    }

# Entry point
if __name__ == "__main__":
    # Parse command line arguments if any
    import sys
    num_riders = 100
    num_stations = 3
    if len(sys.argv) > 1:
        num_riders = int(sys.argv[1])
    if len(sys.argv) > 2:
        num_stations = int(sys.argv[2])
    
    # Run optimization
    result = run_battery_swap_optimization(num_riders, num_stations)
    
    # Save output to file
    with open('plan_output.json', 'w') as f:
        json.dump(result['plan_output'], f, indent=2)
    
    print(f"Plan output saved to 'plan_output.json'")