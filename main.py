#!/usr/bin/env python3
"""
Battery-Swap Routing Optimizer
------------------------------
This script runs the battery-swap routing optimization for electric vehicle riders.
"""

import json
import os
import sys
from datetime import datetime

# Import from our battery swap module
from battery_swap_code import run_battery_swap_optimization

def datetime_converter(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def main():
    """Main function to run the battery swap routing optimizer."""
    print("Battery-Swap Routing Optimizer")
    print("------------------------------")
    
    # Parse command line arguments
    num_riders = 100
    num_stations = 3
    
    if len(sys.argv) > 1:
        try:
            num_riders = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of riders: {sys.argv[1]}. Using default: {num_riders}")
    
    if len(sys.argv) > 2:
        try:
            num_stations = int(sys.argv[2])
        except ValueError:
            print(f"Invalid number of stations: {sys.argv[2]}. Using default: {num_stations}")
    
    print(f"Running optimization with {num_riders} riders and {num_stations} stations...")
    
    # Run the optimization
    start_time = datetime.now()
    result = run_battery_swap_optimization(num_riders, num_stations)
    end_time = datetime.now()
    
    # Extract results
    riders = result['riders']
    stations = result['stations']
    plan_output = result['plan_output']
    
    # Save full output
    with open('plan_output.json', 'w') as f:
        json.dump(plan_output, f, indent=2, default=datetime_converter)
    
    # Save riders data
    with open('riders_data.json', 'w') as f:
        json.dump(riders, f, indent=2, default=datetime_converter)
    
    # Save stations data
    with open('stations_data.json', 'w') as f:
        json.dump(stations, f, indent=2, default=datetime_converter)
    
    # Print summary statistics
    total_riders_swapped = len(plan_output)
    if total_riders_swapped > 0:
        total_detour_km = sum(entry['detour_km'] for entry in plan_output)
        avg_detour_km = total_detour_km / total_riders_swapped
        avg_wait_min = sum((datetime.fromisoformat(entry['swap_start_ts']) - 
                          datetime.fromisoformat(entry['arrive_ts'])).total_seconds() / 60 
                         for entry in plan_output) / total_riders_swapped
    else:
        total_detour_km = 0
        avg_detour_km = 0
        avg_wait_min = 0
    
    print("\nOptimization Results:")
    print(f"Total riders: {num_riders}")
    print(f"Riders scheduled for swaps: {total_riders_swapped} ({total_riders_swapped/num_riders*100:.1f}%)")
    print(f"Total detour kilometers: {total_detour_km:.2f} km")
    print(f"Average detour per rider: {avg_detour_km:.2f} km")
    print(f"Average wait time at stations: {avg_wait_min:.2f} minutes")
    print(f"Execution time: {(end_time - start_time).total_seconds():.2f} seconds")
    
    
    print("\nOptimization complete! Results saved to 'plan_output.json'")
    print("Run with different parameters: python main.py [num_riders] [num_stations]")

if __name__ == "__main__":
    main()
