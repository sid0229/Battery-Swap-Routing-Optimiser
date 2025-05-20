Battery-Swap Routing Optimizer
This project provides a solution for optimizing battery swap schedules for electric vehicle riders in a 60-minute planning horizon. The system ensures that:

Riders' battery state-of-charge (SOC) never drops below 10% before a swap
Total detour kilometers to swap stations are minimized
Station queues never exceed 5 riders at any moment

Project Structure
.
├── README.md                     # This file
├── main.py                       # Main script to run the optimization
├── battery_swap_code.py          # Core optimization algorithm
├── visualization.py              # Visualization utilities
├── solution.md                   # Solution explanation and methodology
├── plan_output.json              # Generated swap plan (output)
├── riders_data.json              # Generated rider data (output)
├── stations_data.json            # Generated station data (output)

Running the Solution
Prerequisites

Python 3.6+
Required packages: numpy, pandas, folium

Installation
bashpip install numpy pandas folium matplotlib
Running the Optimizer
Simple usage:
bashpython main.py
With custom parameters:
bashpython main.py [num_riders] [num_stations]
Example:
bashpython main.py 150 5
Input and Output
Input
The system automatically generates mock data for riders and stations. The input data includes:

Riders: ID, location, battery status, current activity
Stations: ID, location, current queue

Output
The system produces the following outputs:

plan_output.json - The main output file containing the swap schedule
riders_data.json - Data about all riders in the system
stations_data.json - Data about all swap stations




Output Format
The plan_output.json file contains an array of scheduled swaps with the following structure:
json{
  "rider_id": "R042",                      // Rider identifier
  "station_id": "S_A",                     // Assigned station
  "depart_ts": "2025-05-20T10:00:00",      // When rider should depart
  "arrive_ts": "2025-05-20T10:07:30",      // Expected arrival at station
  "swap_start_ts": "2025-05-20T10:07:30",  // When swap begins (after queue)
  "swap_end_ts": "2025-05-20T10:11:30",    // When swap completes
  "eta_back_lat": 18.5234,                 // Destination latitude
  "eta_back_lng": 73.8562,                 // Destination longitude
  "eta_back_ts": "2025-05-20T10:19:45",    // ETA at destination
  "initial_soc": 23.5,                     // SOC at departure
  "arrival_soc": 18.2,                     // SOC upon arrival at station
  "detour_km": 1.32                        // Additional distance for swap
}
Methodology
For a detailed explanation of the solution methodology, please refer to solution.md.
Customization
You can adjust the following parameters in battery_swap_code.py:

BATTERY_CONSUMPTION_PCT_PER_KM - Battery consumption rate
SWAP_TIME_MINUTES - Time needed for battery swap
MAX_QUEUE_LENGTH - Maximum allowed queue length
CRITICAL_SOC_PCT - Minimum allowed SOC
SWAP_TRIGGER_SOC_PCT - SOC threshold to trigger swap planning
AVG_SPEED_KM_PER_HOUR - Average travel speed
