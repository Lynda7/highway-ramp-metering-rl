from sumolib import checkBinary  
import traci
import optparse


def count_vehicles_in_lanes(lanes):
    """
    Count the number of vehicles in each lane based on the vehicle position.
       param lanes: List of lane IDs to check
       return: Dictionary with lane ID as keys and vehicle count as values
       
    """
    vehicle_count_per_lane = dict()
    for lane in lanes:
        vehicle_count_per_lane[lane] = 0
        for vehicle in traci.lane.getLastStepVehicleIDs(lane):
            if traci.vehicle.getLanePosition(vehicle) > 10:  
                vehicle_count_per_lane[lane] += 1
    return vehicle_count_per_lane



def calculate_total_wait_time(lanes):
    """
    Calculate the total waiting time for vehicles in the given lanes.
      param lanes: List of lane IDs to check
      return: Total waiting time for all vehicles in the specified lanes


    """
    total_waiting_time = 0
    for lane in lanes:
        total_waiting_time += traci.lane.getWaitingTime(lane)
    return total_waiting_time

def set_traffic_phase_duration(junction, phase_duration, phase_state):

    """
    Set the traffic light phase duration and its state for a given junction.
    
    :param junction: The ID of the traffic junction
    :param phase_duration: The duration for the current traffic light phase in seconds
    :param phase_state: A string representing the traffic light state (e.g., 'GGGGrrrr' or 'rrrrGGGG')
    """

    traci.trafficlight.setRedYellowGreenState(junction, phase_state)
    traci.trafficlight.setPhaseDuration(junction, phase_duration)



def get_command_line_options(): 
    optParser = optparse.OptionParser()
    optParser.add_option(
        "-m",
        dest='model_name',
        type='string',
        default="model",
        help="Name of the model",
    )
    optParser.add_option(
        "--train",
        action = 'store_true',
        default=False,
        help="Training or testing mode",
    )
    optParser.add_option(
        "-e",
        dest='epochs',
        type='int',
        default=50,
        help="Number of epochs",
    )
    optParser.add_option(
        "-s",
        dest='steps',
        type='int',
        default=500,
        help="Number of steps",
    )
    options, args = optParser.parse_args()
    return options






