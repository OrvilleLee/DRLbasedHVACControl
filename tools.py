from collections import deque
import random
from pprint import pprint
import csv

import csv

class ReplayBuffer():
    def __init__(self, max_length):
        self.deque = deque(maxlen=max_length)

    def add(self, exp:tuple):
        self.deque.append(exp)

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.deque, batch_size))
        return state, action, reward, next_state, done

    def size(self):
        return len(self.deque)


# def HVAC_action_map():
#     HVAC_action_map = []
#     for TZ1 in [0, 1]:
#         for TZ2 in [0, 1]:
#             for TZ3 in [0, 1]:
#                 for TZ4 in [0, 1]:
#                     for TZ5 in [0, 1]:
#                         for TZ6 in [0, 1]:
#                             HVAC_action_map.append([TZ1, TZ2, TZ3, TZ4, TZ5, TZ6])
#     # for TZ1 in [2, 3]:
#     #     for TZ2 in [2, 3]:
#     #         for TZ3 in [2, 3]:
#     #             for TZ4 in [2, 3]:
#     #                 for TZ5 in [2, 3]:
#     #                     for TZ6 in [2, 3]:
#     #                         HVAC_action_map.append([TZ1, TZ2, TZ3, TZ4, TZ5, TZ6])
#
#     return HVAC_action_map

def HVAC_action_map():
    HVAC_action_map = []
    for clg_standard in range(23, 26):
        clg1 = clg_standard
        for clg2 in range(23, 26):
            for clg3 in range(23, 26):
                for clg4 in range(23, 26):
                    for clg5 in range(23, 26):
                        for clg6 in range(23, 26):
                            HVAC_action_map.append([clg1, clg2, clg3, clg4, clg5, clg6])
    return HVAC_action_map

def HVAC_setting_value(on_of):
    if on_of == 0:
        # temp_setting = [22, 28]
        temp_setting = [22, 28]
    elif on_of == 1:
        # temp_setting = [24, 26]
        temp_setting = [24, 26]
    # elif on_of == 2:
    #     temp_setting = [22, 24]
    # else:
    #     temp_setting = [22, 26]
    return temp_setting

def save_to_csv(DATA):
    csvdir = './data.csv'
    header = [
        'Site Outdoor Air Drybulb Temperature',
        'Zone Windows Total Heat Gain Energy',
        'Zone Air Relative Humidity',
        'Zone Mechanical Ventilation Mass Flow',
        'Zone Thermostat Cooling Setpoint Temperature',
        'Electricity_Zone_1'
    ]
    with open(csvdir, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        data = list(zip(DATA.Site_Outdoor_Air_Drybulb_Temperature,
                        DATA.Zone_Windows_Total_Heat_Gain_Energy_1,
                        DATA.Zone_Air_Relative_Humidity_1,
                        DATA.Zone_Mechanical_Ventilation_Mass_1,
                        DATA.Zone_Thermostat_Cooling_Setpoint_Temperature_1,
                        DATA.Electricity_Zone_1
                        ))
        for row in data:
            writer.writerow(row)

if __name__ == '__main__':
    map = HVAC_action_map()
    pprint(map)
    print(len(map))
    # save_to_csv()
