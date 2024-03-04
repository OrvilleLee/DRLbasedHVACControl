import os.path
import sys
import shutil
import datetime
import time

import random
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('agg')
# matplotlib.use('TkAgg')
import matplotlib.animation as animation
import numpy as np
import torch.cuda
from tqdm import tqdm
import copy

from osm2idf import osm2idf
# from callback_function import callback_function
from data_center import Data_Center
from plot import Drawing
from agent import DQN
from tools import HVAC_setting_value, ReplayBuffer, save_to_csv

# Eplus_Dir = "D:/EnergyPlusV23-1-0"
# sys.path.insert(0, Eplus_Dir)
import pyenergyplus
from pyenergyplus.api import EnergyPlusAPI


def update_plot(draw):
    for i in DATA.x:
        if i.hour == 0 and i.minute == 0:
            draw.ax.axvline(i, linewidth=10, color='#ebfced', alpha=0.7)
            continue

    draw.set_ax_view()
    # draw.ax.set_xlim(DATA.x[-432], DATA.x[-1])
    # draw.ax.set_ylim(-25, 40)
    # line1, = ax.plot(DATA.x, DATA.Zone_Air_Relative_Humidity_1, label="Zone Temperature")

    # draw.ax.plot(DATA.x, DATA.Environmental_Impact_Total_CO2_Emissions_Carbon_Equivalent_Mass,
    #      label="CO2 mass", color='black', linewidth=1)
    draw.ax.plot(DATA.x, DATA.Site_Outdoor_Air_Drybulb_Temperature,
                 label="Outdoor Temperature", color='#FFD700', linewidth=1)

    draw.ax.plot(DATA.x, DATA.Zone_Air_Temperature_1,
                 label="Zone 1 Temperature", color='#48D1CC', linewidth=1)
    draw.ax.plot(DATA.x, DATA.Zone_Air_Temperature_2,
                 label="Zone 2 Temperature", color='#7FFFAA', linewidth=1)
    draw.ax.plot(DATA.x, DATA.Zone_Air_Temperature_3,
                 label="Zone 3 Temperature", color='#7B68EE', linewidth=1)
    draw.ax.plot(DATA.x, DATA.Zone_Air_Temperature_4,
                 label="Zone 4 Temperature", color='#FFD700', linewidth=1)
    draw.ax.plot(DATA.x, DATA.Zone_Air_Temperature_5,
                 label="Zone 5 Temperature", color='#20B2AA', linewidth=1)
    draw.ax.plot(DATA.x, DATA.Zone_Mean_Temperature,
                 label="Zone Mean Temperature", color='#20B2BB', linewidth=5, alpha=0.5)


    draw.ax.plot(DATA.x, DATA.Zone_Thermostat_Heating_Setpoint_Temperature_1,
                 label="DATA.Zone_Thermostat_Heating_Setpoint_Temperature_1", color='red', linewidth=0.5,
                 linestyle='-.')
    draw.ax.plot(DATA.x, DATA.Zone_Thermostat_Cooling_Setpoint_Temperature_1,
                 label="DATA.Zone_Thermostat_Cooling_Setpoint_Temperature_1", color='cyan', linewidth=0.5,
                 linestyle='-.')

    if DATA.train_switch:
        draw.ax.plot(DATA.x, DATA.reward,
                     label="Reward", color='grey', linewidth=1)



    # draw.ax2.plot(DATA.x, DATA.Electricity_HVAC,
    #               label="Electricity_HVAC", color='red', linewidth=0.5)
    draw.ax2.plot(DATA.x, DATA.Electricity_Zone_1,
                  label="Electricity_Zone_1", color='#FF6347', linewidth=0.5)

    if draw.is_ion:
        plt.pause(0.01)
    else:
        plt.show()
    # fig.canvas.draw()
    # plt.show(block=True)


def callback_function(EPstate):
    api = EnergyPlusAPI()
    if not DATA.is_handle:
        if not api.exchange.api_data_fully_ready(EPstate):
            # print(('\033[33mStill waiting for api\033[0m'))
            return
        else:
            DATA.is_handle = True
            print('\033[32mApi for data exchange is fully ready\033[0m')

            '''Define variable handles'''
            DATA.handle_Environmental_Impact_Total_CO2_Emissions_Carbon_Equivalent_Mass = api.exchange.get_variable_handle(EPstate, 'Environmental Impact Total CO2 Emissions Carbon Equivalent Mass', 'Thermal Zone 1')
            # This is to get handles for ENVIRONMENT
            DATA.handle_Site_Outdoor_Air_Drybulb_Temperature = api.exchange.get_variable_handle(EPstate, u"Site Outdoor Air Drybulb Temperature", u"ENVIRONMENT")
            DATA.handle_Site_Wind_Speed                      = api.exchange.get_variable_handle(EPstate, u"Site Wind Speed", u"ENVIRONMENT")
            DATA.handle_Site_Wind_Direction                  = api.exchange.get_variable_handle(EPstate, u"Site Wind Direction", u"ENVIRONMENT")
            DATA.handle_Site_Solar_Azimuth_Angle             = api.exchange.get_variable_handle(EPstate, u"Site Solar Azimuth Angle", u"ENVIRONMENT")
            DATA.handle_Site_Solar_Altitude_Angle            = api.exchange.get_variable_handle(EPstate, u"Site Solar Altitude Angle", u"ENVIRONMENT")
            DATA.handle_Site_Solar_Hour_Angle                = api.exchange.get_variable_handle(EPstate, u"Site Solar Hour Angle", u"ENVIRONMENT")

            # This is to get handles for Thermal Zone 1
            DATA.handle_Zone_Air_Relative_Humidity_1                   = api.exchange.get_variable_handle(EPstate, 'Zone Air Relative Humidity', 'Thermal Zone 1')
            DATA.handle_Zone_Windows_Total_Heat_Gain_Energy_1          = api.exchange.get_variable_handle(EPstate, 'Zone Windows Total Heat Gain Energy', 'Thermal Zone 1')
            DATA.handle_Zone_Infiltration_Mass_1                       = api.exchange.get_variable_handle(EPstate, 'Zone Infiltration Mass', 'Thermal Zone 1')
            DATA.handle_Zone_Mechanical_Ventilation_Mass_1             = api.exchange.get_variable_handle(EPstate, 'Zone Mechanical Ventilation Mass', 'Thermal Zone 1')
            DATA.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_1   = api.exchange.get_variable_handle(EPstate, 'Zone Mechanical Ventilation Mass Flow Rate', 'Thermal Zone 1')
            DATA.handle_Zone_Air_Temperature_1                         = api.exchange.get_variable_handle(EPstate, 'Zone Air Temperature', 'Thermal Zone 1')
            DATA.handle_Zone_Mean_Radiant_Temperature_1                = api.exchange.get_variable_handle(EPstate, 'Zone Mean Radiant Temperature', 'Thermal Zone 1')
            DATA.handle_Zone_Thermostat_Heating_Setpoint_Temperature_1 = api.exchange.get_variable_handle(EPstate, 'Zone Thermostat Heating Setpoint Temperature', 'Thermal Zone 1')
            DATA.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_1 = api.exchange.get_variable_handle(EPstate, 'Zone Thermostat Cooling Setpoint Temperature', 'Thermal Zone 1')

            # This is to get handles for Thermal Zone 2
            DATA.handle_Zone_Air_Relative_Humidity_2                   = api.exchange.get_variable_handle(EPstate, 'Zone Air Relative Humidity', 'Thermal Zone 2')
            DATA.handle_Zone_Windows_Total_Heat_Gain_Energy_2          = api.exchange.get_variable_handle(EPstate, 'Zone Windows Total Heat Gain Energy', 'Thermal Zone 2')
            DATA.handle_Zone_Infiltration_Mass_2                       = api.exchange.get_variable_handle(EPstate, 'Zone Infiltration Mass', 'Thermal Zone 2')
            DATA.handle_Zone_Mechanical_Ventilation_Mass_2             = api.exchange.get_variable_handle(EPstate, 'Zone Mechanical Ventilation Mass', 'Thermal Zone 2')
            DATA.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_2   = api.exchange.get_variable_handle(EPstate, 'Zone Mechanical Ventilation Mass Flow Rate', 'Thermal Zone 2')
            DATA.handle_Zone_Air_Temperature_2                         = api.exchange.get_variable_handle(EPstate, 'Zone Air Temperature', 'Thermal Zone 2')
            DATA.handle_Zone_Mean_Radiant_Temperature_2                = api.exchange.get_variable_handle(EPstate, 'Zone Mean Radiant Temperature', 'Thermal Zone 2')
            DATA.handle_Zone_Thermostat_Heating_Setpoint_Temperature_2 = api.exchange.get_variable_handle(EPstate, 'Zone Thermostat Heating Setpoint Temperature', 'Thermal Zone 2')
            DATA.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_2 = api.exchange.get_variable_handle(EPstate, 'Zone Thermostat Cooling Setpoint Temperature', 'Thermal Zone 2')

            # This is to get handles for Thermal Zone 3
            DATA.handle_Zone_Air_Relative_Humidity_3                   = api.exchange.get_variable_handle(EPstate, 'Zone Air Relative Humidity', 'Thermal Zone 3')
            DATA.handle_Zone_Windows_Total_Heat_Gain_Energy_3          = api.exchange.get_variable_handle(EPstate, 'Zone Windows Total Heat Gain Energy', 'Thermal Zone 3')
            DATA.handle_Zone_Infiltration_Mass_3                       = api.exchange.get_variable_handle(EPstate, 'Zone Infiltration Mass', 'Thermal Zone 3')
            DATA.handle_Zone_Mechanical_Ventilation_Mass_3             = api.exchange.get_variable_handle(EPstate, 'Zone Mechanical Ventilation Mass', 'Thermal Zone 3')
            DATA.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_3   = api.exchange.get_variable_handle(EPstate, 'Zone Mechanical Ventilation Mass Flow Rate', 'Thermal Zone 3')
            DATA.handle_Zone_Air_Temperature_3                         = api.exchange.get_variable_handle(EPstate, 'Zone Air Temperature', 'Thermal Zone 3')
            DATA.handle_Zone_Mean_Radiant_Temperature_3                = api.exchange.get_variable_handle(EPstate, 'Zone Mean Radiant Temperature', 'Thermal Zone 3')
            DATA.handle_Zone_Thermostat_Heating_Setpoint_Temperature_3 = api.exchange.get_variable_handle(EPstate, 'Zone Thermostat Heating Setpoint Temperature', 'Thermal Zone 3')
            DATA.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_3 = api.exchange.get_variable_handle(EPstate, 'Zone Thermostat Cooling Setpoint Temperature', 'Thermal Zone 3')

            # This is to get handles for Thermal Zone 4
            DATA.handle_Zone_Air_Relative_Humidity_4                   = api.exchange.get_variable_handle(EPstate, 'Zone Air Relative Humidity', 'Thermal Zone 4')
            DATA.handle_Zone_Windows_Total_Heat_Gain_Energy_4          = api.exchange.get_variable_handle(EPstate, 'Zone Windows Total Heat Gain Energy', 'Thermal Zone 4')
            DATA.handle_Zone_Infiltration_Mass_4                       = api.exchange.get_variable_handle(EPstate, 'Zone Infiltration Mass', 'Thermal Zone 4')
            DATA.handle_Zone_Mechanical_Ventilation_Mass_4             = api.exchange.get_variable_handle(EPstate, 'Zone Mechanical Ventilation Mass', 'Thermal Zone 4')
            DATA.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_4   = api.exchange.get_variable_handle(EPstate, 'Zone Mechanical Ventilation Mass Flow Rate', 'Thermal Zone 4')
            DATA.handle_Zone_Air_Temperature_4                         = api.exchange.get_variable_handle(EPstate, 'Zone Air Temperature', 'Thermal Zone 4')
            DATA.handle_Zone_Mean_Radiant_Temperature_4                = api.exchange.get_variable_handle(EPstate, 'Zone Mean Radiant Temperature', 'Thermal Zone 4')
            DATA.handle_Zone_Thermostat_Heating_Setpoint_Temperature_4 = api.exchange.get_variable_handle(EPstate, 'Zone Thermostat Heating Setpoint Temperature', 'Thermal Zone 4')
            DATA.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_4 = api.exchange.get_variable_handle(EPstate, 'Zone Thermostat Cooling Setpoint Temperature', 'Thermal Zone 4')

            # This is to get handles for Thermal Zone 5
            DATA.handle_Zone_Air_Relative_Humidity_5                   = api.exchange.get_variable_handle(EPstate, 'Zone Air Relative Humidity', 'Thermal Zone 5')
            DATA.handle_Zone_Windows_Total_Heat_Gain_Energy_5          = api.exchange.get_variable_handle(EPstate, 'Zone Windows Total Heat Gain Energy', 'Thermal Zone 5')
            DATA.handle_Zone_Infiltration_Mass_5                       = api.exchange.get_variable_handle(EPstate, 'Zone Infiltration Mass', 'Thermal Zone 5')
            DATA.handle_Zone_Mechanical_Ventilation_Mass_5             = api.exchange.get_variable_handle(EPstate, 'Zone Mechanical Ventilation Mass', 'Thermal Zone 5')
            DATA.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_5   = api.exchange.get_variable_handle(EPstate, 'Zone Mechanical Ventilation Mass Flow Rate', 'Thermal Zone 5')
            DATA.handle_Zone_Air_Temperature_5                         = api.exchange.get_variable_handle(EPstate, 'Zone Air Temperature', 'Thermal Zone 5')
            DATA.handle_Zone_Mean_Radiant_Temperature_5                = api.exchange.get_variable_handle(EPstate, 'Zone Mean Radiant Temperature', 'Thermal Zone 5')
            DATA.handle_Zone_Thermostat_Heating_Setpoint_Temperature_5 = api.exchange.get_variable_handle(EPstate, 'Zone Thermostat Heating Setpoint Temperature', 'Thermal Zone 5')
            DATA.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_5 = api.exchange.get_variable_handle(EPstate, 'Zone Thermostat Cooling Setpoint Temperature', 'Thermal Zone 5')

            # This is to get METER handles for energy consumption
            DATA.handle_Electricity_Facility = api.exchange.get_meter_handle(EPstate, 'Electricity:Facility')
            DATA.handle_Electricity_HVAC     = api.exchange.get_meter_handle(EPstate, 'Electricity:HVAC')
            DATA.handle_Heating_Electricity  = api.exchange.get_meter_handle(EPstate, 'Heating:Electricity')
            DATA.handle_Cooling_Electricity  = api.exchange.get_meter_handle(EPstate, 'Cooling:Electricity')

            DATA.handle_Electricity_Zone_1 = api.exchange.get_meter_handle(EPstate, 'Electricity:Zone:THERMAL ZONE 1')




            '''actuator'''
            # This is to get handles for actuators for each zone
            DATA.handle_Heating_Setpoint_1 = api.exchange.get_actuator_handle(EPstate, 'Zone Temperature Control',  'Heating Setpoint', 'Thermal Zone 1')
            DATA.handle_Cooling_Setpoint_1 = api.exchange.get_actuator_handle(EPstate, 'Zone Temperature Control', 'Cooling Setpoint', 'Thermal Zone 1')

            DATA.handle_Heating_Setpoint_2 = api.exchange.get_actuator_handle(EPstate, 'Zone Temperature Control', 'Heating Setpoint', 'Thermal Zone 2')
            DATA.handle_Cooling_Setpoint_2 = api.exchange.get_actuator_handle(EPstate, 'Zone Temperature Control', 'Cooling Setpoint', 'Thermal Zone 2')

            DATA.handle_Heating_Setpoint_3 = api.exchange.get_actuator_handle(EPstate, 'Zone Temperature Control', 'Heating Setpoint', 'Thermal Zone 3')
            DATA.handle_Cooling_Setpoint_3 = api.exchange.get_actuator_handle(EPstate, 'Zone Temperature Control', 'Cooling Setpoint', 'Thermal Zone 3')

            DATA.handle_Heating_Setpoint_4 = api.exchange.get_actuator_handle(EPstate, 'Zone Temperature Control', 'Heating Setpoint', 'Thermal Zone 4')
            DATA.handle_Cooling_Setpoint_4 = api.exchange.get_actuator_handle(EPstate, 'Zone Temperature Control', 'Cooling Setpoint', 'Thermal Zone 4')

            DATA.handle_Heating_Setpoint_5 = api.exchange.get_actuator_handle(EPstate, 'Zone Temperature Control', 'Heating Setpoint', 'Thermal Zone 5')
            DATA.handle_Cooling_Setpoint_5 = api.exchange.get_actuator_handle(EPstate, 'Zone Temperature Control', 'Cooling Setpoint', 'Thermal Zone 5')

            # print(DATA.get_handles_state())
            if not DATA.get_handles_state():
                print('\033[31mInvalid handles, check spelling and sensor/actuator availability\033[0m')
                sys.exit(1)
    if api.exchange.warmup_flag(EPstate):
        return

    # print(f'区域1温度为：{api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Air_Temperature_1)}')
    '''
    Retrieve data using variable handles 
    '''
    DATA.Environmental_Impact_Total_CO2_Emissions_Carbon_Equivalent_Mass.append(api.exchange.get_variable_value(EPstate, DATA.handle_Environmental_Impact_Total_CO2_Emissions_Carbon_Equivalent_Mass))
    # ENVIRONMENT
    DATA.Site_Outdoor_Air_Drybulb_Temperature.append(api.exchange.get_variable_value(EPstate, DATA.handle_Site_Outdoor_Air_Drybulb_Temperature))
    DATA.Site_Wind_Speed                     .append(api.exchange.get_variable_value(EPstate, DATA.handle_Site_Wind_Speed))
    DATA.Site_Wind_Direction                 .append(api.exchange.get_variable_value(EPstate, DATA.handle_Site_Wind_Direction))
    DATA.Site_Solar_Azimuth_Angle            .append(api.exchange.get_variable_value(EPstate, DATA.handle_Site_Solar_Azimuth_Angle))
    DATA.Site_Solar_Altitude_Angle           .append(api.exchange.get_variable_value(EPstate, DATA.handle_Site_Solar_Altitude_Angle))
    DATA.Site_Solar_Hour_Angle               .append(api.exchange.get_variable_value(EPstate, DATA.handle_Site_Solar_Hour_Angle))

    # Thermal Zone 1
    DATA.Zone_Air_Relative_Humidity_1                  .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Air_Relative_Humidity_1))
    DATA.Zone_Windows_Total_Heat_Gain_Energy_1         .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Windows_Total_Heat_Gain_Energy_1))
    DATA.Zone_Infiltration_Mass_1                      .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Infiltration_Mass_1))
    DATA.Zone_Mechanical_Ventilation_Mass_1            .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mechanical_Ventilation_Mass_1))
    DATA.Zone_Mechanical_Ventilation_Mass_Flow_Rate_1  .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_1))
    DATA.Zone_Air_Temperature_1                        .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Air_Temperature_1))
    DATA.Zone_Mean_Radiant_Temperature_1               .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mean_Radiant_Temperature_1))
    DATA.Zone_Thermostat_Heating_Setpoint_Temperature_1.append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Thermostat_Heating_Setpoint_Temperature_1))
    DATA.Zone_Thermostat_Cooling_Setpoint_Temperature_1.append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_1))


    # Thermal Zone 2
    DATA.Zone_Air_Relative_Humidity_2                  .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Air_Relative_Humidity_2))
    DATA.Zone_Windows_Total_Heat_Gain_Energy_2         .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Windows_Total_Heat_Gain_Energy_2))
    DATA.Zone_Infiltration_Mass_2                      .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Infiltration_Mass_2))
    DATA.Zone_Mechanical_Ventilation_Mass_2            .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mechanical_Ventilation_Mass_2))
    DATA.Zone_Mechanical_Ventilation_Mass_Flow_Rate_2  .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_2))
    DATA.Zone_Air_Temperature_2                        .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Air_Temperature_2))
    DATA.Zone_Mean_Radiant_Temperature_2               .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mean_Radiant_Temperature_2))
    DATA.Zone_Thermostat_Heating_Setpoint_Temperature_2.append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Thermostat_Heating_Setpoint_Temperature_2))
    DATA.Zone_Thermostat_Cooling_Setpoint_Temperature_2.append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_2))

    # Thermal Zone 3
    DATA.Zone_Air_Relative_Humidity_3                  .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Air_Relative_Humidity_3))
    DATA.Zone_Windows_Total_Heat_Gain_Energy_3         .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Windows_Total_Heat_Gain_Energy_3))
    DATA.Zone_Infiltration_Mass_3                      .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Infiltration_Mass_3))
    DATA.Zone_Mechanical_Ventilation_Mass_3            .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mechanical_Ventilation_Mass_3))
    DATA.Zone_Mechanical_Ventilation_Mass_Flow_Rate_3  .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_3))
    DATA.Zone_Air_Temperature_3                        .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Air_Temperature_3))
    DATA.Zone_Mean_Radiant_Temperature_3               .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mean_Radiant_Temperature_3))
    DATA.Zone_Thermostat_Heating_Setpoint_Temperature_3.append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Thermostat_Heating_Setpoint_Temperature_3))
    DATA.Zone_Thermostat_Cooling_Setpoint_Temperature_3.append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_3))

    # Thermal Zone 4
    DATA.Zone_Air_Relative_Humidity_4                  .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Air_Relative_Humidity_4))
    DATA.Zone_Windows_Total_Heat_Gain_Energy_4         .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Windows_Total_Heat_Gain_Energy_4))
    DATA.Zone_Infiltration_Mass_4                      .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Infiltration_Mass_4))
    DATA.Zone_Mechanical_Ventilation_Mass_4            .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mechanical_Ventilation_Mass_4))
    DATA.Zone_Mechanical_Ventilation_Mass_Flow_Rate_4  .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_4))
    DATA.Zone_Air_Temperature_4                        .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Air_Temperature_4))
    DATA.Zone_Mean_Radiant_Temperature_4               .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mean_Radiant_Temperature_4))
    DATA.Zone_Thermostat_Heating_Setpoint_Temperature_4.append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Thermostat_Heating_Setpoint_Temperature_4))
    DATA.Zone_Thermostat_Cooling_Setpoint_Temperature_4.append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_4))

    # Thermal Zone 5
    DATA.Zone_Air_Relative_Humidity_5                  .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Air_Relative_Humidity_5))
    DATA.Zone_Windows_Total_Heat_Gain_Energy_5         .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Windows_Total_Heat_Gain_Energy_5))
    DATA.Zone_Infiltration_Mass_5                      .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Infiltration_Mass_5))
    DATA.Zone_Mechanical_Ventilation_Mass_5            .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mechanical_Ventilation_Mass_5))
    DATA.Zone_Mechanical_Ventilation_Mass_Flow_Rate_5  .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_5))
    DATA.Zone_Air_Temperature_5                        .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Air_Temperature_5))
    DATA.Zone_Mean_Radiant_Temperature_5               .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mean_Radiant_Temperature_5))
    DATA.Zone_Thermostat_Heating_Setpoint_Temperature_5.append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Thermostat_Heating_Setpoint_Temperature_5))
    DATA.Zone_Thermostat_Cooling_Setpoint_Temperature_5.append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_5))

    DATA.Zone_Mean_Temperature.append((DATA.Zone_Air_Temperature_1[-1] +
                                      DATA.Zone_Air_Temperature_2[-1] +
                                      DATA.Zone_Air_Temperature_3[-1] +
                                      DATA.Zone_Air_Temperature_4[-1] +
                                      DATA.Zone_Air_Temperature_5[-1])/5)

    # Energy
    DATA.Electricity_Facility.append(api.exchange.get_meter_value(EPstate, DATA.handle_Electricity_Facility))
    DATA.Electricity_HVAC    .append(api.exchange.get_meter_value(EPstate, DATA.handle_Electricity_HVAC))
    DATA.Heating_Electricity .append(api.exchange.get_meter_value(EPstate, DATA.handle_Heating_Electricity))
    DATA.Cooling_Electricity .append(api.exchange.get_meter_value(EPstate, DATA.handle_Cooling_Electricity))

    DATA.Electricity_Zone_1  .append(api.exchange.get_meter_value(EPstate, DATA.handle_Electricity_Zone_1))

    # Time
    # T_year = api.exchange.year(EPstate)
    T_year             = 2023
    T_month            = api.exchange.month(EPstate)
    T_day              = api.exchange.day_of_month(EPstate)
    T_hour             = api.exchange.hour(EPstate)
    T_minute           = api.exchange.minutes(EPstate)
    T_current_time     = api.exchange.current_time(EPstate)
    T_actual_date_time = api.exchange.actual_date_time(EPstate)
    T_actual_time      = api.exchange.actual_time(EPstate)
    T_time_step        = api.exchange.zone_time_step_number(EPstate)

    DATA.T_years            .append(T_year)
    DATA.T_months           .append(T_month)
    DATA.T_days             .append(T_day)
    DATA.T_hours            .append(T_hour)
    DATA.T_minutes          .append(T_minute)
    DATA.T_current_times    .append(T_current_time)
    DATA.T_actual_date_times.append(T_actual_date_time)
    DATA.T_actual_times     .append(T_actual_time)
    DATA.T_time_steps       .append(T_time_step)

    timedelta = datetime.timedelta()
    if T_minute >= 60:
        T_minute = 59
        timedelta += datetime.timedelta(minutes=1)

    dt = datetime.datetime(
        year=T_year,
        month=T_month,
        day=T_day,
        hour=T_hour,
        minute=T_minute
    )
    dt += timedelta
    DATA.x.append(dt)


    # if T_hour == 8:
    #     api.exchange.set_actuator_value(EPstate, DATA.handle_Heating_Setpoint_1, 28)
    #     print(f'在调整setpoint以后区域1温度为：{api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Air_Temperature_1)}')
    # if T_hour == 12:
    #     api.exchange.set_actuator_value(EPstate, DATA.handle_Heating_Setpoint_1, 24)
    #
    # if T_hour == 16:
    #     api.exchange.set_actuator_value(EPstate, DATA.handle_Heating_Setpoint_1, 26)
    #
    # # if count == draw.x_view:
    # #     api.exchange.set_actuator_value(EPstate, DATA.handle_Heating_Setpoint_1, 20)





    '''DQN training'''
    if DATA.train_switch:

        ''' current_state and also the 'next_state' for the last episode'''
        s1 = DATA.Zone_Air_Temperature_1[-1]
        s2 = DATA.Zone_Air_Temperature_2[-1]
        s3 = DATA.Zone_Air_Temperature_2[-1]
        s4 = DATA.Zone_Air_Temperature_2[-1]
        s5 = DATA.Zone_Air_Temperature_2[-1]
        s6 = DATA.Site_Outdoor_Air_Drybulb_Temperature[-1]
        s7 = DATA.Electricity_HVAC[-1]
        s8 = DATA.T_months[-1]
        s9 = DATA.T_days[-1]
        s10 = DATA.T_hours[-1]
        state = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]
        DATA.state.append(state)

        ''' reward for last episode'''
        #  Energy reward
        factor_E = 1e-5
        reward_E = - factor_E * s7
        Temp_list = [s1, s2, s3, s4, s5]
        Temp_mean = sum(Temp_list)/len(Temp_list)
        #  Temperature reward
        factor_T = 0.5
        if T_hour >= 20 or T_hour <= 8:
            positive = 0
        else:
            positive = 1
        reward_T_list = []
        for T in Temp_list:
            if 22 < T < 24:
                reward_T_list.append(positive * T)
            else:
                reward_T_list.append(-abs(T - 23) ** 2 * factor_T * positive)
        reward_T = np.mean(reward_T_list)

        #  wear-out reward
        if DATA.wear_out_flag:
            shift_signal = (np.array(random.choice(DATA.HVAC_action_map) if len(DATA.action) < 2 else DATA.HVAC_action_map[DATA.action[-2]])
                            ^ np.array(random.choice(DATA.HVAC_action_map) if len(DATA.action) < 1 else DATA.HVAC_action_map[DATA.action[-1]]))
            n_shift_signal = np.sum(shift_signal == 1)
            factor_S = 0.1
            reward_S = - factor_S * n_shift_signal
        # if DATA.wear_out_flag:
            reward = reward_E + reward_T + reward_S
        else:
            reward = reward_E + reward_T
        DATA.reward.append(reward)
        DATA.reward_random_memory.append(reward)

        #  Done
        done = True
        DATA.done.append(done)


        if DATA.count == 0:
            ReplayBuffer.add((DATA.state[-1],
                              0,
                              DATA.reward[-1],
                              DATA.state[-1],
                              DATA.done[-1]))

        ReplayBuffer.add((DATA.state[-1] if len(DATA.state) < 2 else DATA.state[-2],
                          0 if len(DATA.action) < 1 else DATA.action[-1],
                          DATA.reward[-1],
                          DATA.state[-1],
                          DATA.done[-1]))
        if ReplayBuffer.size() > DATA.minimal_episode:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = ReplayBuffer.sample(32)
            transition = {
                'states': batch_state,
                'actions': batch_action,
                'rewards': batch_reward,
                'next_states': batch_next_state,
                'dones': batch_done
                }
            episode_loss = EPagent.update(transition)
            DATA.loss.append(episode_loss)
            DATA.loss_random_memory.append(episode_loss)



        #  take action
        action = EPagent.take_action(state)
        DATA.action.append(action)
        action_list = DATA.HVAC_action_map[action]
        #  put action into EP for next simulation step
        api.exchange.set_actuator_value(EPstate, DATA.handle_Heating_Setpoint_1, HVAC_setting_value(action_list[0])[0])
        api.exchange.set_actuator_value(EPstate, DATA.handle_Cooling_Setpoint_1, HVAC_setting_value(action_list[0])[1])
        api.exchange.set_actuator_value(EPstate, DATA.handle_Heating_Setpoint_2, HVAC_setting_value(action_list[1])[0])
        api.exchange.set_actuator_value(EPstate, DATA.handle_Cooling_Setpoint_2, HVAC_setting_value(action_list[1])[1])
        api.exchange.set_actuator_value(EPstate, DATA.handle_Heating_Setpoint_3, HVAC_setting_value(action_list[2])[0])
        api.exchange.set_actuator_value(EPstate, DATA.handle_Cooling_Setpoint_3, HVAC_setting_value(action_list[2])[1])
        api.exchange.set_actuator_value(EPstate, DATA.handle_Heating_Setpoint_4, HVAC_setting_value(action_list[3])[0])
        api.exchange.set_actuator_value(EPstate, DATA.handle_Cooling_Setpoint_4, HVAC_setting_value(action_list[3])[1])
        api.exchange.set_actuator_value(EPstate, DATA.handle_Heating_Setpoint_5, HVAC_setting_value(action_list[4])[0])
        api.exchange.set_actuator_value(EPstate, DATA.handle_Cooling_Setpoint_5, HVAC_setting_value(action_list[4])[1])




    DATA.count += 1
    """Plot"""
    if draw.is_ion:
        if DATA.count > draw.x_view:
            update_plot(draw)

    if DATA.count % 5000 == 0:
        print(f'Time: {DATA.count} / {dt}   Temp: {Temp_mean:.3f} / 23 / {s6:.3f}   Reward(T/E/S): {reward_T:.3f} / {reward_E:.3f} / {reward_S:.3f}')


class Run_EPlus():

    def __init__(self, weather_Dir, out_Dir, IDF_Dir, weights_Dir=None):
        print(f'Your current "EnergyPlusAPI" version is: V-{pyenergyplus.api.EnergyPlusAPI.api_version()}')
        print(f'Your current "pyenergyplus" directory is: {pyenergyplus.api.api_path()}')

        self.weather_Dir = weather_Dir
        self.out_Dir = out_Dir
        self.IDF_Dir = IDF_Dir
        self.weights_Dir = weights_Dir

        self.isPrecondition1 = True
        self.isPrecondition2 = True
        self.isPrecondition3 = True
        self.isPrecondition4 = True

        if not os.path.exists(self.IDF_Dir):
            self.isPrecondition1 = False
            raise FileNotFoundError(f"The path '{self.IDF_Dir}' does not exist.")
        if not os.path.exists(self.weather_Dir):
            self.isPrecondition2 = False
            raise FileNotFoundError(f"The path '{self.weather_Dir}' does not exist.")
        try:
            if not os.path.exists(self.out_Dir):
                os.makedirs(self.out_Dir)
            else:
                print(f"Directory '{self.out_Dir}' already exists.")
                self.remove_folder(self.out_Dir)
                os.makedirs(self.out_Dir)
                print(f"A new folder '{self.out_Dir}' has been created.")
        except Exception as e:
            self.isPrecondition3 = False
            # 捕获创建目录时可能发生的任何异常
            raise OSError(f"Could not create the directory '{self.out_Dir}'. Reason: {e}")

        if DATA.train_switch:
            try:
                if not os.path.exists(self.weights_Dir):
                    os.makedirs(self.weights_Dir)
                else:
                    print(f"Directory '{self.weights_Dir}' already exists.")
                    self.remove_folder(self.weights_Dir)
                    os.makedirs(self.weights_Dir)
                    print(f"A new folder '{self.weights_Dir}' has been created.")
            except Exception as e:
                self.isPrecondition4 = False
                # 捕获创建目录时可能发生的任何异常
                raise OSError(f"Could not create the directory '{self.weights_Dir}'. Reason: {e}")

        if self.isPrecondition1 and self.isPrecondition2 and self.isPrecondition3 and self.isPrecondition4:
            self.deploy_new_EPstate()
        else:
            raise FileNotFoundError("CANNOT deploy new EneryPlus state, "
                                    "please check if all the needed files has been properly placed")

        if self.IDF_Dir[-4:] == '.osm':
            trans_path = osm2idf(self.IDF_Dir)
            self.IDF_Dir = trans_path.idf_file

    def deploy_new_EPstate(self):
        self.EPapi = EnergyPlusAPI()
        print('EnergyPlus state deployed successfully.')

    def start_simulation(self, iscallback=True, isEPtoConsole=False):
        EPapi = EnergyPlusAPI()
        EPstate = EPapi.state_manager.new_state()
        EPapi.runtime.set_console_output_status(EPstate, isEPtoConsole)  # set EP console output status to False
        if iscallback:
            EPapi.runtime.callback_begin_zone_timestep_after_init_heat_balance(EPstate, callback_function)

        EPapi.runtime.run_energyplus(
            EPstate,
            [
                '-w', self.weather_Dir,
                '-d', self.out_Dir,
                self.IDF_Dir
            ]
        )
        EPapi.state_manager.reset_state(EPstate)
        EPapi.state_manager.delete_state(EPstate)
        # if not DATA.train_switch:
        #     EPapi.state_manager.delete_state(EPstate)
        if not draw.is_ion:
            update_plot(draw)


    def remove_folder(self, path):
        shutil.rmtree(path)
        print(f"The directory '{path}' and all its contents have been removed.")


if __name__ == '__main__':
    weather_Dir = "./weather_data/CHN_Beijing.Beijing.545110_CSWD.epw"
    out_Dir = "./out"
    IDF_Dir = "./building_model/new_ue_room/1-18/1.18.osm"
    # IDF_Dir = "./sp/sp.osm"
    weights_Dir = "./weights"
    # IDF_Dir = "./building_model/1.5-1-no-site.osm"

    DATA = Data_Center()
    DATA.train_switch = False
    DATA.wear_out_flag = False
    if not DATA.train_switch:
        draw = Drawing(DATA, is_ion=False, is_zoom=False)
        run_instance = Run_EPlus(weather_Dir, out_Dir, IDF_Dir)
        run_instance.start_simulation(iscallback=True, isEPtoConsole=False)
        print(f'Total Energy consumption: {sum(DATA.Electricity_HVAC):.3f}J')
        save_to_csv(DATA)

    else:
        '''DQN training'''
        draw = Drawing(DATA, is_ion=False, is_zoom=True)
        ReplayBuffer = ReplayBuffer(max_length=128)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        EPagent = DQN(
            state_dim=10,
            action_dim=32,
            lr=0.001,
            gamma=0.5,
            epsilon=0.05,
            device=device,
            update_interval=72
        )
        Epoch = 10
        run_instance = Run_EPlus(weather_Dir, out_Dir, IDF_Dir, weights_Dir)


        print('torch.version: ', torch.__version__)
        print('torch.version.cuda: ', torch.version.cuda)
        print('torch.cuda.is_available: ', torch.cuda.is_available())
        print('torch.cuda.device_count: ', torch.cuda.device_count())
        print('torch.cuda.current_device: ', torch.cuda.current_device())
        device_default = torch.cuda.current_device()
        torch.cuda.device(device_default)
        print('torch.cuda.get_device_name: ', torch.cuda.get_device_name(device_default))
        device = torch.device("cuda")
        print('\n')
        print('Training begins')
        print('\n')

        GO = time.time()
        # for epoch in tqdm(range(1, Epoch + 1), desc='Training process: ', unit='eposide'):
        for epoch in range(1, Epoch + 1):
            print('==' * 50)
            print(f'Training Processing at Epoch {epoch}/{Epoch}')

            #  initialize some variables at the beginning of each epoch
            DATA.initialize_handels()
            DATA.initialize_valuse()
            DATA.count = 0

            #  Run the training
            run_instance.start_simulation(iscallback=True, isEPtoConsole=False)
            torch.save(EPagent.target_Q_net.state_dict(), f'./weights/EPagent_{epoch}.pth')

            #  output some values to the console
            DATA.loss_epoch.append(sum(DATA.loss_random_memory)/len(DATA.loss_random_memory))
            DATA.reward_epoch.append(sum(DATA.reward_random_memory)/len(DATA.reward_random_memory))
            print(f'Training loss: {DATA.loss_epoch[-1]:.2f}')
            print(f'Reward: {DATA.reward_epoch[-1]:.2f}')
            time_delta = time.time() - GO
            print(f'Total Energy consumption: {sum(DATA.Electricity_HVAC):.3f}J')
            print(f'Time use until Epoch {epoch}: {time_delta//3600:.0f}h {time_delta%3600//60:.0f}min {time_delta%60:.0f}s')
            print('\n')

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        # plt.plot(range(1, len(DATA.loss)+1), DATA.loss, 'ro-', label='train loss')
        plt.plot(range(1, Epoch+1), DATA.loss_epoch, 'ro-', label='train loss')
        plt.xlabel('epoch')
        plt.ylabel('train loss')

        plt.subplot(1, 2, 2)
        # plt.plot(range(1, len(DATA.reward)+1), DATA.reward, 'bs-', label='reward')
        plt.plot(range(1, Epoch+1), DATA.reward_epoch, 'bs-', label='reward')
        plt.xlabel('epoch')
        plt.ylabel('reward')
        plt.show()
