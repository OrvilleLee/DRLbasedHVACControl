import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Drawing():
    def __init__(self, DATA, is_ion=False, is_zoom = True):
        self.DATA = DATA
        self.is_ion = is_ion

        self.is_zoom = is_zoom
        if self.is_ion:
            plt.ion()

        self.x_view = 144 * 7

        self.fig, self.ax = plt.subplots(figsize=(20, 10))

        self.ax.set_title("Zone Temperature and Zone Energy Consumption")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Zone Temperature(℃)")


        self.ax2 = self.ax.twinx()
        self.ax2.set_ylabel("HAVC energy consumption(J)")

        self.plot_line()

        self.ax.legend(loc=2, bbox_to_anchor=(-0.17, 1))
        self.ax2.legend(loc=2, bbox_to_anchor=(-0.17, 0.75))

    def plot_line(self):
        # self.ax.plot(self.DATA.x, self.DATA.Environmental_Impact_Total_CO2_Emissions_Carbon_Equivalent_Mass,
        #              label="CO2 mass", color='black',linewidth=1)
        self.ax.plot(self.DATA.x, self.DATA.Site_Outdoor_Air_Drybulb_Temperature,
                     label="Outdoor Temperature", color='#FFD700', linewidth=1)

        self.ax.plot(self.DATA.x, self.DATA.Zone_Air_Temperature_1,
                     label="Zone 1 Temperature", color='#48D1CC', linewidth=2)
        self.ax.plot(self.DATA.x, self.DATA.Zone_Air_Temperature_2,
                     label="Zone 2 Temperature", color='#7FFFAA', linewidth=2)
        self.ax.plot(self.DATA.x, self.DATA.Zone_Air_Temperature_3,
                     label="Zone 3 Temperature", color='#7B68EE', linewidth=2)
        self.ax.plot(self.DATA.x, self.DATA.Zone_Air_Temperature_4,
                     label="Zone 4 Temperature", color='#FFD700', linewidth=2)
        self.ax.plot(self.DATA.x, self.DATA.Zone_Air_Temperature_5,
                     label="Zone 5 Temperature", color='#20B2AA', linewidth=2)
        self.ax.plot(self.DATA.x, self.DATA.Zone_Mean_Temperature,
                     label="Zone Mean Temperature", color='#20B2BB', linewidth=5, alpha=0.5)


        self.ax.plot(self.DATA.x, self.DATA.Zone_Thermostat_Heating_Setpoint_Temperature_1,
                     label="Zone_Heating_Setpoint_1", color='red', linewidth=0.5)
        self.ax.plot(self.DATA.x, self.DATA.Zone_Thermostat_Cooling_Setpoint_Temperature_1,
                     label="Zone_Cooling_Setpoint_1", color='cyan', linewidth=0.5)

        self.ax.plot(self.DATA.x, self.DATA.reward,
                     label="Reward", color='grey', linewidth=1)



        self.ax2.plot(self.DATA.x, self.DATA.Electricity_HVAC,
                     label="Electricity_HVAC", color='red', linewidth=1)
        self.ax2.plot(self.DATA.x, self.DATA.Electricity_Zone_1,
                      label="Electricity_Zone_1", color='#FF6347', linewidth=0.5)

    def set_ax_view(self):
        self.ax.set_ylim(-25, 40)
        self.ax2.set_ylim(0, 7e7)
        if self.is_zoom:
            self.ax.set_xlim(self.DATA.x[-self.x_view], self.DATA.x[-1])
            # self.ax.set_ylim(-25, 40)
            # self.ax2.set_ylim(0, 1e6)
