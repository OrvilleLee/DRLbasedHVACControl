import pyenergyplus
from pyenergyplus.api import EnergyPlusAPI



def callback_function(EPstate):
    api = EnergyPlusAPI()
    if api.exchange.api_data_fully_ready(EPstate):
        print('Api for data exchange is fully ready')
    else:
        print(('still waiting'))
        return

    # print(DATA.handle_Site_Outdoor_Air_Drybulb_Temperature)

