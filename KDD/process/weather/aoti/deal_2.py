# -*- coding: utf-8 -*-
"""
Created on Thu May 10 13:48:48 2018

@author: zhcao
"""

import pandas as pd

fin = "E:/KDD/KDD_data/Beijing_historical_meo_grid.csv"
dataset = pd.read_csv(fin, header = 0)

def getWeather(point_list):
    cnt = 1
    for x in point_list:
        df = dataset[dataset['stationName'] == x]
        df = df.iloc[:, 3:]
        df.columns = ['utc_time', 'temperature_' + str(cnt), 'pressure_' + str(cnt), 'humidity_' + str(cnt), 'wind_direction_' + str(cnt), 'wind_speed/kph_' + str(cnt)]
        df.to_csv("E:/KDD/weather_data/aoti_data/df_" + str(cnt) + "_weather.csv")
        cnt = cnt + 1

point_list = ['beijing_grid_283', 'beijing_grid_303', 'beijing_grid_304', 'beijing_grid_305', 'beijing_grid_325']
getWeather(point_list) 

     