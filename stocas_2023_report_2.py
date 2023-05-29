import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import matplotlib.dates as mdates
import statsmodels.api as sm
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import itertools
import warnings
from IPython.display import display, Math

df = pd.read_excel(r'C:\LocalData\sujaiban\sujai.banerji\Aerosol Optical Properties\impactor\Impactor2004_2016.xlsx')

df_1_datetime_start = df.iloc[3:1435, 10]
df_2_datetime_start = df.iloc[1438:2675, 10]
df_datetime_start_frames = [df_1_datetime_start, df_2_datetime_start]
df_date_time_start = pd.concat(df_datetime_start_frames)
df_date_time_start = df_date_time_start.to_numpy()
df_date_time_start = pd.DataFrame(df_date_time_start)
df_date_time_start = df_date_time_start.set_axis(['date_and_time_start'], axis = 1)
df_date_time_start['date_and_time_start']= pd.to_datetime(df_date_time_start['date_and_time_start'])
df_date_time_start = df_date_time_start['date_and_time_start'].dt.strftime('%Y-%m-%d %H:%M:%S')

df_1_datetime_end = df.iloc[3:1435, 12]
df_2_datetime_end = df.iloc[1438:2675, 12]
df_datetime_end_frames = [df_1_datetime_end, df_2_datetime_end]
df_date_time_end = pd.concat(df_datetime_end_frames)
df_date_time_end = df_date_time_end.to_numpy()
df_date_time_end = pd.DataFrame(df_date_time_end)
df_date_time_end = df_date_time_end.set_axis(['date_and_time_end'], axis = 1)
df_date_time_end['date_and_time_end']= pd.to_datetime(df_date_time_end['date_and_time_end'])
df_date_time_end = df_date_time_end['date_and_time_end'].dt.strftime('%Y-%m-%d %H:%M:%S')

df_datetime_duration_1 = df.iloc[3:1435, 14]
df_datetime_duration_2 = df.iloc[3:1435, 14]
df_datetime_duration_frames = [df_datetime_duration_1, df_datetime_duration_2]
df_date_time_duration = pd.concat(df_datetime_duration_frames)
df_date_time_duration = df_date_time_duration.to_numpy()
df_date_time_duration = pd.DataFrame(df_date_time_duration)
df_date_time_duration = df_date_time_duration.set_axis(['date_and_time_duration'], axis = 1)

df_1_under_1 = df.iloc[3:1435, 23]
df_2_under_1 = df.iloc[1438:2675, 23]
df_under_1_frames = [df_1_under_1, df_2_under_1]
df_under_1 = pd.concat(df_under_1_frames)
df_under_1 = df_under_1.to_numpy()
df_under_1 = pd.DataFrame(df_under_1)
df_under_1 = df_under_1.set_axis(['less_than_pm_1'], axis = 1)

df_1_under_2point5 = df.iloc[3:1435, 24]
df_2_under_2point5 = df.iloc[1438:2675, 24]
df_under_2point5_frames = [df_1_under_2point5, df_2_under_2point5]
df_under_2point5 = pd.concat(df_under_2point5_frames)
df_under_2point5 = df_under_2point5.to_numpy()
df_under_2point5 = pd.DataFrame(df_under_2point5)
df_under_2point5 = df_under_2point5.set_axis(['between_1_and_2point5'], axis = 1)

df_1_above_2point5 = df.iloc[3:1435, 25]
df_2_above_2point5 = df.iloc[1438:2675, 25]
df_above_2point5_frames = [df_1_above_2point5, df_2_above_2point5]
df_above_2point5 = pd.concat(df_above_2point5_frames)
df_above_2point5 = df_above_2point5.to_numpy()
df_above_2point5 = pd.DataFrame(df_above_2point5)
df_above_2point5 = df_above_2point5.set_axis(['between_2point5_and_10'], axis = 1)

df_1_under_10 = df.iloc[3:1435, 27]
df_2_under_10 = df.iloc[1438:2675, 27]
df_under_10_frames = [df_1_under_10, df_2_under_10]
df_under_10 = pd.concat(df_under_10_frames)
df_under_10 = df_under_10.to_numpy()
df_under_10 = pd.DataFrame(df_under_10)
df_under_10 = df_under_10.set_axis(['less_than_pm_10'], axis = 1)

df_1_above_10 = df.iloc[3:1435, 26]
df_2_above_10 = df.iloc[1438:2675, 26]
df_above_10_frames = [df_1_above_10, df_2_above_10]
df_above_10 = pd.concat(df_above_10_frames)
df_above_10 = df_above_10.to_numpy()
df_above_10 = pd.DataFrame(df_above_10)
df_above_10 = df_above_10.set_axis(['more_than_pm_10'], axis = 1)

df_1_and_10 = pd.DataFrame(df_under_10.values - df_under_1.values)
df_1_and_10 = df_1_and_10.set_axis(['pm_1_and_10'], axis = 1)

df_1_10_plus = pd.DataFrame(df_1_and_10.values + df_above_10.values)
df_1_10_plus = df_1_10_plus.set_axis(['pm_1_10_plus'], axis = 1)

new_df_frames = [df_date_time_start, df_date_time_end, df_date_time_duration, df_under_1, df_under_2point5, df_above_2point5, df_under_10, df_above_10, df_1_and_10, df_1_10_plus]
new_df = pd.concat(new_df_frames, axis = 1)
new_df.iloc[:, 0] = pd.to_datetime(new_df.iloc[:, 0], format = '%Y-%m-%d %H:%M:%S')
new_df.iloc[:, 1] = pd.to_datetime(new_df.iloc[:, 1], format = '%Y-%m-%d %H:%M:%S')
new_df.iloc[:, 2:] = new_df.iloc[:, 2:].astype(float)

pm_10_abs_df = pd.read_csv(r'C:\LocalData\sujaiban\sujai.banerji\Aerosol Optical Properties\abs\ae31_abs_bc_pm10_SMEARii_2006_2017.txt', sep = '\s+')
pm_10_abs_cols = ['year', 'month', 'date', 'hour', 'minute', 'second', 'pm_10_abs_370', 'pm_10_abs_470', 'pm_10_abs_520', 'pm_10_abs_590', 'pm_10_abs_660', 'pm_10_abs_880', 'pm_10_abs_950', 'pm_10_eBC_370', 'pm_10_eBC_470', 'pm_10_eBC_520', 'pm_10_eBC_590', 'pm_10_eBC_660', 'pm_10_eBC_880', 'pm_10_eBC_950']
pm_10_abs_np = pm_10_abs_df.to_numpy()
pm_10_abs_df = pd.DataFrame(pm_10_abs_np)
pm_10_abs_df.columns = pm_10_abs_cols
pm_10_abs_datetime = pm_10_abs_df['year'].astype(int).astype(str) + '-' + pm_10_abs_df['month'].astype(int).astype(str) + '-' + pm_10_abs_df['date'].astype(int).astype(str) + ' ' + pm_10_abs_df['hour'].astype(int).astype(str) + ':' + pm_10_abs_df['minute'].astype(int).astype(str) + ':' + pm_10_abs_df['second'].astype(int).astype(str)
pm_10_abs_datetime = pm_10_abs_datetime.to_frame()
pm_10_abs_datetime.columns = ['date_and_time']
pm_10_abs_df = pm_10_abs_df.drop(columns = ['year', 'month', 'date', 'hour', 'minute', 'second'])
pm_10_abs_frames = [pm_10_abs_datetime, pm_10_abs_df]
pm_10_abs_df = pd.concat(pm_10_abs_frames, axis = 1)
pm_10_abs_df['date_and_time'] = pd.to_datetime(pm_10_abs_df['date_and_time'])
pm_10_abs_df.iloc[:, 1:] = pm_10_abs_df.iloc[:, 1:].astype(float)

start_year_minus_one = 2009
end_year_plus_one = 2018
time_step = 1

pm_10_abs_winter = []
pm_10_abs_spring = []
pm_10_abs_summer = []
pm_10_abs_autumn = []

for i in range(start_year_minus_one, end_year_plus_one, time_step):
    pm_10_abs_dec = pm_10_abs_df[(pm_10_abs_df.iloc[:, 0].dt.year == i) & (pm_10_abs_df.iloc[:, 0].dt.month == 12)]
    pm_10_abs_jan = pm_10_abs_df[(pm_10_abs_df.iloc[:, 0].dt.year == i + 1) & (pm_10_abs_df.iloc[:, 0].dt.month == 1)]
    pm_10_abs_feb = pm_10_abs_df[(pm_10_abs_df.iloc[:, 0].dt.year == i + 1) & (pm_10_abs_df.iloc[:, 0].dt.month == 2)]
    pm_10_abs_winter.append(pm_10_abs_dec)
    pm_10_abs_winter.append(pm_10_abs_jan)
    pm_10_abs_winter.append(pm_10_abs_feb)
    pm_10_abs_mar = pm_10_abs_df[(pm_10_abs_df.iloc[:, 0].dt.year == i) & (pm_10_abs_df.iloc[:, 0].dt.month == 3)]
    pm_10_abs_apr = pm_10_abs_df[(pm_10_abs_df.iloc[:, 0].dt.year == i) & (pm_10_abs_df.iloc[:, 0].dt.month == 4)]
    pm_10_abs_may = pm_10_abs_df[(pm_10_abs_df.iloc[:, 0].dt.year == i) & (pm_10_abs_df.iloc[:, 0].dt.month == 5)]
    pm_10_abs_spring.append(pm_10_abs_mar)
    pm_10_abs_spring.append(pm_10_abs_apr)
    pm_10_abs_spring.append(pm_10_abs_may)
    pm_10_abs_jun = pm_10_abs_df[(pm_10_abs_df.iloc[:, 0].dt.year == i) & (pm_10_abs_df.iloc[:, 0].dt.month == 6)]
    pm_10_abs_jul = pm_10_abs_df[(pm_10_abs_df.iloc[:, 0].dt.year == i) & (pm_10_abs_df.iloc[:, 0].dt.month == 7)]
    pm_10_abs_aug = pm_10_abs_df[(pm_10_abs_df.iloc[:, 0].dt.year == i) & (pm_10_abs_df.iloc[:, 0].dt.month == 8)]
    pm_10_abs_summer.append(pm_10_abs_jun)
    pm_10_abs_summer.append(pm_10_abs_jul)
    pm_10_abs_summer.append(pm_10_abs_aug)
    pm_10_abs_sep = pm_10_abs_df[(pm_10_abs_df.iloc[:, 0].dt.year == i) & (pm_10_abs_df.iloc[:, 0].dt.month == 9)]
    pm_10_abs_oct = pm_10_abs_df[(pm_10_abs_df.iloc[:, 0].dt.year == i) & (pm_10_abs_df.iloc[:, 0].dt.month == 10)]
    pm_10_abs_nov = pm_10_abs_df[(pm_10_abs_df.iloc[:, 0].dt.year == i) & (pm_10_abs_df.iloc[:, 0].dt.month == 11)]
    pm_10_abs_autumn.append(pm_10_abs_sep)
    pm_10_abs_autumn.append(pm_10_abs_oct)
    pm_10_abs_autumn.append(pm_10_abs_nov)
    
pm_10_abs_winter = pd.concat(pm_10_abs_winter)
pm_10_abs_spring = pd.concat(pm_10_abs_spring)
pm_10_abs_summer = pd.concat(pm_10_abs_summer)
pm_10_abs_autumn = pd.concat(pm_10_abs_autumn)

pm_0_abs_df = pd.read_csv(r'C:\LocalData\sujaiban\sujai.banerji\Aerosol Optical Properties\abs\ae31_abs_bc_pm1_SMEARii_2010_2017.txt', sep = '\s+', header = None)
pm_0_abs_cols = ['year', 'month', 'date', 'hour', 'minute', 'second', 'pm_1_abs_370', 'pm_1_abs_470', 'pm_1_abs_520', 'pm_1_abs_590', 'pm_1_abs_660', 'pm_1_abs_880', 'pm_1_abs_950', 'pm_1_eBC_370', 'pm_1_eBC_470', 'pm_1_eBC_520', 'pm_1_eBC_590', 'pm_1_eBC_660', 'pm_1_eBC_880', 'pm_1_eBC_950']
pm_0_abs_np = pm_0_abs_df.to_numpy()
pm_0_abs_df = pd.DataFrame(pm_0_abs_np)
pm_0_abs_df.columns = pm_0_abs_cols
pm_0_abs_datetime = pm_0_abs_df['year'].astype(int).astype(str) + '-' + pm_0_abs_df['month'].astype(int).astype(str) + '-' + pm_0_abs_df['date'].astype(int).astype(str) + ' ' + pm_0_abs_df['hour'].astype(int).astype(str) + ':' + pm_0_abs_df['minute'].astype(int).astype(str) + ':' + pm_0_abs_df['second'].astype(int).astype(str)
pm_0_abs_datetime = pm_0_abs_datetime.to_frame()
pm_0_abs_datetime.columns = ['date_and_time']
pm_0_abs_df = pm_0_abs_df.drop(columns = ['year', 'month', 'date', 'hour', 'minute', 'second'])
pm_0_abs_frames = [pm_0_abs_datetime, pm_0_abs_df]
pm_0_abs_df = pd.concat(pm_0_abs_frames, axis = 1)
pm_0_abs_df['date_and_time'] = pd.to_datetime(pm_0_abs_df['date_and_time'])
pm_0_abs_df.iloc[:, 1:] = pm_0_abs_df.iloc[:, 1:].astype(float)

pm_1_abs_df = pd.read_csv(r'C:\LocalData\sujaiban\sujai.banerji\Aerosol Optical Properties\beijing_hyytiala_winter_school\2018_2021\hyytiala\ae33_abs_bc_pm1_SMEARii_2018_2021_2.txt', sep = '\s+', header = None)
pm_1_abs_cols = ['year', 'month', 'date', 'hour', 'minute', 'second', 'pm_1_abs_370', 'pm_1_abs_470', 'pm_1_abs_520', 'pm_1_abs_590', 'pm_1_abs_660', 'pm_1_abs_880', 'pm_1_abs_950', 'pm_1_eBC_370', 'pm_1_eBC_470', 'pm_1_eBC_520', 'pm_1_eBC_590', 'pm_1_eBC_660', 'pm_1_eBC_880', 'pm_1_eBC_950']
pm_1_abs_np = pm_1_abs_df.to_numpy()
pm_1_abs_df = pd.DataFrame(pm_1_abs_np)
pm_1_abs_df.columns = pm_1_abs_cols
pm_1_abs_datetime = pm_1_abs_df['year'].astype(int).astype(str) + '-' + pm_1_abs_df['month'].astype(int).astype(str) + '-' + pm_1_abs_df['date'].astype(int).astype(str) + ' ' + pm_1_abs_df['hour'].astype(int).astype(str) + ':' + pm_1_abs_df['minute'].astype(int).astype(str) + ':' + pm_1_abs_df['second'].astype(int).astype(str)
pm_1_abs_datetime = pm_1_abs_datetime.to_frame()
pm_1_abs_datetime.columns = ['date_and_time']
pm_1_abs_df = pm_1_abs_df.drop(columns = ['year', 'month', 'date', 'hour', 'minute', 'second'])
pm_1_abs_frames = [pm_1_abs_datetime, pm_1_abs_df]
pm_1_abs_df = pd.concat(pm_1_abs_frames, axis = 1)
pm_1_abs_df['date_and_time'] = pd.to_datetime(pm_1_abs_df['date_and_time'])
pm_1_abs_df.iloc[:, 1:] = pm_1_abs_df.iloc[:, 1:].astype(float)

pm_abs_df_2 = pd.read_csv(r'C:\LocalData\sujaiban\sujai.banerji\Aerosol Optical Properties\beijing_hyytiala_winter_school\2018_2021\hyytiala\ae33_abs_bc_pm1_SMEARii_2022.txt', sep = '\s+', header = None)
abs_2_datetime_0 = pm_abs_df_2.iloc [:, 0]
abs_2_datetime_1 = pm_abs_df_2.iloc[:, 1]
abs_2_list = []
abs_2_list.append(abs_2_datetime_0)
abs_2_list.append(abs_2_datetime_1)
abs_2_datetime = pd.concat(abs_2_list, axis = 1)
pm_abs_list = []
pm_abs_list.append(abs_2_datetime_0)
pm_abs_list.append(abs_2_datetime_1)
pm_2_abs_df = pd.concat(pm_abs_list, axis = 1)
pm_2_abs_df.columns = ['date', 'time']
pm_2_abs_df = pd.to_datetime(pm_2_abs_df['date'] + ' ' + pm_2_abs_df['time'])
pm_2_abs_df = pm_2_abs_df.to_frame()
pm_2_abs_df.columns = ['date_and_time']
pm_2_abs_df['date_and_time'] = pd.to_datetime(pm_2_abs_df['date_and_time'])

abs_2_df_value = pm_abs_df_2.iloc[:, 2:]
abs_2_df_value.columns = ['pm_1_abs_370', 'pm_1_abs_470', 'pm_1_abs_520', 'pm_1_abs_590', 'pm_1_abs_660', 'pm_1_abs_880', 'pm_1_abs_950', 'pm_1_eBC_370', 'pm_1_eBC_470', 'pm_1_eBC_520', 'pm_1_eBC_590', 'pm_1_eBC_660', 'pm_1_eBC_880', 'pm_1_eBC_950']
pm_1_abs_2 = []
pm_1_abs_2.append(pm_2_abs_df)
pm_1_abs_2.append(abs_2_df_value)
pm_abs_2_df = pd.concat(pm_1_abs_2, axis = 1)

pm_abs_df_list = []
abs_0_2_df = pm_0_abs_df.values
abs_0_2_df = pd.DataFrame(abs_0_2_df)
abs_1_2_df = pm_1_abs_df.values
abs_1_2_df = pd.DataFrame(abs_1_2_df)
abs_2_2_df = pm_abs_2_df.values
abs_2_2_df = pd.DataFrame(abs_2_2_df)
pm_abs_df_list.append(abs_0_2_df)
pm_abs_df_list.append(abs_1_2_df)
pm_abs_df_list.append(abs_2_2_df)
pm_1_abs_df = pd.concat(pm_abs_df_list)
pm_1_abs_df.columns = ['date_and_time', 'pm_1_abs_370', 'pm_1_abs_470', 'pm_1_abs_520', 'pm_1_abs_590', 'pm_1_abs_660', 'pm_1_abs_880', 'pm_1_abs_950', 'pm_1_eBC_370', 'pm_1_eBC_470', 'pm_1_eBC_520', 'pm_1_eBC_590', 'pm_1_eBC_660', 'pm_1_eBC_880', 'pm_1_eBC_950']
pm_1_abs_df['date_and_time'] = pd.to_datetime(pm_1_abs_df['date_and_time'])
pm_1_abs_df.iloc[:, 1:] = pm_1_abs_df.iloc[:, 1:].astype(float)

pm_1_abs_df = pm_1_abs_df.resample('h', on = 'date_and_time').mean()
pm_1_abs_df = pm_1_abs_df.reset_index()

pm_1_abs_winter = []
pm_1_abs_spring = []
pm_1_abs_summer = []
pm_1_abs_autumn = []

for i in range(start_year_minus_one, end_year_plus_one, time_step):
    pm_1_abs_dec = pm_1_abs_df[(pm_1_abs_df.iloc[:, 0].dt.year == i) & (pm_1_abs_df.iloc[:, 0].dt.month == 12)]
    pm_1_abs_jan = pm_1_abs_df[(pm_1_abs_df.iloc[:, 0].dt.year == i + 1) & (pm_1_abs_df.iloc[:, 0].dt.month == 1)]
    pm_1_abs_feb = pm_1_abs_df[(pm_1_abs_df.iloc[:, 0].dt.year == i + 1) & (pm_1_abs_df.iloc[:, 0].dt.month == 2)]
    pm_1_abs_winter.append(pm_1_abs_dec)
    pm_1_abs_winter.append(pm_1_abs_jan)
    pm_1_abs_winter.append(pm_1_abs_feb)
    pm_1_abs_mar = pm_1_abs_df[(pm_1_abs_df.iloc[:, 0].dt.year == i) & (pm_1_abs_df.iloc[:, 0].dt.month == 3)]
    pm_1_abs_apr = pm_1_abs_df[(pm_1_abs_df.iloc[:, 0].dt.year == i) & (pm_1_abs_df.iloc[:, 0].dt.month == 4)]
    pm_1_abs_may = pm_1_abs_df[(pm_1_abs_df.iloc[:, 0].dt.year == i) & (pm_1_abs_df.iloc[:, 0].dt.month == 5)]
    pm_1_abs_spring.append(pm_1_abs_mar)
    pm_1_abs_spring.append(pm_1_abs_apr)
    pm_1_abs_spring.append(pm_1_abs_may)
    pm_1_abs_jun = pm_1_abs_df[(pm_1_abs_df.iloc[:, 0].dt.year == i) & (pm_1_abs_df.iloc[:, 0].dt.month == 6)]
    pm_1_abs_jul = pm_1_abs_df[(pm_1_abs_df.iloc[:, 0].dt.year == i) & (pm_1_abs_df.iloc[:, 0].dt.month == 7)]
    pm_1_abs_aug = pm_1_abs_df[(pm_1_abs_df.iloc[:, 0].dt.year == i) & (pm_1_abs_df.iloc[:, 0].dt.month == 8)]
    pm_1_abs_summer.append(pm_1_abs_jun)
    pm_1_abs_summer.append(pm_1_abs_jul)
    pm_1_abs_summer.append(pm_1_abs_aug)
    pm_1_abs_sep = pm_1_abs_df[(pm_1_abs_df.iloc[:, 0].dt.year == i) & (pm_1_abs_df.iloc[:, 0].dt.month == 9)]
    pm_1_abs_oct = pm_1_abs_df[(pm_1_abs_df.iloc[:, 0].dt.year == i) & (pm_1_abs_df.iloc[:, 0].dt.month == 10)]
    pm_1_abs_nov = pm_1_abs_df[(pm_1_abs_df.iloc[:, 0].dt.year == i) & (pm_1_abs_df.iloc[:, 0].dt.month == 11)]
    pm_1_abs_autumn.append(pm_1_abs_sep)
    pm_1_abs_autumn.append(pm_1_abs_oct)
    pm_1_abs_autumn.append(pm_1_abs_nov)
    
pm_1_abs_winter = pd.concat(pm_1_abs_winter)
pm_1_abs_spring = pd.concat(pm_1_abs_spring)
pm_1_abs_summer = pd.concat(pm_1_abs_summer)
pm_1_abs_autumn = pd.concat(pm_1_abs_autumn)

abs_10_1_winter = pd.merge_asof(pm_10_abs_winter, pm_1_abs_winter, on = 'date_and_time')
abs_10_1_winter['date_and_time'] = pd.to_datetime(abs_10_1_winter['date_and_time']) 
abs_10_1_winter.iloc[:, 1:] = abs_10_1_winter.iloc[:, 1:].astype(float) 
abs_10_1_winter['abs_10_1_520'] = abs_10_1_winter['pm_10_abs_520'] - abs_10_1_winter['pm_1_abs_520']
abs_10_1_winter['abs_10_1_520'] = abs_10_1_winter['abs_10_1_520'].astype(float)

abs_10_1_spring = pd.merge_asof(pm_10_abs_spring, pm_1_abs_spring, on = 'date_and_time')
abs_10_1_spring['date_and_time'] = pd.to_datetime(abs_10_1_spring['date_and_time']) 
abs_10_1_spring.iloc[:, 1:] = abs_10_1_spring.iloc[:, 1:].astype(float) 
abs_10_1_spring['abs_10_1_520'] = abs_10_1_spring['pm_10_abs_520'] - abs_10_1_spring['pm_1_abs_520']
abs_10_1_spring['abs_10_1_520'] = abs_10_1_spring['abs_10_1_520'].astype(float)

abs_10_1_summer = pd.merge_asof(pm_10_abs_summer, pm_1_abs_summer, on = 'date_and_time')
abs_10_1_summer['date_and_time'] = pd.to_datetime(abs_10_1_summer['date_and_time']) 
abs_10_1_summer.iloc[:, 1:] = abs_10_1_summer.iloc[:, 1:].astype(float) 
abs_10_1_summer['abs_10_1_520'] = abs_10_1_summer['pm_10_abs_520'] - abs_10_1_summer['pm_1_abs_520']
abs_10_1_summer['abs_10_1_520'] = abs_10_1_summer['abs_10_1_520'].astype(float)

abs_10_1_autumn = pd.merge_asof(pm_10_abs_autumn, pm_1_abs_autumn, on = 'date_and_time')
abs_10_1_autumn['date_and_time'] = pd.to_datetime(abs_10_1_autumn['date_and_time']) 
abs_10_1_autumn.iloc[:, 1:] = abs_10_1_autumn.iloc[:, 1:].astype(float) 
abs_10_1_autumn['abs_10_1_520'] = abs_10_1_autumn['pm_10_abs_520'] - abs_10_1_autumn['pm_1_abs_520']
abs_10_1_autumn['abs_10_1_520'] = abs_10_1_autumn['abs_10_1_520'].astype(float)

abs_10_1_list = []
abs_10_1_list.append(abs_10_1_winter)
abs_10_1_list.append(abs_10_1_spring)
abs_10_1_list.append(abs_10_1_summer)
abs_10_1_list.append(abs_10_1_autumn)
abs_10_1_df = pd.concat(abs_10_1_list)

abs_10_1_df = abs_10_1_df.reset_index()
abs_10_1_df = abs_10_1_df.iloc[:, 1:]

abs_10_1_df['date_and_time'] = pd.to_datetime(abs_10_1_df['date_and_time'])
abs_10_1_df = abs_10_1_df.set_index('date_and_time')

abs_10_1_df = abs_10_1_df.resample('M').agg(['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])

start_year = abs_10_1_df[('abs_10_1_520', '<lambda_0>')].first_valid_index()
start_year = str(start_year)
start_year = pd.to_datetime(start_year)

end_year = abs_10_1_df[('abs_10_1_520', '<lambda_0>')].last_valid_index()
end_year = str(end_year)
end_year = pd.to_datetime(end_year)

abs_10_1_df = abs_10_1_df.loc[(abs_10_1_df.index >= start_year) & (abs_10_1_df.index <= end_year)]

sns.set_style('darkgrid')
p25 = abs_10_1_df[('abs_10_1_520', 'median')].quantile(0.25)
p75 = abs_10_1_df[('abs_10_1_520', 'median')].quantile(0.75)
display(Math(r'The light absorption for > PM_{1} and \leq PM_{10} at 520 nm is as follows:'))
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(abs_10_1_df.index, abs_10_1_df[('abs_10_1_520', 'median')], 'k-', label = 'Median trend line', linewidth = 1)
ax.plot(abs_10_1_df.index, abs_10_1_df[('abs_10_1_520', 'median')], 'o', markersize = 4, color = 'black', mec = 'none', label = 'Median values')
ax.fill_between(abs_10_1_df.index, abs_10_1_df[('abs_10_1_520', '<lambda_0>')], abs_10_1_df[('abs_10_1_520', '<lambda_1>')], color = 'gray', alpha = 0.25)
ax.set_title('Light absorption for > PM$_1$ and ≤ PM$_1$$_0$')
ax.set_ylabel('Light absorption at 520 nm (M m$^{-1}$)')
ax.set_xlabel('Year')
x = abs_10_1_df.dropna().index
y = abs_10_1_df.dropna()[('abs_10_1_520', 'median')]
x_num = mdates.date2num(x)
slope, intercept, r_value, p_value, std_err = stats.linregress(x_num, y)
trendline = intercept + slope * mdates.date2num(abs_10_1_df.index)
ax.fill_between(abs_10_1_df.index, abs_10_1_df[('abs_10_1_520', '<lambda_0>')], abs_10_1_df[('abs_10_1_520', '<lambda_1>')], color = 'gray', alpha = 0.25)
ax.set_title('Light absorption for > PM$_1$ and ≤ PM$_1$$_0$')
ax.set_ylabel('Light absorption at 520 nm (M m$^{-1}$)')
ax.set_xlabel('Year')
trendline = intercept + slope * mdates.date2num(abs_10_1_df.index)
ax.plot(abs_10_1_df.index, trendline, 'red', label = 'OLS trend line', alpha = 1, linewidth = 2)
ax.legend(loc = 'best')
ax.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
plt.show()

display(Math(r'The actual data, trend component, seasonla component and the residual component of the light absorption for > PM_{1} and \leq PM_{10} at 520 nm is as follows:'))
abs_10_1_df.index = pd.to_datetime(abs_10_1_df.index)
stl = sm.tsa.seasonal_decompose(abs_10_1_df[('abs_10_1_520', 'median')].dropna(), model = 'additive', period = 12)
fig, ax_1 = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
data = abs_10_1_df[('abs_10_1_520', 'median')]
data_with_gaps = data.copy()
data_with_gaps[np.isnan(data)] = np.nan
ax_1[0].set_title('Light absorption for > PM$_1$ and ≤ PM$_1$$_0$')
ax_1[0].set_ylabel('Light absorption at 520 nm')
ax_1[0].plot(abs_10_1_df.index, trendline, 'red', label='OLS trend line', alpha = 1, linewidth = 2)
ax_1[0].plot(data_with_gaps.index, data_with_gaps.values, '-ok', label = 'Original Data', markersize = 4, markerfacecolor = 'black')
ax_1[0].fill_between(abs_10_1_df.index, abs_10_1_df[('abs_10_1_520', '<lambda_0>')], abs_10_1_df[('abs_10_1_520', '<lambda_1>')], color = 'gray', alpha = 0.25)
ax_1[0].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
trend = stl.trend
trend_with_gaps = trend.copy()
trend_with_gaps[np.isnan(data)] = np.nan
ax_1[1].plot(trend_with_gaps.index, trend_with_gaps.values, '-ok', label='Trend Component', markersize=4, markerfacecolor = 'black')
ax_1[1].legend()
ax_1[1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
seasonal = stl.seasonal
seasonal_with_gaps = seasonal.copy()
seasonal_with_gaps[np.isnan(data)] = np.nan
ax_1[2].plot(seasonal_with_gaps.index, seasonal_with_gaps.values, '-ok', label='Seasonal Component', markersize=4, markerfacecolor = 'black')
ax_1[2].legend()
ax_1[2].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
residual = stl.resid
residual_with_gaps = residual.copy()
residual_with_gaps[np.isnan(data)] = np.nan
ax_1[3].set_ylabel('Absorption at 520 nm')
ax_1[3].plot(residual_with_gaps.index, residual_with_gaps.values, '-ok', label='Residual Component', markersize=4, markerfacecolor = 'black')
ax_1[3].legend()
ax_1[3].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()

residual = stl.resid
residual_with_gaps = residual.copy()
residual_with_gaps[np.isnan(data)] = np.nan

display(Math(r'The ACF plot for the light absorption for > PM_{1} and \leq PM_{10} at 520 nm is as follows:'))

fig, ax_2 = plt.subplots(figsize=(12, 4))
plot_acf(residual_with_gaps.dropna(), lags = 24, ax = ax_2)
plt.show()

display(Math(r'The PACF plot for the light absorption for > PM_{1} and \leq PM_{10} at 520 nm is as follows:'))

fig, ax_2 = plt.subplots(figsize=(12, 4))
plot_pacf(residual_with_gaps.dropna(), lags = 24, ax = ax_2)
plt.show()

def train_test_split(data, train_size):
    return data[:train_size], data[train_size:]

train_size = int(0.75 * len(abs_10_1_df.dropna()[('abs_10_1_520', 'median')].dropna()))

train, test = train_test_split(abs_10_1_df.dropna()[('abs_10_1_520', 'median')].dropna(), train_size)

train = train.resample('M').mean()
test = test.resample('M').mean()

train.index = pd.DatetimeIndex(train.index.values, freq='M')
test.index = pd.DatetimeIndex(test.index.values, freq='M')

p = d = q = range(0, 2)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

warnings.filterwarnings("ignore")

best_aic = float('inf')
best_pdq = None
best_seasonal_pdq = None
best_mdl = None

arima_success = False

for param in pdq:
    try:
        tmp_mdl = ARIMA(train,
                        order = param,
                        enforce_stationarity=True,
                        enforce_invertibility=True)
        res = tmp_mdl.fit()

        arima_success = True
        if res.aic < best_aic:
            best_aic = res.aic
            best_pdq = param
            best_seasonal_pdq = None
            best_mdl = tmp_mdl
    except:
        continue

if arima_success:
    print("Best ARIMA model - AIC:{} for order:{}".format(best_aic, best_pdq))
else:
    print("None of the ARIMA models were fitted successfully")

best_aic = float('inf')
best_pdq = None
best_seasonal_pdq = None
tmp_model = None
best_mdl = None

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            tmp_mdl = SARIMAX(train,
                              order = param,
                              seasonal_order = param_seasonal,
                              enforce_stationarity=True,
                              enforce_invertibility=True)
            res = tmp_mdl.fit()

            print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, res.aic))

            if res.aic < best_aic:
                best_aic = res.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal
                best_mdl = tmp_mdl
        except:
            continue

print("Best SARIMA model - AIC:{} for order:{} and seasonal_order:{}".format(best_aic, best_pdq, best_seasonal_pdq))

arima_model = ARIMA(train, order = (12, 1, 1)).fit()

arima_preds = arima_model.predict(start = test.index[0], end = test.index[-1], typ='levels')

sarima_model = SARIMAX(train, order = (1, 0, 1), seasonal_order = (1, 0, 1, 12)).fit()

sarima_preds = sarima_model.predict(start=test.index[0], end = test.index[-1], typ='levels')

arima_rmse = np.sqrt(mean_squared_error(test, arima_preds))
print("ARIMA Model RMSE:", arima_rmse)

sarima_rmse = np.sqrt(mean_squared_error(test, sarima_preds))
print("SARIMA Model RMSE:", sarima_rmse)

print("\nARIMA Model Summary:")
print(arima_model.summary())

print("\nSARIMA Model Summary:")
print(sarima_model.summary())

display(Math(r'The actual data and the ARIMA model for the light absorption for > PM_{1} and \leq PM_{10} at 520 nm is as follows:'))
arima_conf_int = arima_model.get_prediction(start = test.index[0], end = test.index[-1]).conf_int()
fig, ax_3 = plt.subplots(figsize=(12, 6))
ax_3.plot(abs_10_1_df.index, abs_10_1_df[('abs_10_1_520', 'median')], 'k-', label = 'Median trend line', linewidth = 1)
ax_3.plot(abs_10_1_df.index, abs_10_1_df[('abs_10_1_520', 'median')], 'o', markersize = 4, color = 'black', mec = 'none', label = 'Median values')
ax_3.fill_between(abs_10_1_df.index, abs_10_1_df[('abs_10_1_520', '<lambda_0>')], abs_10_1_df[('abs_10_1_520', '<lambda_1>')], color = 'gray', alpha = 0.25)
ax_3.set_title('Light absorption for > PM$_1$ and ≤ PM$_1$$_0$')
ax_3.set_ylabel('Light absorption at 520 nm (M m$^{-1}$)')
ax_3.set_xlabel('Year')
x = abs_10_1_df.dropna().index
y = abs_10_1_df.dropna()[('abs_10_1_520', 'median')]
x_num = mdates.date2num(x)
slope, intercept, r_value, p_value, std_err = stats.linregress(x_num, y)
trendline = intercept + slope * mdates.date2num(abs_10_1_df.index)
ax_3.plot(abs_10_1_df.index, trendline, 'red', label = 'OLS trend line', alpha = 1, linewidth = 2)
ax_3.plot(train.index, train.values, '-ok', label='Actual Data', markersize = 4, markerfacecolor = 'black')
ax_3.plot(test.index, test.values, '-ok', markersize=4, markerfacecolor='black')
ax_3.plot(arima_preds.index, arima_preds.values, '-ob', label='ARIMA Predicted', markersize = 4, markerfacecolor = 'steelblue')
ax_3.fill_between(arima_conf_int.index, arima_conf_int.iloc[:, 0], arima_conf_int.iloc[:, 1], color = 'steelblue', alpha = 0.25, label = 'ARIMA Confidence Interval')
ax_3.legend(loc='best')
ax_3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("ARIMA Predicted vs Actual Data with Confidence Intervals")
plt.show()

display(Math(r'The actual data and the SARIMA model for the light absorption for > PM_{1} and \leq PM_{10} at 520 nm is as follows:'))
sarima_conf_int = sarima_model.get_prediction(start=test.index[0], end=test.index[-1]).conf_int()
fig, ax_4 = plt.subplots(figsize = (12, 6))
ax_4.plot(abs_10_1_df.index, abs_10_1_df[('abs_10_1_520', 'median')], 'k-', label='Median trend line', linewidth = 1)
ax_4.plot(abs_10_1_df.index, abs_10_1_df[('abs_10_1_520', 'median')], 'o', markersize = 4, color = 'black', mec = 'none', label = 'Median values')
ax_4.fill_between(abs_10_1_df.index, abs_10_1_df[('abs_10_1_520', '<lambda_0>')], abs_10_1_df[('abs_10_1_520', '<lambda_1>')], color = 'gray', alpha = 0.25)
ax_4.set_title('Light absorption for > PM$_1$ and ≤ PM$_1$$_0$')
ax_4.set_ylabel('Light absorption at 520 nm (M m$^{-1}$)')
ax_4.set_xlabel('Year')
x = abs_10_1_df.dropna().index
y = abs_10_1_df.dropna()[('abs_10_1_520', 'median')]
x_num = mdates.date2num(x)
slope, intercept, r_value, p_value, std_err = stats.linregress(x_num, y)
trendline = intercept + slope * mdates.date2num(abs_10_1_df.index)
ax_4.plot(abs_10_1_df.index, trendline, 'red', label = 'OLS trend line', alpha = 1, linewidth = 2)
ax_4.plot(train.index, train.values, '-ok', label='Actual Data', markersize = 4, markerfacecolor = 'black')
ax_4.plot(test.index, test.values, '-ok', markersize = 4, markerfacecolor='black')
ax_4.plot(sarima_preds.index, sarima_preds.values, '-og', label='SARIMA Predicted', markersize = 4, markerfacecolor = 'green')
ax_4.fill_between(sarima_conf_int.index, sarima_conf_int.iloc[:, 0], sarima_conf_int.iloc[:, 1], color='g', alpha = 0.25, label='SARIMA Confidence Interval')
ax_4.legend(loc='best')
ax_4.legend(loc='center left', bbox_to_anchor = (1, 0.5))
plt.title("SARIMA Predicted vs Actual Data with Confidence Intervals")
plt.show()

display(Math(r'The actual data, the ARIMA model and the SARIMA model for the light absorption for > PM_{1} and \leq PM_{10} at 520 nm is as follows:'))
fig, ax_5 = plt.subplots(figsize = (12, 6))
ax_5.plot(abs_10_1_df.index, abs_10_1_df[('abs_10_1_520', 'median')], 'k-', label = 'Median trend line', linewidth = 1)
ax_5.plot(abs_10_1_df.index, abs_10_1_df[('abs_10_1_520', 'median')], 'o', markersize = 4, color = 'black', mec = 'none', label = 'Median values')
ax_5.fill_between(abs_10_1_df.index, abs_10_1_df[('abs_10_1_520', '<lambda_0>')], abs_10_1_df[('abs_10_1_520', '<lambda_1>')], color = 'gray', alpha = 0.25)
ax_5.set_title('Light absorption for > PM$_1$ and ≤ PM$_1$$_0$')
ax_5.set_ylabel('Light absorption at 520 nm (M m$^{-1}$)')
ax_5.set_xlabel('Year')
x = abs_10_1_df.dropna().index
y = abs_10_1_df.dropna()[('abs_10_1_520', 'median')]
x_num = mdates.date2num(x)
slope, intercept, r_value, p_value, std_err = stats.linregress(x_num, y)
trendline = intercept + slope * mdates.date2num(abs_10_1_df.index)
ax_5.plot(abs_10_1_df.index, trendline, 'red', label = 'OLS trend line', alpha = 1, linewidth = 2)
ax_5.plot(arima_preds.index, arima_preds.values, '-ob', label='ARIMA Predicted', markersize = 4, markerfacecolor = 'steelblue')
ax_5.fill_between(arima_conf_int.index, arima_conf_int.iloc[:, 0], arima_conf_int.iloc[:, 1], color = 'steelblue', alpha = 0.25, label = 'ARIMA Confidence Interval')
ax_5.plot(sarima_preds.index, sarima_preds.values, '-og', label='SARIMA Predicted', markersize = 4, markerfacecolor = 'green')
ax_5.fill_between(sarima_conf_int.index, sarima_conf_int.iloc[:, 0], sarima_conf_int.iloc[:, 1], color = 'g', alpha = 0.25, label = 'SARIMA Confidence Interval')
ax_5.legend(loc = 'best')
ax_5.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
plt.title("ARIMA and SARIMA Predicted vs Actual Data with Confidence Intervals")
plt.show()