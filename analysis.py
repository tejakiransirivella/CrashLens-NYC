import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from preprocess import PreProcess
from visualization import Visualization
import plotly.express as px
from sklearn.cluster import DBSCAN


def preprocess_data(pre):
    '''
    This function preprocesses the data by filtering rows, dropping columns, filling missing values, and converting date and time columns to datetime format.
    '''

    print("Total number of accidents in NYC: ", len(pre.data))
    pre.filter_rows('BOROUGH', 'BROOKLYN')
    print("Total number of accidents in Brooklyn: ", len(pre.data))    

    pre.drop_columns(['CONTRIBUTING FACTOR VEHICLE 4','CONTRIBUTING FACTOR VEHICLE 5','VEHICLE TYPE CODE 4','VEHICLE TYPE CODE 5','LOCATION','CROSS STREET NAME','OFF STREET NAME'])

    pre.fill_na("NUMBER OF PEDESTRIANS INJURED",0)
    pre.fill_na("NUMBER OF PEDESTRIANS KILLED",0)
    pre.fill_na("NUMBER OF CYCLIST INJURED",0)
    pre.fill_na("NUMBER OF CYCLIST KILLED",0)
    pre.fill_na("NUMBER OF MOTORIST INJURED",0)
    pre.fill_na("NUMBER OF MOTORIST KILLED",0)

    pre.sum('NUMBER OF PERSONS INJURED',['NUMBER OF PEDESTRIANS INJURED','NUMBER OF CYCLIST INJURED','NUMBER OF MOTORIST INJURED'])
    pre.sum('NUMBER OF PERSONS KILLED',['NUMBER OF PEDESTRIANS KILLED','NUMBER OF CYCLIST KILLED','NUMBER OF MOTORIST KILLED'])

    pre.data = pre.data[(pre.data['NUMBER OF PERSONS INJURED'] != 0) | (pre.data['NUMBER OF PERSONS KILLED'] != 0)]

    pre.data["VEHICLE TYPE CODE 1"].fillna(pre.data["VEHICLE TYPE CODE 1"].mode()[0], inplace=True)
    pre.data["VEHICLE TYPE CODE 2"].fillna(pre.data["VEHICLE TYPE CODE 2"].mode()[0], inplace=True)
    pre.data["VEHICLE TYPE CODE 3"].fillna(pre.data["VEHICLE TYPE CODE 3"].mode()[0], inplace=True)
    
    pre.data = pre.data[pre.data['CRASH DATE'] != 0]
    pre.data['CRASH DATE'] = pd.to_datetime(pre.data['CRASH DATE'] + ' ' + pre.data['CRASH TIME'])
    pre.data['Month'] = pre.data['CRASH DATE'].dt.month
    pre.data['Year'] = pre.data['CRASH DATE'].dt.year

    print(pre.data.head(1))


def total_accidents_by_month(data,visualize):
    '''
    This function visualizes the number of accidents by month for the years 2020 and 2022.
    '''
    brooklyn_data_2022 = data[data['Year'] == 2022]
    brooklyn_data_2020 = data[data['Year'] == 2020]

    month_crashes_2022 = brooklyn_data_2022['Month'].value_counts().sort_index()
    month_crashes_2020 = brooklyn_data_2020['Month'].value_counts().sort_index()
    months_list = ['Jan', 'Feb', 'Mar', 'April', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    month_crashes_2022 = month_crashes_2022.to_list()
    month_crashes_2020 = month_crashes_2020.to_list()
    visualize.draw_histogram(months_list,month_crashes_2020,month_crashes_2022,'Monthly Accident Frequency for 2020 and 2022','Month',
                'Number of Collisions','2020','2022')


def pedestrian_accidents_by_month(data,visualize):
    '''
    This function visualizes the number of pedestrian accidents by month for the years 2020 and 2022.
    '''
    brooklyn_data_2022 = data[data['Year'] == 2022]
    brooklyn_data_2020 = data[data['Year'] == 2020]
    months_list = ['Jan', 'Feb', 'Mar', 'April', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

    month_crashes_pedestrians_2020 = brooklyn_data_2020[(brooklyn_data_2020['NUMBER OF PEDESTRIANS INJURED'] > 0) | 
                   (brooklyn_data_2020['NUMBER OF PEDESTRIANS KILLED'] > 0)]['Month'].value_counts().sort_index()
    month_crashes_pedestrians_2022 = brooklyn_data_2022[(brooklyn_data_2022['NUMBER OF PEDESTRIANS INJURED'] > 0) | 
                   (brooklyn_data_2022['NUMBER OF PEDESTRIANS KILLED'] > 0)]['Month'].value_counts().sort_index()
    visualize.draw_histogram(months_list,month_crashes_pedestrians_2020.to_list(),month_crashes_pedestrians_2022.to_list(),
               'Monthly Pedestrian Accident Frequency for 2020 and 2022','Month',
               'Number of Pedestrian Accidents','2020','2022')
    
def cyclist_accidents_by_month(data,visualize):
    '''
    This function visualizes the number of cyclist accidents by month for the years 2020 and 2022.
    '''

    brooklyn_data_2022 = data[data['Year'] == 2022]
    brooklyn_data_2020 = data[data['Year'] == 2020]
    months_list = ['Jan', 'Feb', 'Mar', 'April', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    month_crashes_cyclists_2020 = brooklyn_data_2020[(brooklyn_data_2020['NUMBER OF CYCLIST INJURED'] > 0) | 
                   (brooklyn_data_2020['NUMBER OF CYCLIST KILLED'] > 0)]['Month'].value_counts().sort_index()
    month_crashes_cyclists_2022 = brooklyn_data_2022[(brooklyn_data_2022['NUMBER OF CYCLIST INJURED'] > 0) | 
                   (brooklyn_data_2022['NUMBER OF CYCLIST KILLED'] > 0)]['Month'].value_counts().sort_index()
    visualize.draw_histogram(months_list,month_crashes_cyclists_2020.to_list(),month_crashes_cyclists_2022.to_list(),
               'Monthly Cyclist Accident Frequency for 2020 and 2022','Month',
               'Number of Cyclist Accidents','2020','2022')
    
def motorist_accidents_by_month(data,visualize):
    '''
    This function visualizes the number of motorist accidents by month for the years 2020 and 2022.
    '''

    brooklyn_data_2022 = data[data['Year'] == 2022]
    brooklyn_data_2020 = data[data['Year'] == 2020]
    months_list = ['Jan', 'Feb', 'Mar', 'April', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    month_crashes_motorists_2020 = brooklyn_data_2020[(brooklyn_data_2020['NUMBER OF MOTORIST INJURED'] > 0) | 
                   (brooklyn_data_2020['NUMBER OF MOTORIST KILLED'] > 0)]['Month'].value_counts().sort_index()
    month_crashes_motorists_2022 = brooklyn_data_2022[(brooklyn_data_2022['NUMBER OF MOTORIST INJURED'] > 0) | 
                   (brooklyn_data_2022['NUMBER OF MOTORIST KILLED'] > 0)]['Month'].value_counts().sort_index()
    visualize.draw_histogram(months_list,month_crashes_motorists_2020.to_list(),month_crashes_motorists_2022.to_list(),
               'Monthly Motorist Accident Frequency for 2020 and 2022','Month',
               'Number of Motorist Accidents','2020','2022')
    
def total_accidents_by_weekday(data,visualize):
    '''
    This function visualizes the number of accidents by weekday for the years 2020 and 2022.
    '''

    brooklyn_data_2022 = data[data['Year'] == 2022]
    brooklyn_data_2020 = data[data['Year'] == 2020]
    weekday_crashes_2020 = brooklyn_data_2020['CRASH DATE'].dt.day_name().value_counts()
    weekday_crashes_2022 = brooklyn_data_2022['CRASH DATE'].dt.day_name().value_counts()
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    crashes_list_2020 = []
    crashes_list_2022 = []
    for day_name in day_names:
        crashes_list_2020.append(weekday_crashes_2020[day_name])
        crashes_list_2022.append(weekday_crashes_2022[day_name])
    visualize.draw_histogram(day_names,crashes_list_2020,crashes_list_2022,
               'Total Crashes per Weekday for 2020 and 2022','Weekday',
               'Number of Crashes','2020','2022')
    
def total_accidents_between_6am_and_12pm(data,visualize):
    '''
    This function visualizes the number of accidents between 6am and 12pm for the years 2020 and 2022.
    '''
    brooklyn_data_2022 = data[data['Year'] == 2022]
    brooklyn_data_2020 = data[data['Year'] == 2020]
    hourly_crashes_2020 = brooklyn_data_2020[(brooklyn_data_2020['CRASH DATE'].dt.hour >=6) & (brooklyn_data_2020['CRASH DATE'].dt.hour <=12)]
    hourly_crashes_2020['hour'] = hourly_crashes_2020['CRASH DATE'].dt.hour
    hourly_crashes_2020 = hourly_crashes_2020['hour'].value_counts().sort_index()
    hourly_crashes_2022 = brooklyn_data_2022[(brooklyn_data_2022['CRASH DATE'].dt.hour >=6) & (brooklyn_data_2022['CRASH DATE'].dt.hour <=12)]
    hourly_crashes_2022['hour'] = hourly_crashes_2022['CRASH DATE'].dt.hour
    hourly_crashes_2022 = hourly_crashes_2022['hour'].value_counts().sort_index()
    hours = np.arange(6,13)

    visualize.draw_histogram(hours,hourly_crashes_2020,hourly_crashes_2022,
                'Total Hourly Crashes for 2020 and 2022','Hour',
                'Number of Crashes','2020','2022')
    
def total_accidents_sliding_window_60_days(data,visualize):
    '''
    This function visualizes the number of accidents by day with a sliding window of 60 days for the years 2020 and 2022.
    '''

    brooklyn_data_time = data[(data['CRASH DATE'] >= pd.Timestamp('2020-01-01')) & 
                                                 (data['CRASH DATE'] <=  pd.Timestamp('2022-10-01'))]

    daily_accidents = brooklyn_data_time.groupby(brooklyn_data_time['CRASH DATE'].dt.date).size().reset_index(name='accidents')

    window_size = 60
    daily_accidents['rolling_accidents'] = daily_accidents['accidents'].rolling(window=window_size).sum()
    daily_accidents.dropna(subset=['rolling_accidents'], inplace=True)

    max_accidents = daily_accidents[daily_accidents['rolling_accidents'] == daily_accidents['rolling_accidents'].max()]

    visualize.plot_line_graph(daily_accidents['CRASH DATE'], daily_accidents['rolling_accidents'],'End Date with Window size of 60 Days',
                'Number of Accidents (Rolling)','Rolling Accidents Over Time', max_accidents)
    
def top_10_accidents_days(data,visualize):
    '''
    This function visualizes the top 10 days with the most accidents for the year 2022.
    '''
    brooklyn_data_2022 = data[data['Year'] == 2022]
    top_10_dates = brooklyn_data_2022['CRASH DATE'].dt.date.value_counts().sort_values(ascending = False)[:10]
    visualize.draw_single_histogram(top_10_dates.index.astype(str).tolist(),top_10_dates.tolist(),'Date',"Number of crashes",
    'Top 10 Days with Most Crashes in 2022','2022')
    
def total_accidents_by_person_type(data,visualize,year,month):
    '''
    This function visualizes the number of accidents by person type for the given year and month.
    '''

    accident_counts = []
    labels = ["PEDESTRIANS","CYCLISTS","MOTORISTS"]

    brooklyn_year_month_data = data[((data['Year'] == year) & (data['Month'] == month))]

    brooklyn_pedestrian_data = brooklyn_year_month_data[(brooklyn_year_month_data['NUMBER OF PEDESTRIANS INJURED'] != 0) | 
                                            (brooklyn_year_month_data['NUMBER OF PEDESTRIANS KILLED'] != 0)]
    accident_counts.append(len(brooklyn_pedestrian_data))

    brooklyn_cyclist_data = brooklyn_year_month_data[(brooklyn_year_month_data['NUMBER OF CYCLIST INJURED'] != 0) | 
                                            (brooklyn_year_month_data['NUMBER OF CYCLIST KILLED'] != 0)]

    accident_counts.append(len(brooklyn_cyclist_data))

    brooklyn_motorist_data = brooklyn_year_month_data[(brooklyn_year_month_data['NUMBER OF MOTORIST INJURED'] != 0) | 
                                            (brooklyn_year_month_data['NUMBER OF MOTORIST KILLED'] != 0)]

    accident_counts.append(len(brooklyn_motorist_data))
    months = ['Jan', 'Feb', 'Mar', 'April', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    title = 'Affected People ratios in Accidents in '+ months[month-1] + " " + str(year)
    visualize.draw_pie_chart(accident_counts,labels,title)

def total_accidents_by_contributing_factor(data,visualize,year,month):
    '''
    This function visualizes the number of accidents by contributing factor for the given year and month.
    '''

    brooklyn_year_month_data = data[((data['Year'] == year) & (data['Month'] == month))]
    brooklyn_year_month_data = brooklyn_year_month_data[brooklyn_year_month_data['CONTRIBUTING FACTOR VEHICLE 1'].notna()]

    contributing_factor_counts = brooklyn_year_month_data['CONTRIBUTING FACTOR VEHICLE 1'].value_counts()
    months = ['Jan', 'Feb', 'Mar', 'April', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    title = 'Contributing Factor percentage in Accidents in ' + months[month-1] + " " + str(year)
    visualize.draw_pie_chart(contributing_factor_counts,contributing_factor_counts.index,title)


def total_accidents_by_vehicle_type(data,visualize,year,month):
    '''
    This function visualizes the number of accidents by vehicle type for the given year and month.
    '''

    brooklyn_year_month_data = data[((data['Year'] == year) & (data['Month'] == month))]
    brooklyn_year_month_data = brooklyn_year_month_data[brooklyn_year_month_data['VEHICLE TYPE CODE 1'].notna()]

    vehicle_type_counts = brooklyn_year_month_data['VEHICLE TYPE CODE 1'].value_counts()
    months = ['Jan', 'Feb', 'Mar', 'April', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    title = 'Vehicle type contribution for accidents in '+ months[month-1] + " " + str(year)
    visualize.draw_pie_chart(vehicle_type_counts,vehicle_type_counts.index,title)

def generate_heatmap_latitude_longitude(data,visualize):
    '''
    This function generates a heatmap of accidents based on latitude and longitude for the year 2022.
    '''

    brooklyn_data_2022 = data[data['Year'] == 2022]
    geoData = brooklyn_data_2022.dropna(subset=['LATITUDE', 'LONGITUDE'])
    coordinates = geoData[['LATITUDE', 'LONGITUDE']]
    epsilon = 0.003
    min_samples = 2
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    dbscan.fit_predict(coordinates)
    cluster_labels = dbscan.labels_
    accident_data_with_labels = np.column_stack((coordinates, cluster_labels))
    unique_labels = np.unique(cluster_labels)
    geoData = pd.DataFrame(accident_data_with_labels, columns=['LATITUDE', 'LONGITUDE', 'Cluster'])
    fig = px.scatter_mapbox(geoData, 
                        lat="LATITUDE", 
                        lon="LONGITUDE",
                        color="Cluster",
                        color_continuous_scale=px.colors.sequential.YlOrRd,
                        zoom=8, 
                        height=800,
                        width=800)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_traces(showlegend=False)
    fig.show()

    
def perform_analysis(data,visualize):
    '''
    This function performs the analysis by calling the respective functions for each analysis.
    '''
    
    total_accidents_by_month(data,visualize)
    pedestrian_accidents_by_month(data,visualize)
    cyclist_accidents_by_month(data,visualize)
    motorist_accidents_by_month(data,visualize)
    total_accidents_by_weekday(data,visualize)
    total_accidents_between_6am_and_12pm(data,visualize)
    top_10_accidents_days(data,visualize)
    total_accidents_sliding_window_60_days(data,visualize)
    total_accidents_by_person_type(data,visualize,2020,7)
    total_accidents_by_person_type(data,visualize,2022,7)
    total_accidents_by_contributing_factor(data,visualize,2020,7)
    total_accidents_by_contributing_factor(data,visualize,2022,7)
    total_accidents_by_vehicle_type(data,visualize,2020,7)
    total_accidents_by_vehicle_type(data,visualize,2022,7)
    generate_heatmap_latitude_longitude(data,visualize)


def main():
    file_path = "data/crashes.csv"
    pre = PreProcess()
    pre.read_data(file_path)
    preprocess_data(pre)
    visualize = Visualization()
    perform_analysis(pre.data,visualize)


if __name__ == "__main__":
    main()