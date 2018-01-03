# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 15:49:36 2017

@author: jackh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import json


"""
Documentation
http://dublincitynoise.sonitussystems.com/applications/api/api-doc.html

Google API key AIzaSyDfjTJ9_zaRMvxu2xJu9UpHwLYg4YK8d0Y

"""

### Get Location names from API

def get_location_names():
    url = 'http://dublincitynoise.sonitussystems.com/applications/api/dublinnoisedata.php'
    
    parameters = {'returnLocationStrings':True ,'location':"all" }
    
    response_locations = requests.get(url,params=parameters)
    
    locations = response_locations.json()
    
    return locations


### Get location long and lat from google API

# AIzaSyDfjTJ9_zaRMvxu2xJu9UpHwLYg4YK8d0Y

def get_long_lat(locations,API_Key):
    """ Get coordinates for locations vector in Dublin.
    Args:
        API_Key: API Key available from google
        locations: vector of locations in dublin
        
    returns:
        latitude: vector of latitude coordinates
        longitude: vector of longitude coordinates
    """


    # Set up arrays to populate
    lat = np.zeros(np.shape(locations)[0])
    lng = np.zeros(np.shape(locations)[0])
    ite = 0
    url_google = 'https://maps.googleapis.com/maps/api/geocode/json'

    # get coordinate for each location and add to vectors
    for i in locations:
        
        parameters_google = {'address':i+', Dublin, Ireland','key':'AIzaSyDfjTJ9_zaRMvxu2xJu9UpHwLYg4YK8d0Y'}
        response_google = requests.get(url_google,params=parameters_google)
        
        if(response_google.status_code != 200):
            print 'error'
            print locations[ite]
                
    
        long_lat = response_google.json()
    
        results = long_lat['results'][0]
        
        lat[ite] = results['geometry']['location']['lat']
        lng[ite] = results['geometry']['location']['lng']
        
        ite = ite+1
        
    return lat,lng
    

def get_noise_levels(locations,latitude,longitude,start_date = '2017-01-15 00:00:00',end_date = '2017-01-20 00:00:00'):
    """ Get noise levels from locations in Dublin
    Args:
        locations: vector of locations in Dublin
        latitude, longitude: coordinates relating to locations vector
        start_date, end_date: range of data to return. Default is 15 Jan 2017 to 20 Jan 2017
    
    """
    
    # Get First Location data from API
    
    # Convert dates to UNIX
    start = pd.Timestamp(start_date)
    start = start.value // 10 ** 9 #Converts to Unix 
    
    end = pd.Timestamp(end_date)
    end = end.value // 10 ** 9 #Converts to Unix 
    
    """ Unix date
    unix timestamp in seconds since the Epoch (January 1 1970 00:00:00 GMT)
    Timestamp is just unix time with nanoseconds (so divide it by 10**9)
    """
    

    url = 'http://dublincitynoise.sonitussystems.com/applications/api/dublinnoisedata.php'
    
    parameters = {'location':1,'start':start,'end':end}
    
    response_data = requests.get(url,params=parameters)
    
    json_data = response_data.json()
    
    # Extract times, dates and noise data from response
    times = json_data['times']
    dates = json_data['dates']
    noise = np.array(json_data['aleq'],dtype='float')
    
    # date and time dataframe
    date_time=pd.DataFrame({'date':dates,'time':times})
    
    data_tableau = pd.DataFrame({'Date':pd.to_datetime(date_time['date'] + ' ' + date_time['time'],dayfirst = True),'Noise':noise})
    
    data_tableau['Location'] = locations[0]
    
    data_tableau['Lat'] = latitude[0]
    data_tableau['Long'] = longitude[0]
    

    
    
    ### Add data of remaining locations
    # Loop through locations vector
    for i in np.arange(2,np.shape(locations)[0]+1): 
    
        parameters_new = {'location':i,'start':start,'end':end}
        
        response_data_new = requests.get(url,params=parameters_new)
        
        json_data_new = response_data_new.json()
        
        print(locations[i-1])
        print(json_data_new['entries'])
        
        noise_new = np.array(json_data_new['aleq'],dtype='float')
        
        times = json_data_new['times']
        dates = json_data_new['dates']
        
        date_time_new=pd.DataFrame({'date':dates,'time':times})
    
        data_tableau_new = pd.DataFrame({'Date':pd.to_datetime(date_time_new['date'] + ' ' + date_time_new['time'],dayfirst = True),'Noise':noise_new})


        data_tableau_new['Location'] = locations[i-1]
        data_tableau_new['Lat'] = latitude[i-1]
        data_tableau_new['Long'] = longitude[i-1]
                
        data_tableau = data_tableau.append(data_tableau_new)
        
    return data_tableau
        
locations = get_location_names()
latitude,longitude = get_long_lat(locations,'AIzaSyDfjTJ9_zaRMvxu2xJu9UpHwLYg4YK8d0Y')
data_tableau = get_noise_levels(locations,latitude,longitude,start_date = '2017-01-01 00:00:00',end_date = '2017-12-10 00:00:00')

data_tableau.to_csv(path_or_buf = 'C:\Users\jackh\OneDrive\Documents\Projects\Dublin noise\\noise_tableau.csv')






### TODO
"""
Output for tableau
Maps API for Lat/Long
ARIMA

"""









