# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 12:59:29 2017

@author: jackh
"""

''' 
DataQuest Web scraping - APIs
'''

import requests
import json

response = requests.get("http://api.open-notify.org/iss-now.json")
response.status_code

"""
STATUS CODES
200 - Everything went okay, and the server returned a result (if any).
301 - The server is redirecting you to a different endpoint. This can happen 
when a company switches domain names, or an endpoint's name has changed.
401 - The server thinks you're not authenticated. This happens when you don't 
send the right credentials to access an API (we'll talk about this in a later
 mission).
400 - The server thinks you made a bad request. This can happen when you don't 
send the information the API requires to process your request, among other 
things.
403 - The resource you're trying to access is forbidden; you don't have the 
right permissions to see it.
404 - The server didn't find the resource you tried to access.
"""


url2 = 'http://api.open-notify.org/iss-pass.json'
response = requests.get(url2)
response.status_code

# Requires 2 parameters...

parameters = {"lat": 40.71, "lon": -74}

# Make a get request with the parameters.
response = requests.get("http://api.open-notify.org/iss-pass.json", params=parameters)

# Print the content of the response (the data the server returned)
print(response.content)

# This gets the same data as the command above
response = requests.get("http://api.open-notify.org/iss-pass.json?lat=40.71&lon=-74")
print(response.content)

"""
The JSON library has two main methods:

dumps -- Takes in a Python object, and converts it to a string
loads -- Takes a JSON string, and converts it to a Python object
"""

json_data = response.json()
print(type(json_data))
print(json_data)
json_data['response'][0]['duration']

"""
json data includes:
    message
    request
    response
"""

print(response.headers['Content-Type'])

# How many people in space?

url3 = 'http://api.open-notify.org/astros.json'
parameters = {"lat": 40.71, "lon": -74}

response = requests.get(url3,
                        params = parameters)

response.status_code

response.headers
json_data = response.json()

json_data['number']


""" SUMMARY
response.requests.get(url)
response.status_code - to see success or not
json_data = response.json()
"""


















