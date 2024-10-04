"""
Data-driven modeling in Python, winter term 2023/2024
Project: The Norway Lemming - Hippolyte PASCAL
"""

import sys
sys.path.append('C:/Users/hippo_kq2e550/OneDrive/Desktop/KIT/Datengetriebene Modellierung mit Python/DMP_Project_WS2324')
import mod.my_plotter as mp
from scipy.special import binom
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

LITTERS_PER_YEAR = 2.5
NEWBORNS_PER_LITTER = 7
NEWBORNS_PER_YEAR = LITTERS_PER_YEAR * NEWBORNS_PER_LITTER
LIFESPAN_IN_DAYS = 2*365
F = 75 # Number of lemmings that can eat per day
D = 5 # Number of days to reach starvation

def return_b(): 
    # Probability of a lemming being born per unit of time (dt) = Birth rate 
    # NEWBORNS_PER_YEAR = LITTERS_PER_YEAR * NEWBORNS_PER_LITTER
    return ((NEWBORNS_PER_YEAR / 2)/ 365) # Only females give births 


def return_d(snow_depth=None):
    if snow_depth is not None : 
        return get_d(snow_depth) # Using the get_d() function to obtain the death rate depending on the snow depth
    else :
    # Probability of a lemming dying born per unit of time (dt) = Death rate
        return (1/ LIFESPAN_IN_DAYS)

def return_f(N, snow_depth=None, temp=None):
    if snow_depth is not None and temp is not None:
        number_eating_lem = get_f(snow_depth, temp) # Using the get_f() function to obtain the number of eating lemmings 
        # depending on the snow depth and temperature
        return ((((N - number_eating_lem)/N)**D) *(sum(binom(D - 1, i)* (number_eating_lem/N)**i/ (D +i) for i in range(D))))
    else :
        # Additional death rate that depends only on the current population size N
        return ((((N - F)/N)**D) *(sum(binom(D - 1, i)* (F/N)**i/ (D +i) for i in range(D))))

def get_weather(start_date, t):

    base_year= 2020 # Year with a 29th of February
    weather_data = np.load('./data/weather_data.npy', allow_pickle=True).item()
    start_datetime = datetime.strptime(f'{base_year}-{start_date}', "%Y-%m-%d")
    post_datetime = start_datetime + timedelta(days=t)  # Date of the day with a time delay of t days
    
    if post_datetime.month == 2 and start_datetime.day == 29:
        post_datetime = post_datetime.replace(day=28)

    post_month = post_datetime.month
    post_day = post_datetime.day

    # Find the index in the dataset that matches the post date
    for i in range(len(weather_data['day'])):
        if int(weather_data['month'][i]) == post_month and int(weather_data['day'][i]) == post_day: 
        # Return the corresponding weather information
            snow_depth = weather_data['snow'][i]
            temperature = weather_data['temp'][i]
            return snow_depth, temperature

    return None, None

def fit_d_sd():
    # Load the data
    d_sd = np.load('./data/d_sd.npy', allow_pickle=True).item()
    X = np.array(d_sd['snow_depth']).reshape(-1, 1)  # Features (snow depth)
    y = np.array(d_sd['d'])  # Target variable (death rates)

    # Fit a polynomial regression model
    degree = 2  # 2nd degree polynomial to fit
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y) 

    # Save the model coefficients of the 2nd degree polynomial
    np.save('./data/model_parameters.npy', model.steps[1][1].coef_)

    plt.close('all')
    # Creation of the figure + axis
    mp.init_plot()
    fig, ax = plt.subplots()

    # Plot the raw data and the model prediction
    ax.plot(X, y, label='Raw data o', color=mp.lightgreen) 
    X_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1) # Sequence of 100 snow depth values evenly distributed
    y_fit = model.predict(X_fit) # Predict death rates for the generated sequence of snow depth values
    ax.plot(X_fit, y_fit, label='Model fit', color=mp.purple)
    ax.set_title('Death rate variation over snow depth + 2nd degree polynomial fit', fontweight='bold')
    ax.set_xlabel('Snow Depth')
    ax.set_ylabel('Death Rate')
    ax.legend()
    plt.savefig('./exp/d_sd_fit.png', dpi =300)

    
def get_d(snow_depth, data=None):
    if data is None:
    # Load the model data from file
        param = np.load('./data/model_parameters.npy')
    else:
        param = data

    # Using the model parameters of the 2nd degree polynomial
    death_rate = param[0] + param[1] * snow_depth + param[2] * snow_depth**2
    return death_rate
    
def fit_f_sd_T():
    # Load the data
    data = np.load('./data/f_sd_T.npy', allow_pickle=True).item()
    X = np.column_stack((data['snow_depth'], data['temperature']))
    y = data['food']
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
    # Fit the MLPRegressor model
    mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=10000, random_state=42)
    mlp.fit(X_train, y_train)

    # Predict and evaluate model
    y_pred_train = mlp.predict(X_train)
    y_pred_test = mlp.predict(X_test)
    RMSE_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
  
    plt.close('all')
    # Creation of the figure + axis
    mp.init_plot()
    fig, ax1 = plt.subplots()

    # Plotting
    ax1.scatter(X_train[:, 0], y_train, color=mp.purple, label='Training data')
    ax1.scatter(X_test[:, 0], y_test, color=mp.lightgreen, label='Testing data')
    fig.suptitle('Model Performance of the number of eating lemmings over snow depth variation:')
    ax1.set_title(f'RMSE Train = {RMSE_train:.2f}, RMSE Test = {RMSE_test:.2f}')
    ax1.set_xlabel('Snow Depth')
    ax1.set_ylabel('Number of Lemmings Eating')
    ax1.legend()
    plt.savefig('./exp/f_sd_T_fit.png', dpi =300)
  
    # Save the mlp model
    np.save('./data/mlp_model.npy', mlp)

def get_f(snow_depth, temp, data=None):
    if data is None:
        model = np.load('./data/mlp_model.npy', allow_pickle=True).item()
    else:
        model = data
    # Predict the #of eating lemmings based on snow depth and temp
    prediction = model.predict([[snow_depth, temp]])

    return prediction[0]
