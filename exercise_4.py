# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:46:30 2023

@author: sujaiban
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.odr as odr
from scipy.optimize import curve_fit
from scipy.stats import ttest_1samp

# Read data from CSV file
df = pd.read_csv(r'C:/LocalData/sujaiban/sujai.banerji/STOCAS2023/exercise_4/sample500.csv')

# Get independent and dependent variables
x = df['Anthropogenic_index']
y = df['emit_org1']

# Calculate slope, intercept, R^2, p-value and standard error using OLS method
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Print OLS regression results
print('OLS Regression Results:')
print(f'slope = {slope:.2f}')
print(f'intercept = {intercept:.2f}')
print(f'r-squared = {r_value**2:.2f}')
print(f'p-value = {p_value:.2f}')

# Define a function for linear regression
def f(B, x):
    return B[0]*x + B[1]

# Create a linear regression model using ODR method
model_5 = odr.Model(f)
data_5 = odr.RealData(x, y)
odr_fit = odr.ODR(data_5, model_5, beta0=[slope, intercept])
output_5 = odr_fit.run()

# Print ODR regression results
print('\nODR Regression Results:')
print(f'slope = {output_5.beta[0]:.2f}')
print(f'intercept = {output_5.beta[1]:.2f}')
print(f'r-squared = {output_5.res_var:.2f}')
print(f'p-value = {output_5.res_var:.2f}')

# Plot the scatter plot and regression lines for both OLS and ODR
plt.scatter(x, y, edgecolors='b', facecolors='none')
plt.plot(x, slope*x + intercept, color='red', label='OLS')
plt.plot(x, output_5.beta[0]*x + output_5.beta[1], color='green', label='ODR')

# Add labels and title to the plot
plt.xlabel('Anthropogenic_index')
plt.ylabel('emit_org1')
plt.title('OLS vs ODR')

# Add legend to the plot
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)

# Add text to the plot to show OLS and ODR regression equations
plt.text(0.1, 0.9, f'OLS: y = {slope:.2f}x + {intercept:.2f}', transform=plt.gca().transAxes)
plt.text(0.1, 0.8, f'ODR: y = {output_5.beta[0]:.2f}x + {output_5.beta[1]:.2f}', transform=plt.gca().transAxes)

# Add R^2 value and p-value for both OLS and ODR to the plot
plt.text(0.1, 0.7, f'OLS R-squared = {r_value**2:.2f}, p-value = {p_value:.2f}', transform=plt.gca().transAxes)
plt.text(0.1, 0.6, f'ODR R-squared = {output_5.res_var:.2f}, p-value = {output_5.res_var:.2f}', transform=plt.gca().transAxes)

# Display the plot
plt.show()

# define the logarithmic function to fit the data
def logarithmic_function(x, a, b):
    return a * np.log(x) + b

# fit the logarithmic function to the data
popt, pcov = curve_fit(logarithmic_function, df['Anthropogenic_index'], df['emit_org1'])

# calculate the residuals
residuals = df['emit_org1'] - logarithmic_function(df['Anthropogenic_index'], *popt)

# calculate the total sum of squares
tss = np.sum((df['emit_org1'] - np.mean(df['emit_org1'])) ** 2)

# calculate the residual sum of squares
rss = np.sum(residuals ** 2)

# calculate the R-squared value
r_squared = 1 - (rss / tss)

# calculate the standard errors of the parameter estimates
perr = np.sqrt(np.diag(pcov))

# calculate the t-statistics and p-values for the parameters
t, p = ttest_1samp(popt / perr, 0)

# plot the data and the fitted logarithmic function
plt.scatter(df['Anthropogenic_index'], df['emit_org1'], edgecolors='b', facecolors='none', label='Data')
x_values = np.linspace(min(df['Anthropogenic_index']), max(df['Anthropogenic_index']), 100)
y_values = logarithmic_function(x_values, *popt)
plt.plot(x_values, y_values, color='red', label='Logarithmic Fit')

# add equation of the logarithmic function to the plot
a = round(popt[0], 3)
b = round(popt[1], 3)
plt.text(0.5, 0.2, f'y = {a}ln(x) + {b}', fontsize=12, transform=plt.gcf().transFigure)

# add R-squared and p-value to the plot
plt.text(0.5, 0.15, f'R-squared = {round(r_squared, 3)}, p = {round(p, 3)}', fontsize=12, transform=plt.gcf().transFigure)

# set plot labels and title
plt.xlabel('Anthropogenic_index')
plt.ylabel('emit_org1')
plt.title('Logarithmic fit')

# add legend to the plot
plt.legend()

# display the plot
plt.show()

def exponential_function(x, a, b, c):
    return a * np.exp(-b * x**2) + c

# fit the exponential function to the data with different initial parameter values
popt, pcov = curve_fit(exponential_function, df['Anthropogenic_index'], df['emit_org1'], p0=[1, 0.0001, 1], maxfev=10000)

# create a subplot
fig, ax = plt.subplots()

# plot the data and the fitted exponential function
ax.scatter(df['Anthropogenic_index'], df['emit_org1'], edgecolors='b', facecolors='none', label='Data')
x_values = np.linspace(min(df['Anthropogenic_index']), max(df['Anthropogenic_index']), 100)
y_values = exponential_function(x_values, *popt)
ax.plot(x_values, y_values, color='red', label='Exponential fit')

# add equation of the exponential function, R^2 value, and p-value to the plot
a = round(popt[0], 3)
b = round(popt[1], 5)
c = round(popt[2], 3)
textstr = f'y = {a}e^(-{b}x) + {c}\n$R^2$ = {r_squared:.2f}\n$p$-value = {p_value:.2f}'
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)

# set plot labels and title
ax.set_xlabel('Anthropogenic Index')
ax.set_ylabel('Organic Emissions')
ax.set_title('Exponential fit')

# add legend to the plot
ax.legend()

# calculate the correlation coefficient
corr = df['Anthropogenic_index'].corr(df['emit_org1'])
print(f'Correlation coefficient: {corr:.2f}')

# calculate R^2 value and p-value
residuals = df['emit_org1'] - exponential_function(df['Anthropogenic_index'], *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((df['emit_org1'] - np.mean(df['emit_org1']))**2)
r_squared = 1 - (ss_res / ss_tot)
n = len(df['Anthropogenic_index'])
p = len(popt)
p_value = 1 - stats.f.cdf(r_squared / (1 - r_squared) * (n - p) / (p - 1), p, n - p)
print(f'R^2 value: {r_squared:.2f}')
print(f'p-value: {p_value:.2f}')

# display the plot
plt.show()

# define a polynomial function of degree 3
def polynomial_function(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

# fit the polynomial function to the data with different initial parameter values
popt, pcov = curve_fit(polynomial_function, df['Anthropogenic_index'], df['emit_org1'], p0=[1, 1, 1, 1], maxfev=10000)

# create a subplot
fig, ax = plt.subplots()

# plot the data and the fitted polynomial function
ax.scatter(df['Anthropogenic_index'], df['emit_org1'], edgecolors='b', facecolors='none', label='Data')
x_values = np.linspace(min(df['Anthropogenic_index']), max(df['Anthropogenic_index']), 100)
y_values = polynomial_function(x_values, *popt)
ax.plot(x_values, y_values, color='red', label='Polynomial fit')

# add equation of the polynomial function, R^2 value, and p-value to the plot
a = round(popt[0], 3)
b = round(popt[1], 3)
c = round(popt[2], 3)
d = round(popt[3], 3)
textstr = f'y = {a}x^3 + {b}x^2 + {c}x + {d}\n$R^2$ = {r_squared:.2f}\n$p$-value = {p_value:.2f}'
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)

# set plot labels and title
ax.set_xlabel('Anthropogenic Index')
ax.set_ylabel('Organic Emissions')
ax.set_title('Polynomial fit')

# add legend to the plot
ax.legend()

# calculate the correlation coefficient
corr = df['Anthropogenic_index'].corr(df['emit_org1'])
print(f'Correlation coefficient: {corr:.2f}')

# calculate R^2 value and p-value
residuals = df['emit_org1'] - polynomial_function(df['Anthropogenic_index'], *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((df['emit_org1'] - np.mean(df['emit_org1']))**2)
r_squared = 1 - (ss_res / ss_tot)
n = len(df['Anthropogenic_index'])
p = len(popt)
p_value = 1 - stats.f.cdf(r_squared / (1 - r_squared) * (n - p) / (p - 1), p, n - p)
print(f'R^2 value: {r_squared:.2f}')
print(f'p-value: {p_value:.2f}')

# display the plot
plt.show()

# define a quadratic function
def quadratic_function(x, a, b, c):
    return a*x**2 + b*x + c

x_data = df['Anthropogenic_index']
y_data = df['emit_org1']

# fit the quadratic function to the data with different initial parameter values
popt, pcov = curve_fit(quadratic_function, x_data, y_data, p0=[1, 1, 1], maxfev=10000)

# create a subplot
fig, ax = plt.subplots()

# plot the data and the fitted quadratic function
ax.scatter(x_data, y_data, edgecolors='b', facecolors='none', label='Data')
x_values = np.linspace(min(x_data), max(x_data), 100)
y_values = quadratic_function(x_values, *popt)
ax.plot(x_values, y_values, color='red', label='Quadratic fit')

# add equation of the quadratic function, R^2 value, and p-value to the plot
a = round(popt[0], 3)
b = round(popt[1], 3)
c = round(popt[2], 3)
residuals = y_data - quadratic_function(x_data, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_data - np.mean(y_data))**2)
r_squared = 1 - (ss_res / ss_tot)
n = len(x_data)
p = len(popt)
p_value = 1 - stats.f.cdf(r_squared / (1 - r_squared) * (n - p) / (p - 1), p, n - p)
textstr = f'y = {a}x^2 + {b}x + {c}\n$R^2$ = {r_squared:.2f}\n$p$-value = {p_value:.2f}'
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)

# set plot labels and title
ax.set_xlabel('Anthropogenic Index')
ax.set_ylabel('Organic Emissions')
ax.set_title('Quadratic fit')

# add legend to the plot
ax.legend()

# calculate the correlation coefficient
corr = x_data.corr(y_data)
print(f'Correlation coefficient: {corr:.2f}')

# calculate R^2 value and p-value
print(f'R^2 value: {r_squared:.2f}')
print(f'p-value: {p_value:.2f}')

# display the plot
plt.show()
