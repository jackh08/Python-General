# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 11:58:13 2017

@author: jackh
"""

### Libraries
import pandas as pd
import sklearn.preprocessing as sklp
import sklearn.model_selection as sklm
import sklearn.metrics as sklmet
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
import seaborn as sn
import random
from mpl_toolkits.mplot3d import axes3d
import statsmodels.api as sm
from scipy.stats.stats import pearsonr


### General loops function ---------------------------------------------------
    while error_iter > err and i < max_iter:
    
    for j in np.arange(0,10):

    def Name(arg):
        return {'Spec_Mean':spec_mean,'Spec_SD':spec_sd,'Finish_Mean':finish_mean,'Finish_SD':finish_sd,'Pearson':cor}
    
### Numpy-------------------------------------------------------------------
np.round()
np.zeros()
np.arange(0,10) # for 1 to 9
np.power(x,2)
np.abs(-1)

np.cpncatenate(a,b,axis = 0)

np.dot(A,B) # matrix multiplication

A.reshape(3,3) #to matrix 3x3

np.tile(data.iloc[0],(5,1))                 # Tile rows down
np.tile((data['feat1']),(5,1)).T            # TIle columns across

np.argmin()
np.argmax()

### Random-------------------------------------------------------------------
np.random.rand(C,n) #C,n is matrix shape
np.random.uniform(low=-1, high=1, size=(p,n))



### Pandas-------------------------------------------------------------------
pd.read_csv('C:\Users\jackh\OneDrive\Documents\College\Python\Python Exam\ .csv')

#new
data = pd.DataFrame({'feat1':f1,'feat2':f2,})

# Add column
data['new'] = xxx
data.columns = [['a'],['b']]

# Selection
data.drop('Label',1)
num = data[['feat1','feat2']]               # select columns
data.iloc[1:4]                              # select rows
data.iloc[1:4,1:3]                          # Select rows and columns
data.iloc[[0,1],[0,1]]                      # select points
data.iloc[rows.flatten(),:]                 # select rows by np.array

#Aggregate
grouped = data.groupby(by='feature')
grouped.describe(),sum(),mean() etc


# Merge dataframes
new = pd.merge(df1,manu2)

new = df.append(newcol)

#For pd objects eg data.xxx

data.shape
data.label.value_counts()
data.label.nunique()
data.label.describe() # like summary in R
sample.plot.scatter(x='feat1',y='feat2')
data.plot(kind='bar')


# pd functions eg pd.xxx

pd.Categorical(data.Label).codes # to put categorical to numeric
pd.unique(data.labels)
pd.dropna()
pd.fillna()

# plot
data.boxplot(column = 'finish',by = 'material')

### Preprocessing-------------------------------------------------------------
#Scale
sklp.minmax_scale(data,(0,1)) # data must be numerical pd or np
standardized_Dataset = sklp.scale(Dataset, axis=0)
Normalized_Dataset = sklp.normalize(Dataset, norm='l2')
binarized_Dataset = skp.binarize(Dataset,threshold=0.0)

# Missing data

imp = sklp.Imputer(missing_values=0,strategy='mean',axis=0)
imp.fit_transform(Dataset)

# PCA
import sklearn.decomposition as skd
pca = skd.PCA(n_components=n, whiten=False)
pca.fit(Dataset)
Dataset_Reduced_Dim = pca.transform(Dataset)

# Train and Test
x_train, x_test, y_train, y_test = sklm.train_test_split(x,y,test_size = 0.2)


# Dummy encoding
from statsmodels.tools import categorical
cat_encod = categorical(data, dictnames=False, drop=False) #may need reshape(-1,1)







### plot ------------------------------------------------------------------
plt.plot(x,y)
plt.title('Training Error by Iteration')
plt.xlabel('Iteration Number')
plt.ylabel('Error')

# plot data in 3D plot
plt.figure()
axx = plt.axes(projection='3d')
axx.scatter(data[:,0], data[:,1],data[:,2],s = 1.5,c = 'red')
plt.title('3D Plot')
plt.show()

# Multiple plots
fig = plt.figure()
ax1 = fig.add_subplot(131)
plot()...

ax2 = fig.add_subplot(132)
plot()
plt.title('Histogram of Spec')

# MAtrix scatter
axes = pd.tools.plotting.scatter_matrix(X)



### Updating array - undefined size -----------------------------------------
updated = np.empty((0, 3))
updated = np.append(updated, first, axis=0) #add first row
updated = np.append(updated,new,axis = 0)  # In loop

### model building - numpy--------------------------------------------------

import numpy.polynomial.polynomial as nppp
# see week 4 slides
c,stats=nppp.polyfit(x,y,degree,full=True,w=None)
nppp.polyval(datasample,c)




### Model building - stats models------------------------------------------

### Model building - scipy--------------------------------------------------
import scipy.stats as sps
sps.linregress(x,y)
(gradient,intercept,r_value,p_value,stderr) = stats.linregress(x,y)

# clustering
import scipy.cluster.vq as spcv
Centre,Var = spcv.kmeans(X, Num_of_clusters )
id,dist = spcv.vq(X,Centre)




### Regression - Scikitlearn------------------------------------------------

#regression
from sklearn.linear_model import LinearRegression,Ridge,Lasso

lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

Ridge(alpha = n)
Lasso(alpha = n)

from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor(n_estimators=3)


from sklearn.svm import SVR
svr = SVR(kernel= 'linear/poly/rbf/sigmoid')
svr.fit(X_train, y_train)
svr.predict(X_test)

from sklearn.neural_network import MLPRegressor # Week 7 slides

mlp_reg = MLPRegressor(hidden_layer_sizes=10,
activation='relu’, solver='adam’, learning_rate='constant’,
learning_rate_init=0.01, max_iter=1000, tol=0.0001)

mlp_reg.fit(X, y)
mlp_reg.predict(X_dash)

from sklearn import neighbors #k nearest neighbours regression
reg_knn = neighbors.KNeighborsRegressor(n_neighbors,weights='distance/uniform')



### Classification - scikitlearn---------------------------------------------
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=10, random_state=12)

from sklearn.svm import SVC #week 8
svc = SVC(C=1.0, kernel=‘rbf’, degree=3, gamma=‘auto’, probability=False,tol=0.001, max_iter=-1, random_state=None)

from sklearn.naive_bayes import GaussianNB
NB = GaussianNB(priors)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3)
kmeans.fit(data)
clusters_k_means = kmeans.predict(data)

### Metrics----------------------------------------------------------------


#regression
sklmet.mean_absolute_error(y_true, y_pred)
np.sqrt(sklmet.mean_squared_error(y_true, y_pred))         # RMSE
sklmet.r2_score(y_true, y_pred)


sklmet.accuracy_score(y2_test,rf2.predict(X_test))

#Confusion matrix
con=sklmet.confusion_matrix(y_true=, y_pred = )
sn.heatmap(con, annot=True)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Homogeneity
sklm.homogeneity_score(clusters_k_means,clusters_sub)


### Time series---------------------------------------------------------------
# set up
air['Month'] = pd.to_datetime(air['Month'])
indexed_df = air.set_index('Month')
timeseries = indexed_df['Passengers']


"""
Stationary:
    mean not a function of time
    Variance not a function of time
    covariance not a function of time
    
    Check with plots or dickey fuller test
    
    WEEK 9 SLIDES
    
"""

# tests
# DIckey Fuller
from statsmodels.tsa.stattools import adfuller
adfuller(air.Passengers)
    
# Auto correlation FUnction (ACF) corr between series and lagged version
from statsmodels.tsa.stattools import acf
lag_acf = acf(air.Passengers, nlags = 4)

# Partial ACF
from statsmodels.tsa.stattools import pacf
lag_Pacf = pacf(air.Passengers, nlags = 4)

# CrossCorrelation Function (CCF) : The cross-correlation function is a measure of self-similarity between two timeseries.
from statsmodels.tsa.stattools import ccf
lag_ccf= ccf(air.Passengers, air.Passengers)


# Plotting
plt.subplot(221)
plt.plot(timeseries, color='black', label='original')
plt.plot(rolmean, color='blue', label='Rolling Mean')
plt.plot(rolstd, color='red', label='Rolling Deviation')
plt.legend(loc='best')
plt.title('Original Data, Rolling Mean & Standard Deviation')
plt.subplot(223)
plt.plot(lag_pacf, color='orange', label='auto correlation func')
plt.legend(loc='best')
plt.title('Partial Auto Correlation Function')
plt.subplot(224)
plt.plot(lag_acf, color='green', label='partial auto correlation func ')
plt.legend(loc='best')
plt.title('Auto Correlation Function')
plt.show()


# ARIMA
from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime
# ============ Creating a random timeseries ========== #
counts= np.arange(1, 21) + 0.2 * (np.random.random(size=(20,)) - 0.5)
start = pd.datetime.strptime("1 Nov 16", "%d %b %y")
daterange = pd.date_range(start, periods=20)
table = {"count": counts, "date": daterange}
# ================= Pre-processing ====================#
data = pd.DataFrame(table)
data.set_index("date", inplace=True)
print(data)
# =============== Setting up ARIMA model ============= #
model = ARIMA(data[0:len(data)-1], (1,1,1))
model_fit = model.fit(disp=0)
print(model_fit.forecast())


### Images
#Libraries
from skimage import io
from skimage.transform import rotate
from skimage.transform import resize
from skimage.transform import rescale
from skimage.color import rgb2hsv
from skimage.color import hsv2rgb 
from skimage.color import rgb2gray 
from skimage.draw import line, polygon, circle, ellipse, bezier_curve
from skimage.filters import sobel, roberts, scharr, prewitt
from skimage import data
from skimage.measure import find_contours
from skimage.feature import match_template

dice = io.imread('C:\Users\jackh\OneDrive\Documents\College\Python\Images\dice.jpg')
plt.imshow(dice)

rotated = rotate(pose,180)
resized = resize(pose,(150,150))

coloured = rgb2hsv(pose)
coloured2 = hsv2rgb(pose)
coloured3 = rgb2gray(pose)

edge_sobel = sobel(image)
edge_roberts = roberts(image)
edge_scharr = scharr(image)
edge_prewitt = prewitt(image)

#====== Find contours at a constant value of 0.8 =====# really doesn't work
contours = find_contours(r, 0.8)
#=== Display the image and plot all contours found ===#
plt.imshow(r, interpolation='nearest', cmap=plt.cm.gray)
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
        plt.show()


# really for anything else look in week 10 slides and hope for best.










