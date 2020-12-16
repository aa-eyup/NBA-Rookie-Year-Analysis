#
#
#

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

my_data = pd.read_csv('C://path to data//nba_stats.csv')

pre_x = my_data.iloc[:, 2:-1].values
y = my_data.iloc[:, -1].values

# normalize the attributes
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(pre_x)

#------------------------------------PCA---------------------------------------
from sklearn.decomposition import PCA


pca = PCA(n_components = 2)
pca_train = pca.fit_transform(x_train)
pca_test = pca.transform(x_test)
cumvar = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
comps = pca.components_



comps = pca.components_
print('PC1 Eigen Vector: ', comps[0])
print('                                ')
print('PC2 Eigen Vector: ', comps[1])


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(pca_train, y_train)
b0 = regressor.intercept_
b_coefs = regressor.coef_

# PREDICTTING TEST
y_pred = regressor.predict(pca_test)


#---------------------------------------Decision Tree------------------------
#train test split
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


from sklearn.tree import DecisionTreeClassifier
dcc_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dcc_model = dcc_classifier.fit(x_train, y_train)

dccy_pred = dcc_model.predict(x_test)

from sklearn.metrics import confusion_matrix
dcc_cm = confusion_matrix(dccy_pred, y_test)

pca_dc_regressor = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
pca_dc_model = pca_dc_regressor.fit(pca_train, y_train)
pca_dc_pred = pca_dc_model.predict(pca_test)
pca_dc_cm = confusion_matrix(pca_dc_pred, y_test)

(x_set, y_set) = (pca_train, y_train)
x1, x2 = np.meshgrid(np.arange(x_set[:, 0].min() - 1, x_set[:, 0].max() + 1, 0.01),
                     np.arange(x_set[:, 1].min() - 1, x_set[:, 1].max() + 1, 0.01))
pca_dc_regressor.predict(np.array([x1.ravel(), x2.ravel()]).T)

plt.contourf(x1, x2, pca_dc_regressor.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'blue')))

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

# i is a counter starting @ 0 (a result of enumerate)
# j is the value (a result of enumerate)
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], s = 3,
                c = ListedColormap(('yellow', 'black'))(i), label = str(j))
plt.title('Decision Tree Classifier (yellow = 0, black = 1)')
plt.xlabel('PCA 1 (63% varience)')
plt.ylabel('PCA 2 (15% varience)')
plt.legend()
plt.show()

#---------------------------------------Random Forrest------------------------
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 200, random_state = 0)
rf_model = rf_classifier.fit(x_train, y_train)

rfy_pred = rf_model.predict(x_test)

rf_cm = confusion_matrix(rfy_pred, y_test)


pca_rf_regressor = RandomForestClassifier(n_estimators = 200, random_state = 0)
pca_rf_model = pca_rf_regressor.fit(pca_train, y_train)
pca_rf_pred = pca_rf_model.predict(pca_test)
pca_rf_cm = confusion_matrix(pca_rf_pred, y_test)

(x_set, y_set) = (pca_train, y_train)
x1, x2 = np.meshgrid(np.arange(x_set[:, 0].min() - 1, x_set[:, 0].max() + 1, 0.01),
                     np.arange(x_set[:, 1].min() - 1, x_set[:, 1].max() + 1, 0.01))
pca_rf_regressor.predict(np.array([x1.ravel(), x2.ravel()]).T)

plt.contourf(x1, x2, pca_rf_regressor.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'blue')))

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

# i is a counter starting @ 0 (a result of enumerate)
# j is the value (a result of enumerate)
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], s = 3,
                c = ListedColormap(('yellow', 'black'))(i), label = str(j))
plt.title('Random Forest Classification (yellow = 0, black = 1)')
plt.xlabel('PCA 1 (63% varience)')
plt.ylabel('PCA 2 (15% varience)')
plt.legend()
plt.show()



#------------------------------------Logistic Regression---------------------------------------
from sklearn.linear_model import LogisticRegression

lr_regressor = LogisticRegression(random_state = 0)
lr_model = lr_regressor.fit(x_train, y_train)
lry_pred = lr_model.predict(x_test)
lr_cm = confusion_matrix(lry_pred, y_test)

pca_lr_regressor = LogisticRegression(random_state = 0)
pca_lr_model = pca_lr_regressor.fit(pca_train, y_train)
pca_lr_pred = pca_lr_model.predict(pca_test)
pca_lr_cm = confusion_matrix(pca_lr_pred, y_test)

(x_set, y_set) = (pca_train, y_train)
x1, x2 = np.meshgrid(np.arange(x_set[:, 0].min() - 1, x_set[:, 0].max() + 1, 0.01),
                     np.arange(x_set[:, 1].min() - 1, x_set[:, 1].max() + 1, 0.01))
pca_lr_regressor.predict(np.array([x1.ravel(), x2.ravel()]).T)

plt.contourf(x1, x2, pca_lr_regressor.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'blue')))

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

# i is a counter starting @ 0 (a result of enumerate)
# j is the value (a result of enumerate)
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], s = 3,
                c = ListedColormap(('yellow', 'black'))(i), label = str(j))
plt.title('Logistic Regression (yellow = 0, black = 1)')
plt.xlabel('PCA 1 (63% varience)')
plt.ylabel('PCA 2 (15% varience)')
plt.legend()
plt.show()

