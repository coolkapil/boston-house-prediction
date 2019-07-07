

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
df = pd.read_csv('/home/kapil/Downloads/boston_x_y_train.csv')
data = np.loadtxt('/home/kapil/Downloads/boston_x_y_train.csv',delimiter = ',')
data.shape

print(type(df))
```

    <class 'pandas.core.frame.DataFrame'>



```
# Loading data
data = np.genfromtxt('/home/kapil/Downloads/boston_x_y_train.csv', delimiter=',')
# Splitting into X_train and Y_train
X_train = data[:, 0:13]
Y_train = data[:, 13]


# we will try to convert the training data into degree 2 to calculate a more complex boundary
columns = X_train.shape[1]

# cross pairs
for j in range(columns):
    for k in range(j + 1, columns):
        new_column = (X_train[:, j] * X_train[:, k]).reshape(-1, 1)
        X_train = np.append(X_train, new_column, axis=1)

# pair with self
for i in range(columns):
    new_column = (X_train[:, i] * X_train[:, i]).reshape(-1,1)
    X_train = np.append(X_train, new_column, axis=1)
    
# adding a row of ones to training data for gradient descent implementation
ones = np.ones(X_train.shape[0]).reshape(-1, 1)
X_train = np.append(X_train, ones, axis=1)

X_test = np.genfromtxt('/home/kapil/Downloads/boston_x_y_train.csv', delimiter=',')

#testing data - degree 2
columns = X_test.shape[1]

# cross pairs
for j in range(columns):
    for k in range(j + 1, columns):
        new_column = (X_test[:, j] * X_test[:, k]).reshape(-1, 1)
        X_test = np.append(X_test, new_column, axis=1)

# pair with self
for i in range(columns):
    new_column = (X_test[:, i] * X_test[:, i]).reshape(-1,1)
    X_test = np.append(X_test, new_column, axis=1)

test_ones = np.ones(X_test.shape[0]).reshape(-1, 1)
X_test = np.append(X_test, test_ones, axis=1)


data.shape, X_train.shape, Y_train.shape, X_test.shape
```




    ((379, 14), (379, 105), (379,), (379, 120))




```
# Creating a data frame
df = pd.DataFrame(X_train)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>95</th>
      <th>96</th>
      <th>97</th>
      <th>98</th>
      <th>99</th>
      <th>100</th>
      <th>101</th>
      <th>102</th>
      <th>103</th>
      <th>104</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.407850</td>
      <td>-0.487722</td>
      <td>-1.266023</td>
      <td>-0.272599</td>
      <td>-0.576134</td>
      <td>1.239974</td>
      <td>0.840122</td>
      <td>-0.520264</td>
      <td>-0.752922</td>
      <td>-1.278354</td>
      <td>...</td>
      <td>0.331930</td>
      <td>1.537535</td>
      <td>0.705805</td>
      <td>0.270675</td>
      <td>0.566892</td>
      <td>1.634190</td>
      <td>0.091866</td>
      <td>0.168569</td>
      <td>1.205582</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.407374</td>
      <td>-0.487722</td>
      <td>0.247057</td>
      <td>-0.272599</td>
      <td>-1.016689</td>
      <td>0.001946</td>
      <td>-0.838337</td>
      <td>0.336351</td>
      <td>-0.523001</td>
      <td>-0.060801</td>
      <td>...</td>
      <td>1.033656</td>
      <td>0.000004</td>
      <td>0.702809</td>
      <td>0.113132</td>
      <td>0.273531</td>
      <td>0.003697</td>
      <td>0.012776</td>
      <td>0.084779</td>
      <td>0.270893</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.125179</td>
      <td>-0.487722</td>
      <td>1.015999</td>
      <td>-0.272599</td>
      <td>1.367490</td>
      <td>-0.439699</td>
      <td>0.687212</td>
      <td>-0.577309</td>
      <td>1.661245</td>
      <td>1.530926</td>
      <td>...</td>
      <td>1.870030</td>
      <td>0.193335</td>
      <td>0.472260</td>
      <td>0.333285</td>
      <td>2.759736</td>
      <td>2.343736</td>
      <td>0.650565</td>
      <td>14.408063</td>
      <td>0.794016</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.028304</td>
      <td>-0.487722</td>
      <td>1.015999</td>
      <td>-0.272599</td>
      <td>1.859875</td>
      <td>-0.047918</td>
      <td>0.801005</td>
      <td>-0.712836</td>
      <td>1.661245</td>
      <td>1.530926</td>
      <td>...</td>
      <td>3.459136</td>
      <td>0.002296</td>
      <td>0.641610</td>
      <td>0.508136</td>
      <td>2.759736</td>
      <td>2.343736</td>
      <td>0.650565</td>
      <td>0.004363</td>
      <td>0.046414</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.412408</td>
      <td>-0.487722</td>
      <td>-0.969827</td>
      <td>-0.272599</td>
      <td>-0.913029</td>
      <td>-0.384137</td>
      <td>-0.834781</td>
      <td>0.300508</td>
      <td>-0.752922</td>
      <td>-0.957633</td>
      <td>...</td>
      <td>0.833622</td>
      <td>0.147561</td>
      <td>0.696859</td>
      <td>0.090305</td>
      <td>0.566892</td>
      <td>0.917061</td>
      <td>0.000423</td>
      <td>0.185825</td>
      <td>0.000841</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 105 columns</p>
</div>




```

# Describing our data
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>95</th>
      <th>96</th>
      <th>97</th>
      <th>98</th>
      <th>99</th>
      <th>100</th>
      <th>101</th>
      <th>102</th>
      <th>103</th>
      <th>104</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>...</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>3.790000e+02</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>379.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.019628</td>
      <td>0.002455</td>
      <td>0.036170</td>
      <td>0.028955</td>
      <td>0.028775</td>
      <td>0.032202</td>
      <td>0.038395</td>
      <td>-0.001288</td>
      <td>0.043307</td>
      <td>0.043786</td>
      <td>...</td>
      <td>0.997504</td>
      <td>1.000742</td>
      <td>0.969551</td>
      <td>1.053594e+00</td>
      <td>1.031945</td>
      <td>1.039519</td>
      <td>0.998321</td>
      <td>1.029370</td>
      <td>1.028610</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.067490</td>
      <td>1.000813</td>
      <td>1.017497</td>
      <td>1.048995</td>
      <td>0.999656</td>
      <td>1.001174</td>
      <td>0.985209</td>
      <td>1.027803</td>
      <td>1.016265</td>
      <td>1.019974</td>
      <td>...</td>
      <td>1.396079</td>
      <td>1.935390</td>
      <td>1.008492</td>
      <td>1.658692e+00</td>
      <td>1.087950</td>
      <td>0.946287</td>
      <td>1.318942</td>
      <td>3.077050</td>
      <td>1.508958</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.417713</td>
      <td>-0.487722</td>
      <td>-1.516987</td>
      <td>-0.272599</td>
      <td>-1.465882</td>
      <td>-3.880249</td>
      <td>-2.335437</td>
      <td>-1.267069</td>
      <td>-0.982843</td>
      <td>-1.313990</td>
      <td>...</td>
      <td>0.001645</td>
      <td>0.000004</td>
      <td>0.000020</td>
      <td>4.799164e-07</td>
      <td>0.031727</td>
      <td>0.000269</td>
      <td>0.000423</td>
      <td>0.000012</td>
      <td>0.000003</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.408171</td>
      <td>-0.487722</td>
      <td>-0.867691</td>
      <td>-0.272599</td>
      <td>-0.878475</td>
      <td>-0.571480</td>
      <td>-0.768994</td>
      <td>-0.829872</td>
      <td>-0.637962</td>
      <td>-0.755697</td>
      <td>...</td>
      <td>0.159407</td>
      <td>0.069469</td>
      <td>0.279809</td>
      <td>1.932673e-01</td>
      <td>0.273531</td>
      <td>0.264064</td>
      <td>0.118483</td>
      <td>0.100486</td>
      <td>0.134259</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.383729</td>
      <td>-0.487722</td>
      <td>-0.180458</td>
      <td>-0.272599</td>
      <td>-0.144217</td>
      <td>-0.103479</td>
      <td>0.338718</td>
      <td>-0.329213</td>
      <td>-0.523001</td>
      <td>-0.440915</td>
      <td>...</td>
      <td>0.635784</td>
      <td>0.319303</td>
      <td>0.779327</td>
      <td>6.304826e-01</td>
      <td>0.406995</td>
      <td>0.616844</td>
      <td>0.650565</td>
      <td>0.164186</td>
      <td>0.572306</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.055208</td>
      <td>0.156071</td>
      <td>1.015999</td>
      <td>-0.272599</td>
      <td>0.628913</td>
      <td>0.529069</td>
      <td>0.911243</td>
      <td>0.674172</td>
      <td>1.661245</td>
      <td>1.530926</td>
      <td>...</td>
      <td>1.427365</td>
      <td>0.947824</td>
      <td>1.246801</td>
      <td>1.161516e+00</td>
      <td>2.759736</td>
      <td>2.343736</td>
      <td>1.283215</td>
      <td>0.194527</td>
      <td>1.246537</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.941735</td>
      <td>3.804234</td>
      <td>2.422565</td>
      <td>3.668398</td>
      <td>2.732346</td>
      <td>3.555044</td>
      <td>1.117494</td>
      <td>3.960518</td>
      <td>1.661245</td>
      <td>1.798194</td>
      <td>...</td>
      <td>7.465717</td>
      <td>15.056335</td>
      <td>5.454266</td>
      <td>1.568570e+01</td>
      <td>2.759736</td>
      <td>3.233502</td>
      <td>7.329902</td>
      <td>15.078246</td>
      <td>11.628092</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 105 columns</p>
</div>




```
def step_gradient(X_train, Y_train, learning_rate, m,j):
    # Calculate new slope for jth feature
    m_j = 0
    n_data_pts = X_train.shape[0]
    N = len(m)
    for i in range(n_data_pts):
        # calculate the formula m1xi(1)+m2xi(2)+...
        x_i = X_train[i, :]
        y_i = Y_train[i]
        temp_sum = 0
        for k in range(N):
            temp_sum += m[k]*x_i[k]
        ### sub y_i from temp sum
        temp_sum = y_i - temp_sum
        ## complete formula
        m_j += (-2/n_data_pts) * (temp_sum) * x_i[j]
    # update m[j] and return
    m[j] = m[j] - (learning_rate*m_j)
    return m[j]
```


```
def cost(X_train, Y_train, m):
    # This will calculate mean square error
    cost = 0
    n_data_pts = len(X_train)
    N = len(m)
    for i in range(n_data_pts):
        x_i = X_train[i, :]
        y_i = Y_train[i]
        temp_sum = 0
        for k in range(N):
            temp_sum += m[k]*x_i[k]
        temp_sum = y_i - temp_sum
        cost += (1/n_data_pts) * ((temp_sum)**2)
    return cost
```


```
def gradient_descent(X_train, Y_train, learning_rate, num_iterations):
    # Start with random values for all m's
    m = [0]*(X_train.shape[1])
    m[-1] = 1 #c
    N = len(m)
    x_data = []
    y_data = []
    fig = plt.figure()
    for i in range(num_iterations):
        # For all iterations do the following
        for j in range(N):
            m[j] = step_gradient(X_train, Y_train, learning_rate, m,j)
        a = cost(X_train, Y_train, m)
        x_data.append(i)
        y_data.append(a)
        plt.plot(x_data,y_data,'*')
        print("Cost - : ", i, a)
        plt.draw()
        plt.pause(1e-17)
        time.sleep(0.1)
    return m
```


```

def predict(X_test, m):
    n_fts = X_train.shape[1]
    n_m = np.array(m).reshape(n_fts, 1)
    return np.dot(X_test, n_m)
```


```
df_test = pd.DataFrame(X_test)
df_test.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>110</th>
      <th>111</th>
      <th>112</th>
      <th>113</th>
      <th>114</th>
      <th>115</th>
      <th>116</th>
      <th>117</th>
      <th>118</th>
      <th>119</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>...</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>3.790000e+02</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>379.000000</td>
      <td>379.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.019628</td>
      <td>0.002455</td>
      <td>0.036170</td>
      <td>0.028955</td>
      <td>0.028775</td>
      <td>0.032202</td>
      <td>0.038395</td>
      <td>-0.001288</td>
      <td>0.043307</td>
      <td>0.043786</td>
      <td>...</td>
      <td>1.000742</td>
      <td>0.969551</td>
      <td>1.053594e+00</td>
      <td>1.031945</td>
      <td>1.039519</td>
      <td>0.998321</td>
      <td>1.029370</td>
      <td>1.028610</td>
      <td>599.122269</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.067490</td>
      <td>1.000813</td>
      <td>1.017497</td>
      <td>1.048995</td>
      <td>0.999656</td>
      <td>1.001174</td>
      <td>0.985209</td>
      <td>1.027803</td>
      <td>1.016265</td>
      <td>1.019974</td>
      <td>...</td>
      <td>1.935390</td>
      <td>1.008492</td>
      <td>1.658692e+00</td>
      <td>1.087950</td>
      <td>0.946287</td>
      <td>1.318942</td>
      <td>3.077050</td>
      <td>1.508958</td>
      <td>530.692365</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.417713</td>
      <td>-0.487722</td>
      <td>-1.516987</td>
      <td>-0.272599</td>
      <td>-1.465882</td>
      <td>-3.880249</td>
      <td>-2.335437</td>
      <td>-1.267069</td>
      <td>-0.982843</td>
      <td>-1.313990</td>
      <td>...</td>
      <td>0.000004</td>
      <td>0.000020</td>
      <td>4.799164e-07</td>
      <td>0.031727</td>
      <td>0.000269</td>
      <td>0.000423</td>
      <td>0.000012</td>
      <td>0.000003</td>
      <td>25.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.408171</td>
      <td>-0.487722</td>
      <td>-0.867691</td>
      <td>-0.272599</td>
      <td>-0.878475</td>
      <td>-0.571480</td>
      <td>-0.768994</td>
      <td>-0.829872</td>
      <td>-0.637962</td>
      <td>-0.755697</td>
      <td>...</td>
      <td>0.069469</td>
      <td>0.279809</td>
      <td>1.932673e-01</td>
      <td>0.273531</td>
      <td>0.264064</td>
      <td>0.118483</td>
      <td>0.100486</td>
      <td>0.134259</td>
      <td>278.890000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.383729</td>
      <td>-0.487722</td>
      <td>-0.180458</td>
      <td>-0.272599</td>
      <td>-0.144217</td>
      <td>-0.103479</td>
      <td>0.338718</td>
      <td>-0.329213</td>
      <td>-0.523001</td>
      <td>-0.440915</td>
      <td>...</td>
      <td>0.319303</td>
      <td>0.779327</td>
      <td>6.304826e-01</td>
      <td>0.406995</td>
      <td>0.616844</td>
      <td>0.650565</td>
      <td>0.164186</td>
      <td>0.572306</td>
      <td>445.210000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.055208</td>
      <td>0.156071</td>
      <td>1.015999</td>
      <td>-0.272599</td>
      <td>0.628913</td>
      <td>0.529069</td>
      <td>0.911243</td>
      <td>0.674172</td>
      <td>1.661245</td>
      <td>1.530926</td>
      <td>...</td>
      <td>0.947824</td>
      <td>1.246801</td>
      <td>1.161516e+00</td>
      <td>2.759736</td>
      <td>2.343736</td>
      <td>1.283215</td>
      <td>0.194527</td>
      <td>1.246537</td>
      <td>663.265000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.941735</td>
      <td>3.804234</td>
      <td>2.422565</td>
      <td>3.668398</td>
      <td>2.732346</td>
      <td>3.555044</td>
      <td>1.117494</td>
      <td>3.960518</td>
      <td>1.661245</td>
      <td>1.798194</td>
      <td>...</td>
      <td>15.056335</td>
      <td>5.454266</td>
      <td>1.568570e+01</td>
      <td>2.759736</td>
      <td>3.233502</td>
      <td>7.329902</td>
      <td>15.078246</td>
      <td>11.628092</td>
      <td>2500.000000</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 120 columns</p>
</div>




```
def run():
    # Intialize parameters for gradient descent
    num_iterations = 1200
    learning_rate = 0.01
    m = gradient_descent(X_train, Y_train, learning_rate, num_iterations)
    print(m)
    Y_pred = predict(X_test, m)
    np.savetxt('y_pred.csv', Y_pred, fmt='%.12f')
    df.to_csv(path=/homw/kapil/Downloads, columns=one,  index=True,)
    

      
```


```
run()
```
