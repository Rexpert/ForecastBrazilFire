import math
from glob import glob
from itertools import repeat
from os.path import join
from subprocess import call

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from geopandas import read_file
from IPython.display import display
from statsmodels.api import OLS, add_constant
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

sns.set()

# Parse df to the function
df = pd.read_csv("./data/amazon.csv", encoding="ISO-8859-1", thousands=".")

# Read map data: shape file for Brazil
map_df = read_file("./data/brazil-shapefile/Central-West Region_AL3-AL4.shp")
map_df = map_df.iloc[5:, :]

'''# Temporary Disable
# Understand the data structure
print("The data has " + str(df.shape[0]) + " rows, and " + str(df.shape[1]) + " columns\n\n")
print("Overview on the data type of each column: \n")
df.info()
print("\n\n")
print("Check for Missing Values: \n")
df.isna().sum()  # check for Na
print("\n\n")
df.head()
df.describe()
df["month"].nunique()
df["state"].nunique()
'''

# 1. Date: Possible Duplicate Column
# Check if date is a duplicated column with year
year = df.iloc[:, 0]
date = df.iloc[:, -1]
date = date.str.extract(r"(\d+)")  # Extract year from date
date = date.iloc[:, 0]            # Transform to Series type
date = date.astype(year.dtype)    # Convert String to int

not_same = (date != year)
not_same.any()                    # All years in date is same as year column

# Check if month & day is all equal 1
date = df.iloc[:, -1]
date = date.str.extract(r"-(\d+)-(\d+)")  # Extract month & day from date
date = date.applymap(int)                # Convert String to int
not1 = date != 1
not1.any()                               # All values in (column 0 & 1) is 1

# The date column is duplicated, hence it is allowed to drop date column
df = df.iloc[:, 0:4]

# 2. Deal with time column
month = list(range(1, 13))
foo = pd.DataFrame({"month": df.iloc[:, 2].unique(), "mmm": month})
df = pd.merge(df, foo, on="month", how="left")
df["period"] = df.apply(lambda t: pd.Period(
    year=t["year"], month=t["mmm"], freq="M"), axis=1)
df = df[["period", "state", "number"]]


# 3. Duplicate Data
state_por = map_df.iloc[:, 2]
state_eng = df.iloc[:, 2].unique()
(len(state_por), len(state_eng))  # different length

# detect the duplication by counting the entries
df["state"].value_counts()

# Most of the state has 239 entries, so there may exist some duplicate entries
# By comparing with state_por, the following entries are label wrongly:
# Rio <- Rio de Janeiro
# Rio <- Rio Grande do Norte
# Rio <- Rio Grande do Sul
# Mato Grosso <- Mato Grosso
# Mato Grosso <- Mato Grosso do Sul
# Paraiba <- ParaÃ­ba
# Paraiba <- ParanÃ¡
# Hence the actual count of entry for these state are stated below:
# Rio           = 239 * 3 = 717
# Mato Grosso   = 239 * 2 = 478
# Paraiba       = 239 * 2 = 478
# The only duplication occured in Alagoas:
d = df[df.state == "Alagoas"]
duplicate = d[d.duplicated()].index

df = df.drop(index=duplicate, axis=1)
df["state"].value_counts()  # Check for deletion


# 4. State: Replace with actual portugese state name
# Check for order in df.state
df.state.unique()
state_por

# Found inconsistent order in df.state compared to state_por
# The order for Rio, Mato Grosso and Paraiba was assumed followed alphabetical order
# The order EspÃ­rito Santo & Federal District should be reversed:
index = list(range(27))
index[6], index[7] = index[7], index[6]
state_por = state_por.iloc[index]

df.state = [x for item in state_por for x in repeat(item, 239)]


# Finding mean of each month in a year instead of total up all month, because there are missing value for Dec 2017
wide_df = df.copy()
wide_df["year"] = wide_df.period.map(lambda t: t.year)
wide_df = wide_df.groupby(["year", "state"])
wide_df = wide_df.sum().unstack(level=0)
wide_df.columns = wide_df.columns.droplevel()
wide_df = wide_df.reset_index()

'''# Temporary Disable
# Try plot map with 1 example
left = map_df.copy()
right = wide_df.copy().iloc[:,0:2]
right.rename(columns={"state" : "name"}, inplace=True)
plot_df = pd.merge(left, right, on="name", how="left")

vmin = wide_df.iloc[:,1:].min().min()
vmin = int(math.floor(vmin/1000))*1000
vmax = wide_df.iloc[:,1:].max().max()
vmax = int(math.ceil(vmax/1000))*1000

year = 1998
fig = plot_df.plot(column=year, cmap="Purples", figsize=(10, 10), linewidth=0.8, edgecolor="0.8", vmin=vmin, vmax=vmax, legend=True, norm=plt.Normalize(vmin=vmin, vmax=vmax))

fig.axis("off")
fig.set_title('Brazil Forest Fire', fontdict={'fontsize': '25', 'fontweight' : '3'})

x_coord = 2 * year - 4071
fig.plot(x_coord, -36, "o", color="#69549E", markersize=20, alpha=0.5, clip_on=False, zorder=100)
fig.text(x_coord, -39, str(year), horizontalalignment="center", fontsize=15)

plt.ylim(-35,7)
plt.xlim(-75,-34)

for n in range(1, 20) :
    x_coord = -75 + (n - 1) * 2
    color = '#69549E' if n % 2 else '#E8E7EF'
    plt.hlines(-36, x_coord, x_coord + 2, colors=color, lw=5, clip_on=False, zorder=100)

plt.show()
'''

'''# Temporary Disable
# Plot multiple maps
left = map_df.copy()
right = wide_df.copy()
right.rename(columns={"state" : "name"}, inplace=True)
plot_df = pd.merge(left, right, on="name", how="left")

vmin = wide_df.iloc[:,1:].min().min()
vmin = int(math.floor(vmin/1000))*1000
vmax = wide_df.iloc[:,1:].max().max()
vmax = int(math.ceil(vmax/1000))*1000

year_list = wide_df.columns[1:]

for year in year_list :
    fig = plot_df.plot(column=year, cmap="Purples", figsize=(10, 10), linewidth=0.8, edgecolor="0.8", vmin=vmin, vmax=vmax, legend=True, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    
    fig.axis("off")
    fig.set_title('Brazil Forest Fire', fontdict={'fontsize': '25', 'fontweight' : '3'})
    
    x_coord = 2 * year - 4071
    fig.plot(x_coord, -36, "o", color="#69549E", markersize=20, alpha=0.5, clip_on=False, zorder=100)
    fig.text(x_coord, -39, str(year), horizontalalignment="center", fontsize=15)

    plt.ylim(-35,7)
    plt.xlim(-75,-34)

    for n in range(1, 20) :
        x_coord = -75 + (n - 1) * 2
        color = '#69549E' if n % 2 else '#E8E7EF'
        plt.hlines(-36, x_coord, x_coord + 2, colors=color, lw=5, clip_on=False, zorder=100)

    filepath = join("maps", str(year)+'.png')
    chart = fig.get_figure()
    chart.savefig(filepath, dpi=300)

    plt.close()

print("done")
'''

'''# Temporary Disable
# Output images to Gif using image magick
imgs = glob('*.png')

cmd = ['magick','convert', '-loop', '0', '-delay', '40'] + imgs + ['magicksmap.gif']
call(cmd)
'''

# finalise cleaning
df = df.groupby(["period"]).sum()
# df.to_csv(r'testing.csv')

# Decomposition plot
result = seasonal_decompose(df, model='additive', freq=12)
result.plot()
plt.show()

# Plot of forest fire
df.plot()
plt.title("Monthly Report on Number of Brazil Forest Fire")
plt.show()

# In table form
wide_df = df.copy().reset_index()
wide_df["year"] = wide_df.period.map(lambda t: t.year)
wide_df["month"] = wide_df.period.map(lambda t: t.month)
wide_df = wide_df.groupby(["year", "month"])
wide_df = wide_df.sum().unstack()
wide_df.columns = wide_df.columns.droplevel()

# Overall Trend
by_year = df.copy().reset_index()
by_year["year"] = by_year.period.map(lambda t: t.year)
by_year = by_year.groupby(["year"]).sum().plot()
plt.title("Yearly Report on Number of Brazil Forest Fire")
plt.xticks(np.arange(1998, 2017, 2))
plt.show()

# Seasonal of forest fire
seed = 111
wide_df.sample(n=4, random_state=seed).T.plot()
plt.title("Seasonal Plot: Brazil Forest Fire (4 samples)")
plt.show()

# ACF plot / PACF plot (Correlogram)
fig, ax = plt.subplots(2, figsize=(12, 6))
ax[0] = plot_acf(df, ax=ax[0], lags=60)
ax[1] = plot_pacf(df, ax=ax[1], lags=60)
plt.show()

# Train-Test Split
split_index = pd.Period("2016-12", "M")


def MSE(a):
    return np.square(np.subtract(a["y_t"], a["F_t"])).mean()


error = pd.DataFrame()

# Naive: Seasonal Naive
new_df = df.copy()
new_df.columns = ["y_t"]
new_df["F_t"] = new_df["y_t"].shift(12)
new_df = new_df.dropna()

train = new_df[:split_index.strftime("%Y-%m")]
test = new_df[(split_index+1).strftime("%Y-%m"):]
plt.figure(figsize=(16, 8))
new_df["y_t"].plot(label="Original Data")
test["F_t"].plot(label="Seasonal Naive")
plt.show()

method = "Seasonal Naive"
if method in error["method"].values:
    error = error[error.method != method]
error = error.append(dict(method=method, train=MSE(train),
                          test=MSE(test)), ignore_index=True)

# Exponential Smoothing: Holt-Winter's
new_df = df.copy()
new_df.columns = ["y_t"]

train = new_df[:split_index.strftime("%Y-%m")]
test = new_df[(split_index+1).strftime("%Y-%m"):]

fit1 = ExponentialSmoothing(np.asarray(
    train['y_t']), seasonal_periods=12, trend='add', seasonal='add',).fit()

train["F_t"] = fit1.fittedvalues
test["F_t"] = fit1.forecast(len(test))

plt.figure(figsize=(16, 8))
new_df["y_t"].plot()
test["F_t"].plot()
plt.show()

print(fit1.summary())

method = "Holt-Winter's"
if method in error["method"].values:
    error = error[error.method != method]
error = error.append(dict(method=method, train=MSE(train),test=MSE(test)), ignore_index=True)

# Decomposition Method
new_df = df.copy()
new_df.columns = ["y_t"]

train = new_df[:split_index.strftime("%Y-%m")]
test = new_df[(split_index+1).strftime("%Y-%m"):]

deseasonal = new_df.iloc[:, 0] - result.seasonal.iloc[:, 0]

deseasonal.plot()
plt.show()

train["deseasonal"] = deseasonal[:split_index.strftime("%Y-%m")]
fit2 = OLS(train["deseasonal"], add_constant(range(len(train)))).fit()
fit2.summary()

train["F_t"] = fit2.fittedvalues + result.seasonal.iloc[:len(train), 0]
test["F_t"] = fit2.predict(add_constant(
    range(len(train), len(new_df)))) + result.seasonal.iloc[len(train):, 0]

plt.figure(figsize=(16, 8))
new_df["y_t"].plot()
test["F_t"].plot()
plt.show()

method = "Decomposition"
if method in error["method"].values:
    error = error[error.method != method]
error = error.append(dict(method=method, train=MSE(train),test=MSE(test)), ignore_index=True)

# Time Series Regression
new_df = df.copy()
new_df.columns = ["y_t"]

dummies = pd.get_dummies(df.index.month, drop_first=True)
dummies["t"] = range(len(df))
dummies.index = new_df.index

train = new_df[:split_index.strftime("%Y-%m")]
test = new_df[(split_index+1).strftime("%Y-%m"):]
train_dummy = dummies[:split_index.strftime("%Y-%m")]
test_dummy = dummies[(split_index+1).strftime("%Y-%m"):]

fit3 = OLS(train, add_constant(train_dummy)).fit()
fit3.summary()

train["F_t"] = fit3.fittedvalues
test["F_t"] = fit3.predict(add_constant(test_dummy))

plt.figure(figsize=(16, 8))
new_df["y_t"].plot()
test["F_t"].plot()
plt.show()

method = "Time-Series Regression"
if method in error["method"].values:
    error = error[error.method != method]
error = error.append(dict(method=method, train=MSE(train),test=MSE(test)), ignore_index=True)
