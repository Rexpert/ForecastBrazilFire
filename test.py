from itertools import repeat
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import math
import os

# Parse df to the function
df = pd.read_csv("./data/amazon.csv", encoding="ISO-8859-1")

# Read map data: shape file for Brazil
map_df = gpd.read_file("data/brazil-shapefile/Central-West Region_AL3-AL4.shp")

''' Temporary Disable
# Understand the data structure
df.shape
df.info()
df.isna().sum()  # check for Na
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

not_same = date != year
not_same.any()                    # All years in date is same as year column

# Check if month & day is all equal 1
date = df.iloc[:, -1]
date = date.str.extract(r"-(\d+)-(\d+)")  # Extract month & day from date
date = date.applymap(int)                # Convert String to int
not1 = date != 1
not1.any()                               # All values in (column 0 & 1) is 1

# The date column is duplicated, hence it is allowed to drop date column
df = df.iloc[:, 0:4]


# 2. Month: Portugese -> Number
month = list(range(1, 13))
month = list(map(lambda x: str(x).zfill(2), month))
foo = pd.DataFrame({"month": df.iloc[:, 2].unique(),
                    "mmm": month})

df = pd.merge(df, foo, on="month", how="left")
df = df[["year", "mmm", "state", "number"]]
df.columns = ["year", "month", "name", "number"]

# 3. Duplicate Data
map_df = map_df.iloc[5:, :]
state_por = map_df.iloc[:, 2]
state_eng = df.iloc[:, 2].unique()
(len(state_por), len(state_eng))  # different length

# detect the duplication by counting the entries
df["name"].value_counts()

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
d = df[df.name == "Alagoas"]
duplicate = d[d.duplicated()].index

df = df.drop(index=duplicate, axis=1)
df["name"].value_counts()  # Check for deletion

# 4. State: Replace with actual portugese name
# Check for order in df.state
df.name.unique()
state_por

# Found inconsistent order in df.state compared to state_por
# The order for Rio, Mato Grosso and Paraiba was assumed followed alphabetical order
# The EspÃ­rito Santo & Federal District should be reversed:
index = list(range(27))
index[6], index[7] = index[7], index[6]
state_por = state_por.iloc[index]

df.name = [x for item in state_por for x in repeat(item, 239)]

'''# Try plot map with 1 example
right = df.query("year == 2017 and month == '11'")[["name", "number"]]
plot_df = map_df.copy()
plot_df = pd.merge(plot_df, right, on = "name", how = "left")

fig = plot_df.plot(column = "number", cmap = "Purples", figsize = (10, 10), linewidth = 0.8, edgecolor = "0.8", vmin = 0, vmax = 900, legend = True, norm = plt.Normalize(vmin = 0, vmax = 900))

## Legend with no upper bound
# fig = plot_df.plot(column = "number", cmap = "Purples", figsize = (10, 10), linewidth = 0.8, edgecolor = "0.8", vmin = 0, vmax = 900, norm = plt.Normalize(vmin = 0, vmax = 900))
# ax = plt.gca()
# im = ax.imshow(np.arange(0,1000).reshape(10,100), cmap = "Purples")
# plt.colorbar(im,fraction=0.046, pad=0.04)

fig.axis("off")
fig.set_title('Brazil Forest Fire', fontdict={'fontsize': '25', 'fontweight' : '3'})

mon = int(str(201711)[4:])
year = str(201711)[:4]
eng_month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
annotation = eng_month[mon-1] + " " + year
x_coord = 2 * (int(year) + mon/12) - 4071
fig.annotate(annotation, xy=(.3, .4), xycoords='figure fraction', horizontalalignment='right',  verticalalignment ='top', fontsize=35)
fig.plot(x_coord, -36, "o", color = "#69549E", markersize = 20, alpha = 0.5, clip_on=False, zorder = 100)
fig.text(x_coord, -39, annotation, horizontalalignment = "center", fontsize = 15)

plt.ylim(-35,7)
plt.xlim(-75,-34)

for n in range(1, 21) :
    x_coord = -75 + (n - 1) * 2
    color = '#69549E' if n % 2 else '#E8E7EF'
    plt.hlines(-36, x_coord, x_coord + 2, colors=color, lw=5, clip_on = False, zorder = 100)

plt.show()
'''

## Plot multiple maps
vmin = int(math.floor(min(df.number)/100))*100
vmax = int(math.ceil(max(df.number)/100))*100

df["time"] = df.year.astype(str) + df.month
df["time"] = df.time.astype(int)
df = df.sort_values("time")

wide_df = df.copy()
wide_df = wide_df[["time", "name", "number"]]
wide_df = wide_df.pivot(index="name", columns="time", values="number")

plot_df = pd.merge(map_df, wide_df, on="name", how="left")

year_list = wide_df.columns

eng_month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
for now in year_list :
    fig = plot_df.plot(column = now, cmap = "Purples", figsize = (10, 10), linewidth = 0.8, edgecolor = "0.8", vmin = vmin, vmax = vmax, legend = True, norm = plt.Normalize(vmin = vmin, vmax = vmax))

    fig.axis("off")
    fig.set_title('Brazil Forest Fire', fontdict={'fontsize': '25', 'fontweight' : '3'})

    mon = int(str(now)[4:])
    year = str(now)[:4]
    annotation = eng_month[mon-1] + " " + year
    x_coord = 2 * (int(year) + mon/12) - 4071
    fig.annotate(annotation, xy=(.3, .4), xycoords='figure fraction', horizontalalignment='right',  verticalalignment ='top', fontsize=35)
    fig.plot(x_coord, -36, "o", color = "#69549E", markersize = 20, alpha = 0.5, clip_on=False, zorder = 100)
    fig.text(x_coord, -39, annotation, horizontalalignment="center", fontsize=15)

    plt.ylim(-35,7)
    plt.xlim(-75,-34)

    for n in range(1, 21) :
        x_coord = -75 + (n - 1) * 2
        color = '#69549E' if n % 2 else '#E8E7EF'
        plt.hlines(-36, x_coord, x_coord + 2, colors=color, lw=5, clip_on = False, zorder = 100)

    filepath = os.path.join("maps", str(now)+'.png')
    chart = fig.get_figure()
    chart.savefig(filepath, dpi=300)

    plt.close()

print("done")



from PIL import Image
import glob
 
# Create the frames
frames = []
imgs = glob.glob("maps/*.png")[:5]
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)
 
# Save into a GIF file that loops forever
frames[0].save('png_to_gif.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=150, loop=0)