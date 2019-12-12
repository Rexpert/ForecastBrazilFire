# ForecastBrazilFire
This repository stored all resources to deploy our Forecasting subject's academic paper.

## Project Structure
Below showing the details and purposes of some important files and folders in this project:

| folder                      | description                                                                                        |
| :-------------------------- | :------------------------------------------------------------------------------------------------- |
| `data`                      | Folder raw data (`.csv`) and shapefiles (`.shp`)                                                   |
| `data/brazil-shapefile`     | Folder contains all shape files to plot map in the `app.ipynb`                                     |
| `maps`                      | Folder contains all map outputs by `app.ipynb`                                                     |
| `maps/*.png`                | Map outputs by Python                                                                              |
| `maps/magicksmap.gif`       | Animated Gif output by ImageMagick                                                                 |
| `TODO.md`                   | Markdown contains all TODO tasks                                                                   |
| `app.ipynb`                 | Main file that contains code for the analysis                                                      |
| `environment.yml`           | Conda enviroment list                                                                              |
| `evaluation.csv`            | SARIMA results output from `app.ipynb`                                                             |
| `testing.csv`               | Cleaned data to be testing on Microsoft Excel                                                      |

## Local Development
Below are some guidance for develop this project locally. Before that, there are some dependencies for `Python` you would need to install prior for any script execution.

### Installation for `ImageMagick`
`ImageMagick` is needed to output the animated GIF. You can download [it](https://imagemagick.org/download/binaries/ImageMagick-7.0.9-8-Q16-x64-dll.exe) from thier [official website](https://imagemagick.org/script/download.php#windows)

### Package Installation for `Python`  
`Python3` is needed to execute `app.ipynb`. If you are using `conda`, you can install all the needed packages by running the following commands on project root:
```sh
conda env create -f environment.yml
```
This will create a new `forecasting` environment in `Python` and install all the dependencies needed in `app.ipynb` script (e.g. `Geopandas`).

### Get Started with Analysis
After installed the environment, you need to activate it by running the following commands and open the `Jupyter Notebook` on the project root:
```sh
conda activate forecasting
jupyter notebook
```

## License

MIT © 2019 [Neoh HaiLiang](https://github.com/Rexpert)


[website]: https://nbviewer.jupyter.org/github/Rexpert/ForecastBrazilFire/blob/master/app.ipynb
