#### Introduction

This web application tool possesses the ability to analyze satellite images of a specific geographic area and identify locations where solar panels are present. Its functionalities include:

1. The web dashboard allows users to input two sets of geo-coordinates, defining a rectangular area. Upon submission, the program scans all satellite images within this region and archives all images indicating the presence of solar panels in a designated folder.
2. Initially, these positive identifications are marked as unconfirmed. Within the dashboard, there's a tab where users can view these unconfirmed positive images. Users have the option to verify and confirm these identifications as either positive or negative, correcting any errors made by the program.
3. Subsequently, all confirmed positive images are cross-referenced with a database of registered businesses to identify companies that have installed solar panels. The resulting list of companies, along with relevant information, is presented to the user for further action, such as contacting the companies.

#### Installation

1. First you need an ubuntu 20.04 with a GPU with VRAM > 10GB.
2. Create a conda environment with python version of 3.10.0
3. Install the following packages

```
torch
flask
redis
leafmap
pandas
numpy
opencv-python
rasterio
einops
PIL
matplotlib
```

4. use this command to install and start a redis server:

``` shell
sudo apt install redis
```

#### How to deploy

Firstly you need to configure the redis database. The default configuration is fine, but you can change the disk persistency frequency if you want. You can connect to the database using the following command:
``` shell
redis-cli
```

Next, you need the trained model files. Store these files in any folder and change their path in config.py file. 

In order to start the flask application run the following command:
```shell
python -m flask --app webapp/app run --host 0.0.0.0 --port 5000
```

There is a .service file in this repo that is used for creating a systemctl service for this application. Make the necessary changes in this file and move it to ```/etc/systemd/system/``` directory.


