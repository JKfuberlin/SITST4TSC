import os # for general operating system functionality
import glob # for retrieving files using strings/regex
import re # for regex
import rasterio # for reading rasters
import geopandas as gpd # for reading shapefiles
import numpy as np
import datetime # for benchmarking
import torch # for loading the model and actual inference
from torch import nn
import rioxarray as rxr # for raster clipping
from shapely.geometry import mapping # for clipping
from rasterio.transform import from_origin # for assigning an origin to the created map
import pandas as pd

def predict(pixel): # data_for_prediction should be a tensor of a single pixel
    model_pkl.eval()  # set model to eval mode to avoid dropout layer
    pixel.to(device)
    with torch.no_grad():  # do not track gradients during forward pass to speed up
        output_probabilities = model_pkl(pixel)  # prediction
        _, predicted_class = torch.max(output_probabilities,1)  # retrieving the class with the highest probability after softmax
    return predicted_class

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Device configuration
All = False
GROMIT = True
EOLAB = False

if All == True:
    print("dall'inizio")
    model_pkl = torch.load('/point_storage/data/Transformer_1.pkl', map_location=torch.device('cpu'))  # loading the trained model
    raster_paths = []
    #tile ="X0066_Y0056"
    tile ="X0067_Y0058" # Landshut
    s2dir = glob.glob(('/force/FORCE/C1/L2/ard/' + tile + '/'), recursive=True)  # Initialize path to Sentinel-2 time series
    raster_paths = []  # create object to be filled with rasters to be stacked
    tifs = glob.glob(os.path.join(s2dir[0], "*SEN*BOA.tif"))  # i want to ignore the landsat files and stack only sentinel 2 bottom of atmosphere observations
### get dates
    dates = [pd.to_datetime(s[35:43], format='%Y%m%d') for s in tifs]
    dates = pd.to_datetime(dates, format="%Y%m%d").sort_values() # sort ascending
    comparison_date = pd.to_datetime('20230302', format="%Y%m%d")
# Filter dates to exclude entries before 20230302
    dates = dates[dates <= comparison_date]
# these dates are the DOY that need to be passed to the forward method
### here, we need to retrieve and assign the correct DOY values
# get day of the year of startDate
    t0 = dates[0].timetuple().tm_yday
    input_dates = np.array([(date - dates[0]).days for date in dates]) + t0
    seq_len = len(input_dates)
    doy = np.zeros((seq_len,), dtype=int)
    DOY = input_dates
    raster_paths.extend(tifs)  # write the ones i need into my object to stack from
    years = [int(re.search(r"\d{8}", raster_path).group(0)) for raster_path in raster_paths]  # i need to extract the date from each file path...
    raster_paths = [raster_path for raster_path, year in zip(raster_paths, years) if year <= 20230302]  # ... to cut off at last image 20230302 as i trained my model with this sequence length, this might change in the future with transformers
    datacube = [rxr.open_rasterio(raster_path) for raster_path in raster_paths]  # i user rxr because the clip function from rasterio sucks
    # datacube.append(doy)# the datacube is too large for the RAM of my contingent so i need to subset using the 5x5km tiles
# grid_tiles = glob.glob(os.path.join('/my_volume/FORCE_tiles_subset_BW/', '*' + tile + '*.gpkg'))
    grid_tiles = '/point_storage/landshut_minibatch.gpkg'
# checking workflow:
    minitile = grid_tiles
    crop_shape = gpd.read_file(minitile)
    clipped_datacube = [] # empty object for adding clipped rasters
    i = 1
    total = str(len(datacube))
    for raster in datacube:  # clip the rasters using the geometry
        crop_shape = gpd.read_file(minitile)
        print('cropping ' + str(i) + ' out of ' + total)
        i = i + 1
        try:
            clipped_raster = raster.rio.clip(crop_shape.geometry.apply(mapping))
            clipped_datacube.append(clipped_raster)
        except:
            print('not working')
            break
if GROMIT == True:
    print('running on gromit')
    model_pkl = torch.load('/home/j/data/outputs/models/Transformer_2.pkl', map_location=torch.device(device))
    clipped_datacube = np.load('/home/j/data/landshut_cropped_dc.npy')
    DOY = pd.read_csv('/home/j/data/doy_pixel_subset.csv', sep = '\t', header = None)
if EOLAB == True:
    DOY = pd.read_csv('/point_storage/data/doy_pixel_subset.csv', sep='\t', header=None)
    clipped_datacube = np.load('/point_storage/data/landshut_cropped_dc.npy')
    model_pkl = torch.load('/point_storage/data/Transformer_1.pkl', map_location=device)
    crop_shape = gpd.read_file('/point_storage/landshut_minibatch.gpkg')

print('DOY, model and datacube loaded')

DOY = np.array(DOY)
datacube_np = np.array(clipped_datacube, ndmin = 4) # this is now a clipped datacube for the first minitile, fixing it to be 4 dimensions
doy_reshaped = DOY.reshape((329, 1, 1, 1)) # Reshape doy to have a new axis
doy_reshaped = np.repeat(doy_reshaped, 500, axis=2) # Repeat the doy values along the third and fourth dimensions to match datacube_np
doy_reshaped = np.repeat(doy_reshaped, 500, axis=3)
datacube_np = np.concatenate((datacube_np, doy_reshaped), axis=1) # Concatenate along the second axis
# datacube_np.shape #for inspection if necessary
# rearrange npy array
datacube_rearranged = np.transpose(datacube_np, (2, 3, 0, 1))
num_bands = datacube_rearranged.shape[3] # retrieving numer of bands
batch_norm = nn.BatchNorm1d(num_bands)  # Create a BatchNorm1d layer with `num_bands` as the number of input features.
x = datacube_rearranged.shape[0]-1
y = datacube_rearranged.shape[1]-1
normalized_inference_datacube = np.zeros((x, y, 329, 11))
for row in range(x): # assuming, data_for_prediction is a x*y raster
    print(row)
    print(datetime.datetime.now())
    for col in range(y):
        pixel = datacube_rearranged[row, col, :,:] # get the pixel timeseries
        pixel[pixel == -9999] = np.nan # FORCE NA value is -9999 but the model expects np.nan
        pixel_torch16 = torch.tensor(pixel)  # turn the numpy array into a pytorch tensor, the result is in int16..
        pixel_torch32 = pixel_torch16.to(torch.float32) # ...so we need to transfer it to float32 so that the model can use it as input
        pixel_normalized = batch_norm(pixel_torch32).detach().numpy()# run the normalization on the tensor. detach because i don't want gradients from batch_norm. #WTF why zack to numpy?
        normalized_inference_datacube[row, col, :] = pixel_normalized
        normalized_inference_datacube[row:row + 329, col:col + 11, :] = pixel_normalized

data_for_prediction = normalized_inference_datacube.to(device) # pass data to GPU
print('starting for loop prediction')
result = np.zeros((x, y, 1))
for row in range(x):
    print(row)
    print(datetime.datetime.now())
    for col in range(y): # TODO: instead of predicting single pixels, maybe predict entire rows/columns using the data loader
        pixel = data_for_prediction[x, y, :, :] # select pixel at this position # TODO: verify x and y to avoid 90 degrees rotation
        predicted_class = predict(pixel)
        predicted_class = predicted_class.cpu().numpy()
        result[row, col, :] = predicted_class
map = result
print('loop done')

crop_geometry = crop_shape.geometry.iloc[0]
crop_bounds = crop_geometry.bounds
origin = (crop_bounds[0], crop_bounds[3])  # Extract the origin from the bounds

# construct metadata from minitile and matrix
metadata = {
    'driver': 'GTiff',
    'width': map.shape[1],
    'height': map.shape[0],
    'count': 1,  # Number of bands
    'dtype': map.dtype,
    'crs': 'EPSG:3035',  # Set the CRS (coordinate reference system) code as needed
    'transform': from_origin(origin[0], origin[1], 10, 10)  # Set the origin and pixel size (assumes each pixel is 1 unit)
}

print('writing')
with rasterio.open(os.path.join('/point_storage/', 'landshut_transformer.tif'), 'w', **metadata) as dst:
    # dst.write_band(1, map.astype(rasterio.float32))
    dst.write(map.astype(rasterio.float32), indexes=1)
print('written')

# map_reshaped = map.transpose(1, 2, 0, 3)
#
# with rasterio.open(os.path.join('/point_storage/', 'landshut_transformer.tif'), 'w', **metadata) as dst:
#     for i in range(map_reshaped.shape[3]):
#         dst.write_band(i + 1, map_reshaped[:, :, :, i].astype(rasterio.float32))
