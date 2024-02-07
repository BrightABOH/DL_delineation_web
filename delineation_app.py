import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from rasterio import mask

import glob
import csv
#from s2_s1_fusion import total_water_area
import os

import torch
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date

from shapely.ops import unary_union

import rasterio
from werkzeug.utils import secure_filename
import geopandas as gpd
from shapely.geometry import shape
from sentinelhub import BBox, CRS, DataCollection, SentinelHubRequest, MimeType, SHConfig
from rasterio.plot import show
from torchvision import transforms
from PIL import Image
import numpy as np

import tarfile
import shutil
from flask import Flask, request, redirect, url_for, render_template,jsonify,send_from_directory, send_file, after_this_request
import geopandas as gpd
from datetime import datetime






area_per_pixel_m2 = 100  # Area represented by each pixel in square meters

# Conversion factor from square meters to hectares
m2_to_hectares = 1e-0

# Set your Sentinel Hub credentials
config = SHConfig()
config.sh_client_id = 'your sh_client'  # Add your Sentinel Hub instance ID
config.sh_client_secret = 'your client_secret'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Set the path to your shapefile



#image size to be downloaded
image_size = (512, 512)



# Define the mean and standard deviation values for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Load the saved model
loaded_model = smp.Unet(encoder_name='resnet50', in_channels=3, classes=1, activation=None).to(device)
loaded_model.load_state_dict(torch.load('segmentation_model_resnet50.pth'))
loaded_model.eval()

cropland_threshold = 0.7  # Set your segmentation threshold for considering cropland(<=0.7 is ideal)

import os
import fiona
import geopandas as gpd
from flask import Flask, render_template, request
from fiona.env import Env



app = Flask(__name__)


# Define the load_shapefile function

import geopandas as gpd
from fiona import Env




# Continue with the rest of your code...

    # Continue with the rest of your code...

# Set up Sentinel Hub API
bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "SCL", "B11", "B12","SCL"]
bands_s1 = ['VV', 'VH']
l_band = ["B02", "B03", "B04","B05"]
api = SentinelAPI(config.sh_client_id, config.sh_client_secret, 'https://scihub.copernicus.eu/dhus')
evalscript = """
//VERSION=3
function setup() {
    return {
        input: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12","SCL"],
        output: [
            { id: "B01", bands: 1, sampleType: SampleType.AUTO },
            { id: "B02", bands: 1, sampleType: SampleType.AUTO },
            { id: "B03", bands: 1, sampleType: SampleType.AUTO },
            { id: "B04", bands: 1, sampleType: SampleType.AUTO },
            { id: "B05", bands: 1, sampleType: SampleType.AUTO },
            { id: "B06", bands: 1, sampleType: SampleType.AUTO },
            { id: "B07", bands: 1, sampleType: SampleType.AUTO },
            { id: "B08", bands: 1, sampleType: SampleType.AUTO },
            { id: "B8A", bands: 1, sampleType: SampleType.AUTO },
            { id: "B09", bands: 1, sampleType: SampleType.AUTO },
            { id: "B11", bands: 1, sampleType: SampleType.AUTO },
            { id: "B12", bands: 1, sampleType: SampleType.AUTO },
            { id: "RGB", bands: 3, sampleType: SampleType.AUTO },
            { id: "RGBN", bands: 4, sampleType: SampleType.AUTO },
            { id: "TCI", bands: 3, sampleType: SampleType.AUTO },
            { id: "NDVI", bands: 1, sampleType: SampleType.FLOAT32 },  // NDVI band
            { id: "SAVI", bands: 3, sampleType: SampleType.FLOAT32 },
            { id: "SCL", bands: 3, sampleType: SampleType.FLOAT32 },  
           
        ]
    };
}

function evaluatePixel(samples, scenes, inputMetadata, customData, outputMetadata) {
    ndvi = (samples.B08 - samples.B04) / (samples.B08 + samples.B04);
    
    // Calculate SAVI
    L = 0.5;  // Soil brightness correction factor (adjust as needed)
    savi = ((samples.B08 - samples.B04) / (samples.B08 + samples.B04 + L)) * (1 + L);

    return {
        B01: [samples.B01],
        B02: [samples.B02],
        B03: [samples.B03],
        B04: [samples.B04],
        B05: [samples.B05],
        B06: [samples.B06],
        B07: [samples.B07],
        B08: [samples.B08],
        B8A: [samples.B8A],
        B09: [samples.B09],
        B11: [samples.B11],
        B12: [samples.B12],
        RGB: [2.5*samples.B04, 2.5*samples.B03, 2.5*samples.B02],
        RGBN: [samples.B04, samples.B03, samples.B02, samples.B08],
        TCI: [3*samples.B04, 3*samples.B03, 3*samples.B02],
        NDVI: [ndvi],  
        SAVI: [savi],
        SCL: [samples.SCL],
    };
}
"""
api = SentinelAPI(config.sh_client_id, config.sh_client_secret, 'https://scihub.copernicus.eu/dhus')
evalscript_s1 = """
//VERSION=3
function setup() {
    return {
        input: ["VV", "VH"],
        output: [
            { id: "VV", bands: 1, sampleType: SampleType.AUTO },
            { id: "VH", bands: 1, sampleType: SampleType.AUTO },
            { id: "RGB", bands: 3, sampleType: SampleType.AUTO }
            
        ],
        visualization: {
            bands: ["VV", "VH"],
            min: [-25,-25], // Adjust these values based on your data distribution
            max: [5,5], // Adjust these values based on your data distribution
        }
    };
}

function evaluatePixel(samples, scenes, inputMetadata, customData, outputMetadata) {
    // Adjust the coefficients for a natural color representation
    ratio = samples.VH-samples.VV
    rgb_ratio = samples.VH+ samples.VV+ratio
    red = samples.VH;
    green = samples.VV;
    blue = rgb_ratio;
    return {
        VH: [red],
        VV: [green],
        RGB: [red, green, blue] 
    };
}
"""

api = SentinelAPI(config.sh_client_id, config.sh_client_secret, 'https://scihub.copernicus.eu/dhus')
evalscript_l8 = """
//VERSION=3
function setup() {
    return {
        input: ["B02", "B03", "B04","B05"], // Bands for true color and NIR
        output: [
            { id: "rgb", bands: 3,  sampleType: SampleType.AUTO}, // True color RGB
            { id: "ndvi", bands: 3,  sampleType: SampleType.AUTO} // NDVI
        ]
        
    };
}

function evaluatePixel(samples, scenes, inputMetadata, customData, outputMetadata) {
    // Calculate NDVI
        ndvi = (samples.B05 - samples.B04) / (samples.B05 + samples.B04);

    // Return true color RGB and NDVI values
    return {
        rgb: [2.5*samples.B04, 2.5*samples.B03, 2.5*samples.B02], // True color RGB
        ndvi: [ndvi] // NDVI
    };
}

"""


# Function to download Sentinel-2 images using sentinelhub
def download_sentinel_images(api, shapefile_path, start_date, end_date, output_folder):
    # Load the shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.set_geometry('geometry')

    # Set the common CRS for both the shapefile and tiles
    common_crs = 'EPSG:32736'
    # Read the shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)

    # Calculate the area of each polygon in square meters
    target_crs = 'EPSG:32736'
    gdf = gdf.to_crs(target_crs)

    gdf['area_m2'] = gdf['geometry'].area

    # Sum the areas to get the total area of the shapefile
    total_area_shapefile_m2 = gdf['area_m2'].sum()
    # Convert total area to hectares
    total_area_shapefile_hectares = total_area_shapefile_m2 / 10000

    # Convert total area to square kilometers
    total_area_shapefile_square_km = total_area_shapefile_hectares / 100
    # Print the total area in square kilometers
    #print(f"Total area of the shapefile: {total_area_shapefile_square_km:.2f} square kilometers")


    # Print the total area
    #print(f"Total area of the shapefile: {total_area_shapefile_m2:.2f} square meters")
    #print(f"Total area of the shapefile: {total_area_shapefile_hectares:.2f} square meters")

    # Calculate the bounding box of the union of all geometries in the shapefile
    shapefile_union = unary_union(gdf['geometry'])
    bbox = BBox(bbox=shape(shapefile_union).bounds, crs=CRS(common_crs))

    # Iterate over polygons
    for idx, row in gdf.iterrows():
        polygon = row['geometry']


        request = SentinelHubRequest(
            data_folder=os.path.join(output_folder, 'sentinel2'),
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=(start_date, end_date),
                    mosaicking_order='mostRecent',
                    maxcc=0.25
                )
            ],
            responses=[
                SentinelHubRequest.output_response('RGB', MimeType.TIFF),
                SentinelHubRequest.output_response('SCL', MimeType.TIFF)
            ],
            bbox=bbox,
            size=image_size,
            config=config
        )

        try:
            request.save_data()


            print(f"Data saved successfully for polygon {idx}!")
        except Exception as e:
            print(f"Error saving data for polygon {idx}: {e}")
    return total_area_shapefile_hectares



def download_sentinel1_images(api, shapefile_path, start_date, end_date, output_folder):
    # Load the shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.set_geometry('geometry')

    # Set the common CRS for both the shapefile and tiles
    common_crs = 'EPSG:32736'
    # Read the shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)

    # Calculate the area of each polygon in square meters
    target_crs = 'EPSG:32736'
    gdf = gdf.to_crs(target_crs)

    gdf['area_m2'] = gdf['geometry'].area

    # Sum the areas to get the total area of the shapefile
    total_area_shapefile_m2 = gdf['area_m2'].sum()
    # Convert total area to hectares
    total_area_shapefile_hectares = total_area_shapefile_m2 / 10000

    # Convert total area to square kilometers
    total_area_shapefile_square_km = total_area_shapefile_hectares / 100
    # Print the total area in square kilometers
    #print(f"Total area of the shapefile: {total_area_shapefile_square_km:.2f} square kilometers")

    # Print the total area
    #print(f"Total area of the shapefile: {total_area_shapefile_m2:.2f} square meters")
    #print(f"Total area of the shapefile: {total_area_shapefile_hectares:.2f} square meters")

    # Iterate over polygons
    for idx, row in gdf.iterrows():
        polygon = row['geometry']

        # Calculate the bounding box of the current polygon
        bbox = BBox(bbox=shape(polygon).bounds, crs=CRS(common_crs))

        request = SentinelHubRequest(
            data_folder=os.path.join(output_folder, 'sentinel1'),
            evalscript=evalscript_s1,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL1_IW_DES,
                    time_interval=(start_date, end_date),
                    mosaicking_order='mostRecent',
                    maxcc=0.25
                )
            ],
            responses=[
                SentinelHubRequest.output_response('VV', MimeType.TIFF),
                SentinelHubRequest.output_response('VH', MimeType.TIFF),
                SentinelHubRequest.output_response('RGB', MimeType.TIFF),
            ],
            bbox=bbox,  # Use the bounding box of the current polygon
            size=image_size,
            config=config
        )

        try:
            request.save_data()
            print(f"Data saved successfully for polygon {idx}!")
        except Exception as e:
            print(f"Error saving data for polygon {idx}: {e}")

    return total_area_shapefile_hectares

def download_landsat_images(api, shapefile_path, start_date, end_date, output_folder):
    # Load the shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.set_geometry('geometry')


    # Set the common CRS for both the shapefile and tiles
    common_crs = 'EPSG:32736'
    # Read the shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)

    # Calculate the area of each polygon in square meters
    target_crs = 'EPSG:32736'
    gdf = gdf.to_crs(target_crs)

    gdf['area_m2'] = gdf['geometry'].area

    # Sum the areas to get the total area of the shapefile
    total_area_shapefile_m2 = gdf['area_m2'].sum()
    # Convert total area to hectares
    total_area_shapefile_hectares = total_area_shapefile_m2 / 10000

    # Convert total area to square kilometers
    total_area_shapefile_square_km = total_area_shapefile_hectares / 1000
    # Print the total area in square kilometers
    #print(f"Total area of the shapefile: {total_area_shapefile_square_km:.2f} square kilometers")



    # Print the total area
    #print(f"Total area of the shapefile: {total_area_shapefile_m2:.2f} square meters")
    #print(f"Total area of the shapefile: {total_area_shapefile_hectares:.2f} square hectares")

    # Calculate the bounding box of the union of all geometries in the shapefile
    shapefile_union = unary_union(gdf['geometry'])
    bbox = BBox(bbox=shape(shapefile_union).bounds, crs=CRS(common_crs))

    # Iterate over polygons
    for idx, row in gdf.iterrows():
        polygon = row['geometry']


        request = SentinelHubRequest(
            data_folder=os.path.join(output_folder, 'landsat'),
            evalscript=evalscript_l8,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.LANDSAT_OT_L1,
                    time_interval=(start_date, end_date),
                    mosaicking_order='mostRecent',
                    maxcc=0.25

                )
            ],
            responses=[
                SentinelHubRequest.output_response('rgb', MimeType.TIFF),
                SentinelHubRequest.output_response('ndvi', MimeType.TIFF)
            ],
            bbox=bbox,
            size=image_size,
            config=config
        )

        try:
            request.save_data()


            print(f"Data saved successfully for polygon {idx}!")
        except Exception as e:
            print(f"Error saving data for polygon {idx}: {e}")
    return total_area_shapefile_hectares
    #def shapefile_wihtout_water():
      #  Land_without_water = total_area_shapefile_square_km - total_water_area
      #  return Land_without_water

def extract_tar(tar_path, extract_path):
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(extract_path)

def remove_file(file_path):
    try:
        os.remove(file_path)
        print(f"File removed: {file_path}")
    except OSError as e:
        print(f"Error removing file {file_path}: {e}")

def preprocess_patch(patch):
    # Convert to RGB format
    patch_rgb = np.transpose(patch, (1, 2, 0))
    # Convert to PIL Image
    patch_image = Image.fromarray((patch_rgb * 255).astype(np.uint8))
    # Resize the image to match the model's input size
    patch_image = patch_image.resize((512, 512))
    # Apply transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # Convert to PyTorch tensor
    patch_tensor = transform(patch_image)
    # Add a batch dimension
    patch_tensor = patch_tensor.unsqueeze(0)
    return patch_tensor

def create_patches_from_single_image(image_path, patch_size=512):
    num_patches_total = 0
    total_predictions = np.zeros((1, 1, 512, 512))

    try:
        with rasterio.open(image_path) as src:
            num_patches = 0

            for i in range(0, src.width, patch_size):
                for j in range(0, src.height, patch_size):
                    window = rasterio.windows.Window(i, j, patch_size, patch_size)
                    patch = src.read(window=window)

                    # TODO: Process the patch as needed (e.g., save it to disk)
                    processed_patch = preprocess_patch(patch)

                    # Make predictions on the processed patch
                    with torch.no_grad():
                        predictions_patch = loaded_model(processed_patch.to(device))
                        predictions_patch = torch.sigmoid(predictions_patch)
                        predictions_patch = predictions_patch.cpu().numpy()

                        # Accumulate the predictions
                        total_predictions += predictions_patch

                    # For now, just print the patch information


                    num_patches += 1
                    num_patches_total += 1



    except rasterio.errors.RasterioIOError as e:
        print(f"Error opening the image at {image_path}: {e}")

    return num_patches_total, total_predictions
# Function to visualize pixel distributions
def plot_pixel_distribution(data, title):
    plt.hist(data.flatten(), bins=range(0, 12), align='left', rwidth=0.8)
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.show()

# Function to create colored RGB image with clouds and shadows replaced by Sentinel-1 VV band

def replace_clouds_and_shadow_with_sentinel1(rgb, s1_vv, cloud_mask,shadow_mask):
    colored_rgb = rgb.copy()


     # Replace cloud pixels in RGB with corresponding values from Sentinel-1 VV
    colored_rgb[:, cloud_mask] = s1_vv[:, cloud_mask]
    colored_rgb[:, shadow_mask] = s1_vv[:, shadow_mask]
    # Replace shadow pixels in RGB with corresponding values from Sentinel-1 VV
    #colored_rgb[shadow_mask] = s1_vv[shadow_mask]

    return colored_rgb


def water_pixel_areas(scl_clip, pixel_size_m2):
    """
    Calculate the total area covered by cloud pixels in a Sentinel-2 SCL band image.

    Parameters:
    - scl_band: NumPy array representing the Sentinel-2 SCL band image.
    - pixel_size_m2: Size of each pixel in square meters.

    Returns:
    - Total area covered by cloud pixels in square meters.
    """
    water_threshold = [6]
    # Create a binary mask for cloud pixels
    water_mask = np.isin(scl_clip, water_threshold)

    # Calculate the total area covered by cloud pixels
    total_water_area = np.sum(water_mask) * pixel_size_m2

    return total_water_area



def clip_and_save_image(image_path, shapefile_path, output_folder, output_filename):
    # Read the Sentinel-2 image
    with rasterio.open(image_path) as src:
        # Read the shapefile
        gdf = gpd.read_file(shapefile_path)
        gdf = gdf.set_geometry('geometry')

        # Convert the GeoDataFrame to the same CRS as the Sentinel-2 image
        gdf = gdf.to_crs(src.crs)

        # Use the bounds of the GeoDataFrame as the bounding box for clipping
        bbox = gdf.geometry.total_bounds

        # Perform the clipping
        clipped_image, transform = mask.mask(src, gdf.geometry, crop=True)

        # Update metadata for the clipped image
        out_meta = src.meta
        out_meta.update({"driver": "GTiff",
                         "height": clipped_image.shape[1],
                         "width": clipped_image.shape[2],
                         "transform": transform})

        # Save the clipped image
        output_path = os.path.join(output_folder, output_filename)
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(clipped_image)

    print(f"Clipped image saved to: {output_path}")



def open_landsat_image(output_folder, subfolder_name):
    # Get the subfolder path
    subfolder_path = os.path.join(output_folder, subfolder_name)

    # Traverse through subfolders using os.walk
    for root, dirs, files in os.walk(subfolder_path):
        # Check if 'rgb.tiff' is present in the current folder
        if 'rgb.tif' in files:
            rgb_file_path = os.path.join(root, 'rgb.tif')

            # Open the Sentinel-2 image
            with rasterio.open(rgb_file_path) as src:
                # Read the RGB bands
                l8_rgb = src.read([1, 2, 3], masked=True)
                show(l8_rgb)
            return l8_rgb # Stop searching once the image is found

    print("RGB file not found in the specified subfolder.")

def open_sentinel2_image(output_folder, subfolder_name):
    # Get the subfolder path
    subfolder_path = os.path.join(output_folder, subfolder_name)

    # Traverse through subfolders using os.walk
    for root, dirs, files in os.walk(subfolder_path):
        # Check if 'rgb.tiff' is present in the current folder
        if 'RGB.tif' in files:
            rgb_file_path = os.path.join(root, 'RGB.tif')

            # Open the Sentinel-2 image
            with rasterio.open(rgb_file_path) as src:
                # Read the RGB bands
                rgb = src.read([1, 2, 3], masked=True)

            return rgb # Stop searching once the image is found

    print("RGB file not found in the specified subfolder.")

def open_sentinel1_image(output_folder, subfolder_name):
    # Get the subfolder path
    subfolder_path = os.path.join(output_folder, subfolder_name)

    # Traverse through subfolders using os.walk
    for root, dirs, files in os.walk(subfolder_path):
        # Check if 'rgb.tiff' is present in the current folder
        if 'RGB.tif' in files:
            rgb_file= os.path.join(root, 'RGB.tif')

            # Open the Sentinel-2 image
            with rasterio.open(rgb_file) as src:
                # Read the RGB bands
                s1rgb = src.read([1, 2, 3], masked=True)

            return s1rgb # Stop searching once the image is found

    print("RGB file not found in the specified subfolder.")

def open_scl_image(shapefile_path,output_folder, subfolder_name):
    # Get the subfolder path
    subfolder_path = os.path.join(output_folder, subfolder_name)

    # Traverse through subfolders using os.walk
    for root, dirs, files in os.walk(subfolder_path):
        # Check if 'rgb.tiff' is present in the current folder
        if 'SCL.tif' in files:
            scl_file_path = os.path.join(root, 'SCL.tif')

            # Open the Sentinel-2 image
            with rasterio.open(scl_file_path) as src:

                # Read the RGB bands
                scl_band = src.read(1, masked=True)
                gdf = load_shapefile(shapefile_path)
                gdf = gdf.set_geometry('geometry')

        # Convert the GeoDataFrame to the same CRS as the Sentinel-2 image
                gdf = gdf.to_crs(src.crs)
                scl_clip, transform = mask.mask(src, gdf.geometry, crop=True)

            # Update the metadata of the clipped image
                scl_meta = src.meta.copy()
                scl_meta.update({
                        "driver": "GTiff",
                        "height": scl_clip.shape[1],
                        "width": scl_clip.shape[2],
                        "transform": transform
                })

            return scl_band


            # Stop searching once the image is found

    print("RGB file not found in the specified subfolder.")

def open_scl_image_and_clip(shapefile_path,output_folder, subfolder_name):
    # Get the subfolder path
    subfolder_path = os.path.join(output_folder, subfolder_name)

    # Traverse through subfolders using os.walk
    for root, dirs, files in os.walk(subfolder_path):
        # Check if 'rgb.tiff' is present in the current folder
        if 'SCL.tif' in files:
            scl_file_path = os.path.join(root, 'SCL.tif')

            # Open the Sentinel-2 image
            with rasterio.open(scl_file_path) as src:

                # Read the RGB bands
                scl_band = src.read(1, masked=True)
                gdf = gpd.read_file(shapefile_path)
                gdf = gdf.set_geometry('geometry')

        # Convert the GeoDataFrame to the same CRS as the Sentinel-2 image
                gdf = gdf.to_crs(src.crs)
                scl_clip, transform = mask.mask(src, gdf.geometry, crop=True)
                # Update the metadata of the clipped image
                scl_meta = src.meta.copy()
                scl_meta.update({
                        "driver": "GTiff",
                        "height": scl_clip.shape[1],
                        "width": scl_clip.shape[2],
                        "transform": transform
                })

                total_water_area = water_pixel_areas(scl_clip, m2_to_hectares)
            return total_water_area

 #Convert pixels to hectares
def pixels_to_hectares(pixels, area_per_pixel):
    area_m2 = pixels * area_per_pixel
    area_hectares = area_m2 * m2_to_hectares
    return area_hectares

 # Set your segmentation threshold for considering cropland

def count_pixels(predictions, threshold):
    above_threshold = (predictions > threshold).sum()
    below_threshold = (predictions <= threshold).sum()
    return above_threshold, below_threshold

def delete_subfolders(directory):
    # List all items (files and subdirectories) in the given directory
    items = os.listdir(directory)

    # Iterate over items
    for item in items:
        item_path = os.path.join(directory, item)

        # Check if the item is a subdirectory
        if os.path.isdir(item_path):
            try:
                # Use shutil.rmtree to delete the subdirectory and its contents
                shutil.rmtree(item_path)
                print(f"Subfolder '{item}' deleted successfully.")
            except Exception as e:
                print(f"Error deleting subfolder '{item}': {e}")
def load_shapefile(file_path):
    return gpd.read_file(file_path)

def process_shapefile(shapefile_path, start_date, end_date):
    return load_shapefile(shapefile_path)

def get_shapefile_list():
    # Define the directory where shapefiles are uploaded
    upload_folder = 'uploads/Agriculture_Sample_Data'

    # Get a list of shapefiles in the folder
    shapefiles = [filename for filename in os.listdir(upload_folder) if filename.endswith('.shp')]
    return shapefiles
def save_reconstructed_image(img_path):
    # Placeholder implementation

    print("Reconstructed image saved at:", img_path)

def save_clip_image(clipped_tiff_path):
    # Placeholder implementation
    print("clip image saved at:", clipped_tiff_path)
# Function to convert TIFF image to PNG format
def convert_tiff_to_png(tiff_path, png_output_path):

    try:
        with Image.open(tiff_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(png_output_path, format="PNG")
        return True
    except Exception as e:
        print(f"Error converting TIFF to PNG: {e}")
        return False


###Flask app routes
@app.route('/get_shapefile_suggestions')
def get_shapefile_suggestions():
    upload_folder = 'uploads/Agriculture_Sample_Data'
    input_shapefile = request.args.get('input', '').lower()
    suggestions = []

    # Iterate over all shapefiles in the upload folder
    for filename in os.listdir(upload_folder):

        if filename.lower().startswith(input_shapefile):
            suggestions.append(filename)

    return jsonify(suggestions)



@app.route('/get_shapefile_list')
def get_shapefile_list_route():
    shapefiles = get_shapefile_list()
    return jsonify(shapefiles)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def prediction():
    if request.method == 'POST':
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        selected_shapefile = request.form['selected_shapefile']  # Assuming you have a form field for the selected shapefile

        # Define the directory where shapefiles are uploaded
        upload_folder = 'uploads/Agriculture_Sample_Data'

        # Construct the full path to the selected shapefile
        shapefile_path = os.path.join(upload_folder, selected_shapefile)

        # Check if the selected shapefile exists
        if os.path.isfile(shapefile_path) and shapefile_path.endswith('.shp'):
            # Process the selected shapefile
            process_shapefile(shapefile_path, start_date, end_date)

            # Set the output folder for downloaded images
            output_folder = "downloaded_images"

            image_folder = "static"
            #img_path = os.path.join(img_folder,"constructed.tif")

            if not os.path.exists(image_folder):

                os.makedirs(image_folder)


            img_name = "constructed.tif"
            img_path = os.path.join(image_folder, img_name)
            save_reconstructed_image(img_path)

            # Convert clipped image to PNG format
            clipped_tiff_path = os.path.join(output_folder, f"{os.path.splitext(selected_shapefile)[0]}_clipped.tif")
            clipped_png_path = os.path.join(output_folder, f"{os.path.splitext(selected_shapefile)[0]}_clipped.png")
            #save_clip_image(clipped_png_path)
            convert_tiff_to_png(clipped_tiff_path, clipped_png_path)



            # Create output folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Create output folder if it doesn't exist

            # Step 1: Download Sentinel-2 images
            total_area_shapefile_hectares = download_sentinel_images(api, shapefile_path, start_date, end_date, output_folder)
            #Download landsat images
            download_landsat_images(api, shapefile_path, start_date, end_date, output_folder)
            #Step 2: Download Sentinel-1 images
            #download_sentinel1_images(api, shapefile_path, start_date, end_date, output_folder)

            for root, dirs, files in os.walk(output_folder):

                for file in files:
                    if file.endswith(".tar"):
                        tar_path = os.path.join(root, file)
                        extract_path = os.path.join(root, file[:-4])  # Remove '.tar' extension
                        extract_tar(tar_path, extract_path)
                        remove_file(tar_path)




            open_landsat_image(output_folder, 'landsat')
            open_sentinel2_image(output_folder,'sentinel2')
            open_scl_image(shapefile_path,output_folder,'sentinel2')

            #Define the crop_land prediction threshold
            cropland_threshold = 0.6
            # Define thresholds for cloud and shadow pixels in SCL band
            cloud_threshold = [8,9]
            shadow_threshold = [3]


            #cloud_mask = np.isin(scl_clip, water_threshold)


                    # Create a cloud and shadow mask

            cloud_mask = np.isin(open_scl_image(shapefile_path,output_folder,'sentinel2'), cloud_threshold)
            shadow_mask = np.isin(open_scl_image(shapefile_path,output_folder,'sentinel2'), shadow_threshold)


            rgb = open_sentinel2_image(output_folder,'sentinel2')
            l8_rgb = open_landsat_image(output_folder,'landsat')



            rgb_replaced = replace_clouds_and_shadow_with_sentinel1(rgb,l8_rgb,cloud_mask,shadow_mask)

            def replace_clouds_and_shadow_with_landsat(rgb, l8_rgb, cloud_mask, shadow_mask):

    # Your code to replace clouds and shadows with Sentinel 1 pixels

                # Save the composite images to files
                rgb_filename = 'original_rgb_image_with_cloud_covers.png'
                rgb_replaced_filename = 'replaced_rgb_image.png'
                plt.imsave(os.path.join('static', rgb_filename), np.moveaxis(rgb.data, 0, -1))
                plt.imsave(os.path.join('static', rgb_replaced_filename), np.moveaxis(rgb_replaced.data, 0, -1))

                return rgb_filename, rgb_replaced_filename
            # Call the function to replace clouds and shadows
            rgb_filename, rgb_replaced_filename = replace_clouds_and_shadow_with_landsat(rgb, l8_rgb, cloud_mask, shadow_mask)









            # Copy georeferencing information from the original Sentinel-2 image
            print("Stopping here!")
            print("Copying georeferencing information...")
            s2_rgb_file_path_options = glob.glob(os.path.join(root, 'response', '[Rr][Gg][Bb].*'))
            print(s2_rgb_file_path_options)

        # Check if any matching files were found
            if s2_rgb_file_path_options:

                # Sort the options to prioritize 'RGB.tif' over 'rgb.tif'
                s2_rgb_file_path_options.sort()
                s2_rgb_file_path = s2_rgb_file_path_options[0]  # Use the first matching file
                print(f"s2_rgb_file_path: {s2_rgb_file_path}")

                # Open the Sentinel-2 image
                with rasterio.open(s2_rgb_file_path) as src:
                    transform = src.transform
                    crs = src.crs

                with rasterio.open(img_path, 'w', driver='GTiff', height=src.height, width=src.width, count=3, dtype=rgb_replaced.dtype, crs=crs, transform=transform) as dst:

                    dst.write(rgb_replaced)

                print(f"Reconstructed image saved at: {img_path}")
            else:

                print("No matching RGB file found!")



                # Save the clipped image with the name of the shapefile
            clip_and_save_image(img_path,shapefile_path, image_folder, f"{os.path.splitext(os.path.basename(shapefile_path))[0]}_clipped.png")


            scl_band = open_scl_image(shapefile_path,output_folder,'sentinel2')
            total_water_area = open_scl_image_and_clip(shapefile_path,output_folder, 'sentinel2')
            total_shapefiles_without_water = total_area_shapefile_hectares - total_water_area
            print(f'Area covered by water is in square hectares is : {total_water_area}')
            print(f'Area of shapefile without water in square hectares is: {total_shapefiles_without_water}')

            # Step 2: Process downloaded images and create patches
            # Assuming only one image is downloaded, modify accordingly if multiple images are downloaded
            delete_subfolders(output_folder)
            downloaded_image_path = os.path.join(image_folder, os.listdir(image_folder)[0])
            print(downloaded_image_path)
            num_patches, total_predictions = create_patches_from_single_image(downloaded_image_path)


            # Print the number of patches created
            print(f"Number of patches created: {num_patches}")

            # Sum all predictions from all patches if any
            total_predictions = np.sum(total_predictions, axis=(0, 1))
            print(f'total predictions{total_predictions}')


            cropland_predictions_above_threshold = (total_predictions >= cropland_threshold).sum()
            cropland_predictions_below_threshold = (total_predictions < cropland_threshold).sum()


            # Assuming area_per_pixel_m2 is defined as you've done earlier
            # Convert pixels to hectares
            cropland_area_above_threshold = pixels_to_hectares(cropland_predictions_above_threshold, area_per_pixel_m2)
            cropland_area_below_threshold = pixels_to_hectares(cropland_predictions_below_threshold,area_per_pixel_m2)

            print(f"Area covered by cropland above the segmentation threshold: {cropland_area_above_threshold:.2f} hectares")
            print(f"Area covered by background below the segmentation threshold: {cropland_area_below_threshold:.2f} hectares")
            total_prediction =  cropland_area_above_threshold + cropland_area_below_threshold

            # Calculate the total area covered by cropland based on predictions and the total area of the shapefile
            total_cropland_area_predicted = (cropland_area_above_threshold/total_prediction) * total_shapefiles_without_water

            # Calculate the percentage of the shapefile area covered by cropland
            percentage_cropland_area = (total_cropland_area_predicted / total_shapefiles_without_water) * 100

            print(f"Total area covered by cropland: {total_cropland_area_predicted:.2f} hectares")
            print(f"Percentage of the shapefile area covered by cropland: {percentage_cropland_area:.2f}%")




            #print(total_water_area)
            #Total_water = total_area_shapefile_square_km - total_water_area
            #print(f"Total area covered by land without water: {Total area covered by land without water:.2f} hectares")
            delete_subfolders(output_folder)
            shutil.rmtree(output_folder)


            print(f"Processing completed for shapefile: {shapefile_path}")
            Prediction={
                1:f'{total_shapefiles_without_water:.2f}',
                2:f'{total_area_shapefile_hectares:.2f}',
                3:f'{percentage_cropland_area:.2f}%',
                4:f'{total_cropland_area_predicted:.2f}',
                5:f'{total_water_area:.2f}',
                'statement1':"Total area of shapefile without water",
                'statement2':"Total area of shapefile in square hectares",
                'statement3':"Percentage of cropland area",
                'statement4':"Total cropland area predicted",
                'statement5':"Total area covered by water"


            }

            def remove_static_folder(response):
                shutil.rmtree(image_folder)
                return response

            results_file_path = save_results_to_csv(Prediction,selected_shapefile)
            # Return only the clipped image associated with the selected shapefile
            static_files = [f"{os.path.splitext(selected_shapefile)[0]}_clipped.png"]
            print(static_files)
            print("Deleting it from here")




            return render_template('result.html', prediction=Prediction, selected_shapefile=selected_shapefile,
                       static_files=[f"{os.path.splitext(selected_shapefile)[0]}_clipped.png"],
                       original_rgb_image=rgb_filename, replaced_rgb_image=rgb_replaced_filename)


      
            shutil.rmtree(image_folder)





@app.route('/static/<path:filename>')
def static_file(filename):
    static_folder = 'static'  # Specify the name of your static folder
    return send_from_directory(static_folder, filename)
'''
@app.after_request
def remove_static_folder(response):
    static_folder = os.path.join(app.root_path, 'static')
    shutil.rmtree(static_folder)
    return response
'''

@app.route('/download_results/<selected_shapefile>')
def download_results(selected_shapefile):
    # Assuming results.csv is stored in the 'downloads' folder
    results_file_path = os.path.join(downloads_folder, f'{selected_shapefile}_results.csv')
    print(results_file_path)

    # Check if the file exists
    if os.path.exists(results_file_path):
        return send_file(results_file_path, as_attachment=True)
    else:
        return "File not found",404

# Save prediction results to CSV file
def save_results_to_csv(predicton, selected_shapefile):
    filename = f'{selected_shapefile}_results.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in predicton.items():
            writer.writerow([key, value])

    return filename  # Return the filename for reference




downloads_folder = 'download'

# Ensure the downloads folder exists
if not os.path.exists(downloads_folder):
    os.makedirs(downloads_folder)




if __name__ == "__main__":




    app.run(host='0.0.0.0')




    



