# DL_delineation_web
This is an AI-powered  web application for delineating agricultural lands based on shapefile data. Input the start and end dates, select a shapefile, and submit the form to receive predictions. For optimal performances, kindly use the seasonal dates as start and end dates. In the background, the application uses the input shapefile, start and end dates to: 
1 download sentinel 2 and Landsat 8,9 images, 
2. Process these images  including replacing cloud pixels in Sentinel 2 with Landast 8, and 9 pixels. 
3. Clipping the reconstructed image and making predictions thereof. 
