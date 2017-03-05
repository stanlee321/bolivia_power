# bolivia_power
Repo containing info about how to extract #powerdata from OpenStreetMap and plotit, in this case for Bolivian territory.

![alt tag](https://github.com/stanlee321/bolivia_power/blob/master/index.png)

The DataViz-Copy1.ipynb show the procedure of ploting the geo data extracted from OpenStreetMap (OSM) metadata that has been cleaned into the file "nodes_clean.csv, where you can find latitude, longitude and ID of each "node" in the metadata of OSM. This medatada is extracted with the use of the extract_data_1.1.py.

The to_csv.py code makes the above mencioned, clean the metadata from OpenSteetMap into a clean csv file.

The last file SIN_BOLIVIA.py is a bunch of metadata that left to clean and treat, but with the help of some friend that provied me with some aditional SIN data and my original data extracted from OSM, I plot the whole "Bolivian Interconected System" in a 3D fashion with the help of OpenGraphiti as you can see in this youtube video. 

https://www.youtube.com/watch?v=3qg-eD3dXrk

![alt tag] (https://github.com/stanlee321/bolivia_power/blob/master/neural.png)
This last image shows the traning of the neural backprop program for powerflow calculation.
