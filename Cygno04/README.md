# Scripts to make simple 2D maps

## From the images (no reconstruction)
This is to make average 2D maps from several runs. This is done in the simplest way:
1.  Downloads the midas file from the storage
2. Downloads a pedestal run and computes *mean *and* std* of it
3. For each image, does `image - pedestal mean`, and suppresses the pixel if it is `< pedestal std` 
4. Does the average of the result. 

### Dependencies:
cygno lib (to read S3) and midas (to open raw files):

`pip install git+https://github.com/CYGNUS-RD/cygno.git -U`

`pip install 'https://github.com/CYGNUS-RD/middleware/blob/master/midas/midaslib.tar.gz?raw=true' `

### Usage:

`python make_2dmaps_frommidas.py --run-start 124079 --run-end 124083 --pedrun 124073 --output muon_map_images.root`

## From reconstructed clusters
This is faster, because it downloads lighter RECO files and makes the average from redpixels of any reconstructed clusters, with a very loose preselection on them. Cons: response corrections (like vignetting) is applied already.
### Usage:
`python3 make_2dmaps_fromclusters_s3.py --run-start 124074 --run-end 124083 --output muon_map_from_clusters_124074_124083.root   `

Plot the maps
Both plots save 3 TH2D: light yield of pixels (int2d), occupancy of pixels (occ2d), and their ratio (i.e. the average LY, prof2d). Use this script to make PDF/PNG plots. Just change the name to "prof2d/occ2d/int2d" -> "prof2d\_images/occ2d\_images/int2d\_images" if you use the first script and not the second. 

` python3 plot_2dmaps.py --input muon_map_images.root --output-dir plots --zmax-prof 8 `

Output are plots like [these ones](https://emanuele.web.cern.ch/emanuele/Cygnus/plots/cygno04/commissioning_lnf/?match=prof2d). 
