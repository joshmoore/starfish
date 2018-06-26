#!/usr/bin/env python
# coding: utf-8
#
# EPY: stripped_notebook: {"metadata": {"hide_input": false, "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.6.5"}, "toc": {"nav_menu": {}, "number_sections": true, "sideBar": true, "skip_h1_title": false, "toc_cell": false, "toc_position": {}, "toc_section_display": "block", "toc_window_display": false}}, "nbformat": 4, "nbformat_minor": 2}

# EPY: START markdown
# ## Reproduce Published results with Starfish
# 
# This notebook walks through a workflow that reproduces a MERFISH result for one field of view using the starfish package.
# EPY: END markdown

# EPY: START code
import os
import pprint
import time

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import scoreatpercentile

from showit import image, tile
from starfish.constants import Indices
from starfish.io import Stack
from starfish.viz import tile_lims

# EPY: ESCAPE %matplotlib inline
# EPY: END code

# EPY: START code
# load the data from cloudfront
s = Stack()
s.read('https://dmf0bdeheu4zf.cloudfront.net/MERFISH/fov_001/experiment.json')
# EPY: END code

# EPY: START code
# data from one FOV correspond to 16 single plane images as shown here (see below for details)
tile(s.image.squeeze());
# EPY: END code

# EPY: START markdown
# Individual hybridization rounds and channels can also be visualized
# EPY: END markdown

# EPY: START code
# show all hybridization rounds of channel 0
s.image.show_stack({Indices.CH: 0})
# EPY: END code

# EPY: START markdown
# ## Show input file format that specifies how the tiff stack is organized
# 
# The stack contains multiple images corresponding to the channel and hybridization round. MERFISH builds a 16 bit barcode from 8 hybridization rounds, each of which measures two channels that correspond to contiguous (but not necessarily consistently ordered) bits of the barcode. 
# 
# The MERFISH computational pipeline also constructs a scalar that corrects for intensity differences across each of the 16 images, e.g., one scale factor per bit position.
# 
# The stacks in this example are pre-registered using fiduciary beads. 
# EPY: END markdown

# EPY: START code
pp = pprint.PrettyPrinter(indent=2)
pp.pprint(s.org)
# EPY: END code

# EPY: START markdown
# ## Visualize codebook
# EPY: END markdown

# EPY: START markdown
# The MERFISH codebook maps each barcode to a gene (or blank) feature. The codes in the MERFISH codebook are constructed from a 4-hamming error correcting code with exactly 4 "on" bits per barcode
# EPY: END markdown

# EPY: START code
codebook = pd.read_csv('https://dmf0bdeheu4zf.cloudfront.net/MERFISH/codebook.csv', dtype={'barcode': object})
codebook.head(20)
# EPY: END code

# EPY: START markdown
# ## Filter and scale raw data before decoding
# EPY: END markdown

# EPY: START code
from starfish.pipeline.filter.gaussian_high_pass import GaussianHighPass
from starfish.pipeline.filter.gaussian_low_pass import GaussianLowPass
from starfish.pipeline.filter.richardson_lucy_deconvolution import DeconvolvePSF
from starfish.viz import tile_lims
# EPY: END code

# EPY: START markdown
# Begin filtering with a high pass filter to remove background signal.
# EPY: END markdown

# EPY: START code
from starfish.pipeline.filter.gaussian_high_pass import GaussianHighPass
ghp = GaussianHighPass(sigma=3)
ghp.filter(s)
# EPY: END code

# EPY: START markdown
# The below algorithm deconvolves out the point spread function introduced by the microcope and is specifically designed for this use case. The number of iterations is an important parameter that needs careful optimization. 
# EPY: END markdown

# EPY: START code
from starfish.pipeline.filter.richardson_lucy_deconvolution import DeconvolvePSF
dpsf = DeconvolvePSF(num_iter=15, sigma=2)
dpsf.filter(s)
# EPY: END code

# EPY: START markdown
# Recall that the image is pre-registered, as stated above. Despite this, individual RNA molecules may still not be perfectly aligned across hybridization rounds. This is crucial in order to read out a measure of the itended barcode (across hybridization rounds) in order to map it to the codebook. To solve for potential mis-alignment, the images can be blurred with a 1-pixel Gaussian kernel. The risk here is that this will obfuscate signals from nearby molecules. 
# 
# A local search in pixel space across hybridization rounds can also solve this. 
# EPY: END markdown

# EPY: START code
from starfish.pipeline.filter.gaussian_low_pass import GaussianLowPass
glp = GaussianLowPass(sigma=1)
glp.filter(s)
# EPY: END code

# EPY: START markdown
# Use MERFISH-calculated size factors to scale the channels across the hybridization rounds and visualize the resulting filtered and scaled images. Right now we have to extract this information from the metadata and apply this transformation manually.
# EPY: END markdown

# EPY: START code
scale_factors = {(t[Indices.HYB], t[Indices.CH]): t['scale_factor'] for index, t in s.image.tile_metadata.iterrows()}
# EPY: END code

# EPY: START code
# this is a scaling method. It would be great to use image.apply here. It's possible, but we need to expose H & C to 
# at least we can do it with get_slice and set_slice right now.

for indices in s.image._iter_indices():
    data = s.image.get_slice(indices)[0]
    scaled = data / scale_factors[indices[Indices.HYB], indices[Indices.CH]]
    s.image.set_slice(indices, scaled)
# EPY: END code

# EPY: START code
from scipy.stats import scoreatpercentile
# EPY: END code

# EPY: START code
mp = s.image.max_proj(Indices.HYB, Indices.CH, Indices.Z)
clim = scoreatpercentile(mp, [0.5, 99.5])
image(mp, clim=clim)
# EPY: END code

# EPY: START markdown
# ## Use spot-detector to create 'encoder' table  for standardized input  to decoder
# 
# Each pipeline exposes a spot detector, and this spot detector translates the filtered image into an encoded table by detecting spots. The table contains the spot_id, the corresponding intensity (val) and the channel (ch), hybridization round (hyb), and bit position (bit) of each spot. 
# 
# The MERFISH pipeline merges these two steps together by finding pixel-based features, and then later collapsing these into spots and filtering out undesirable (non-spot) features. 
# 
# Therefore, no encoder table is generated, but a robust SpotAttribute and DecodedTable are both produced:
# EPY: END markdown

# EPY: START markdown
# ## Decode
# 
# Each assay type also exposes a decoder. A decoder translates each spot (spot_id) in the encoded table into a gene that matches a barcode in the codebook. The goal is to decode and output a quality score, per spot, that describes the confidence in the decoding. Recall that in the MERFISH pipeline, each 'spot' is actually a 16 dimensional vector, one per pixel in the image. From here on, we will refer to these as pixel vectors. Once these pixel vectors are decoded into gene values, contiguous pixels that are decoded to the same gene are labeled as 'spots' via a connected components labeler. We shall refer to the latter as spots.
# 
# There are hard and soft decodings -- hard decoding is just looking for the max value in the code book. Soft decoding, by contrast, finds the closest code by distance in intensity. Because different assays each have their own intensities and error modes, we leave decoders as user-defined functions. 
# 
# For MERFISH, which uses soft decoding, there are several parameters which are important to determining the result of the decoding method: 
# 
# ### Distance threshold
# In MERFISH, each pixel vector is a 16d vector that we want to map onto a barcode via minimum euclidean distance. Each barcode in the codebook, and each pixel vector is first mapped to the unit sphere by L2 normalization. As such, the maximum distance between a pixel vector and the nearest single-bit error barcode is 0.5176. As such, the decoder only accepts pixel vectors that are below this distance for assignment to a codeword in the codebook. 
# 
# ### Magnitude threshold
# This is a signal floor for decoding. Pixel vectors with an L2 norm below this floor are not considered for decoding. 
# 
# ### Area threshold
# Contiguous pixels that decode to the same gene are called as spots via connected components labeling. The minimum area of these spots are set by this parameter. The intuition is that pixel vectors, that pass the distance and magnitude thresholds, shold probably not be trusted as genes as the mRNA transcript would be too small for them to be real. This parameter can be set based on microscope resolution and signal amplification strategy.
# 
# ### Crop size 
# The crop size crops the image by a number of pixels large enough to eliminate parts of the image that suffer from boundary effects from both signal aquisition (e.g., FOV overlap) and image processing. Here this value is 40.
# 
# Given these three thresholds, for each pixel vector, the decoder picks the closest code (minimum distance) that satisfies each of the above thresholds, where the distance is calculated between the code and a normalized intensity vector and throws away subsequent spots that are too small.
# EPY: END markdown

# EPY: START code
from starfish.pipeline.features.pixels.pixel_spot_detector import PixelSpotDetector
psd = PixelSpotDetector(
    codebook='https://s3.amazonaws.com/czi.starfish.data.public/MERFISH/codebook.csv',
    distance_threshold=0.5176,
    magnitude_threshold=1,
    area_threshold=2,
    crop_size=40
)

spot_attributes, decoded = psd.find(s)
# EPY: END code

# EPY: START code
spot_attributes.head()
# EPY: END code

# EPY: START code
res = decoded.result  # this should be consistent across assays; 
# this one doesn't have a quality, but it should eventually converge to a shared type
res.head()
# EPY: END code

# EPY: START markdown
# In the above method, the private method of the decoder is used, which exposes additional metadata about the spots. 
# EPY: END markdown

# EPY: START code
print('Additional metadata:')
[f for f in dir(decoded) if not f.startswith('_')]
# EPY: END code

# EPY: START markdown
# Spot attributes are stored as skimage RegionProperties attributes
# EPY: END markdown

# EPY: START code
decoded.spot_props[:3]
# EPY: END code

# EPY: START markdown
# ## Compare to results from paper 
# 
# The below plot aggregates gene copy number across single cells in the field of view and compares the results to the published intensities in the MERFISH paper. 
# 
# To make this match perfectly, run deconvolution 15 times instead of 14. As presented below, STARFISH displays a lower detection rate.  
# EPY: END markdown

# EPY: START code
sns.set_context('talk')
sns.set_style('ticks')

bench = pd.read_csv('https://dmf0bdeheu4zf.cloudfront.net/MERFISH/benchmark_results.csv', 
                    dtype = {'barcode':object})
x_cnts = res.groupby('gene').count()['area']
y_cnts = bench.groupby('gene').count()['area']
tmp = pd.concat([x_cnts, y_cnts], axis=1, join='inner').values
r = np.corrcoef(tmp[:,1], tmp[:,0])[0,1]

x = np.linspace(50, 2000)
plt.scatter(tmp[:,1],tmp[:,0], 50,zorder=2)
plt.plot(x,x,'-k',zorder=1)

plt.xlabel('Gene copy number Benchmark')
plt.ylabel('Gene copy number Starfish')
plt.xscale('log')
plt.yscale('log')
plt.title('r = {}'.format(r))

sns.despine(offset=2)
# EPY: END code

# EPY: START markdown
# ## Visualize results
# 
# This image applies a pseudo-color to each gene channel to visualize the position and size of all called spots in a subset of the test image
# EPY: END markdown

# EPY: START code
props = decoded.spot_props
area_lookup = lambda x: 0 if x == 0 else props[x-1].area
vfunc = np.vectorize(area_lookup)
mask = vfunc(decoded.label_img)
image((decoded.decoded_img*(mask > 2))[200:500,200:500], cmap = 'nipy_spectral', size=10)
# EPY: END code
