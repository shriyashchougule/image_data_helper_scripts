import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage import io
import copy
import glob
import os


label_directory = '/home/rosrepo/Pictures/baidu/DataLabels'
write_directory = '/home/rosrepo/Pictures/baidu/DataLabels_polished'
for filename in glob.iglob(label_directory + '/*.png'):
	print(filename)
	#print(os.path.basename(filename))
	image = io.imread(filename)
	# apply threshold
	thresh = threshold_otsu(image)
	bw = closing(image > thresh, square(3))

	# remove artifacts connected to image border
	#cleared = clear_border(bw)
	cleared = bw
	# label image regions
	label_image = label(cleared)
	#print(np.unique(label_image))
	tmp = copy.copy(label_image)
	road_labels = np.unique(label_image[1150, 601:1200])
	#print(road_labels)
	image_label_overlay = label2rgb(label_image, image=image)

	tmp = tmp.astype('uint8')
	for l in road_labels:
		if l != 0:
			tmp[tmp == l] = 1
	#print(np.unique(tmp))
	tmp[tmp !=1] = 0
	#print(np.unique(tmp))
	#fig1, ax1 = plt.subplots(figsize=(10, 6))
	#ax1.imshow(tmp)
	#plt.show()
	io.imsave(write_directory + '/' + os.path.basename(filename), tmp)
