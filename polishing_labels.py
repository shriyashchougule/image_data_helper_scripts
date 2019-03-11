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
#image = data.coins()[50:-50, 50:-50]
image = io.imread("/home/rosrepo/Pictures/baidu/DataLabels/6829.png")
# apply threshold
thresh = threshold_otsu(image)
bw = closing(image > thresh, square(3))

# remove artifacts connected to image border
#cleared = clear_border(bw)
cleared = bw
# label image regions
label_image = label(cleared)
print(np.unique(label_image))
tmp = copy.copy(label_image)
road_labels = np.unique(label_image[1150, 601:1200])
print(road_labels)
image_label_overlay = label2rgb(label_image, image=image)

tmp = tmp.astype('uint8')
for l in road_labels:
	if l != 0:
		tmp[tmp == l] = 255
#print(np.unique(tmp))
tmp[tmp !=255] = 0
#print(np.unique(tmp))
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.imshow(tmp)

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image_label_overlay)

for region in regionprops(label_image):
    # take regions with large enough areas
    if region.area >= 100:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

ax.set_axis_off()
plt.tight_layout()
plt.show()

