import caffe
import lmdb
from PIL import Image
import cStringIO as StringIO
import numpy as np
import cv2

def show_bbox(img, xmin, ymin, xmax, ymax, window_name="image"):
	for i in range(0,len(xmin)):
		x_min = int(xmin[i])
		y_min = int(ymin[i])
		x_max = int(xmax[i])
		y_max = int(ymax[i])
		cv2.rectangle(img, (x_min,y_min),(x_max,y_max),(0,255,0),2)
	# cv2.namedWindow(window_name,)
	cv2.imshow(window_name,img)
	cv2.waitKey(0)

with lmdb.open('/home/admin1/combined_lmdb') as lmdb_env:
	lmdb_txn = lmdb_env.begin()
	lmdb_cursor = lmdb_txn.cursor()
	ad = caffe.proto.caffe_pb2.AnnotatedDatum()

	for key, value in lmdb_cursor:
		ad.ParseFromString(value)
		print(ad.datum.channels, ad.datum.height, ad.datum.width, ad.segmentation_label.channels, ad.segmentation_label.height, ad.segmentation_label.width)
		# Read image data
		# first read the ".png" encoded file
		filestream = StringIO.StringIO(ad.datum.data)
		# read image from filestream
		pil_image = Image.open(filestream) 
		cv_image = np.array(pil_image)
		# Convert RGB to BGR 
		cv_image = cv_image[:, :, ::-1].copy()

		# Read Segmentation mask
		filestream = StringIO.StringIO(ad.segmentation_label.data)
		pil_segmentation_mask = Image.open(filestream)
		cv_seg = np.array(pil_segmentation_mask)
		#cv_seg = cv_seg[:,:,::-1].copy()

		#cv2.imshow("image", cv_image);
		cv2.imshow("segmentation_mask", cv_seg);
		#cv2.waitKey(0)
		#image.show()

		xmin = []
		ymin = []
		xmax = []
		ymax = []	

		for group in ad.annotation_group:
			label_id = group.group_label
			for objects in group.annotation:
				box = objects.bbox
				xmin.append(float(box.xmin) * 304)
				ymin.append(float(box.ymin) * 304)
				xmax.append(float(box.xmax) * 304)
				ymax.append(float(box.ymax) * 304)
				print(box.xmin, box.xmax, box.ymin, box.ymax)	
 		show_bbox(cv_image, xmin, ymin, xmax, ymax)		
		
		break	
		print(key)
