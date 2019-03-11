#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is evaluation script for freespace ENet.
"""
import os
import numpy as np
from argparse import ArgumentParser
from os.path import join
import argparse
import sys
import Metrics_V1 as Metrics
from eval_segm import mean_IU
caffe_root = '/home/rosrepo/Document/rtc_ws_2/Caffe_Src/caffe/'  # Change this to the absolute directory to ENet Caffe
sys.path.insert(0, caffe_root + 'python')
import caffe
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import matplotlib.pyplot as plt
#from statistics import mean 



def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='.prototxt file for inference')
    parser.add_argument('--weights', type=str, required=True, help='.caffemodel file')
    parser.add_argument('--colours', type=str, required=True, help='label colours')
    parser.add_argument('--mapping_file', type=str, required=True, help='image and ground truth mapping file')
    parser.add_argument('--out_dir', type=str, default=None, help='output directory in which the segmented images '
                                                                   'should be stored')
    parser.add_argument('--gpu', type=str, default='0', help='0: gpu mode active, else gpu mode inactive')

    return parser


if __name__ == '__main__':
    parser1 = make_parser()
    args = parser1.parse_args()
    if args.gpu == 0:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

	label_colours = cv2.imread(args.colours, 1).astype(np.uint8)

    net = caffe.Net(args.model, args.weights, caffe.TEST)

    input_shape = net.blobs['data'].data.shape
    output_shape = net.blobs['deconv6_0_0'].data.shape

    MIoU = []
    count = 1
    with open(args.mapping_file) as f:
		for line in f:
		# you may also want to remove whitespace characters like `\n` at the end of each line
			line = line.rstrip()
			image_path, gt_path = line.split()
			#print(line)
			#print(gt_path)

			input_image = cv2.imread(image_path, 1).astype(np.float32)
			gt_image    = cv2.imread(gt_path,cv2.IMREAD_GRAYSCALE)

			input_image = cv2.resize(input_image, (input_shape[3], input_shape[2]))
			input_image = input_image.transpose((2, 0, 1))
			input_image = np.asarray([input_image])
			gt_image = cv2.resize(gt_image, (output_shape[3], output_shape[2]))

			out = net.forward_all(**{net.inputs[0]: input_image})

			prediction_argmax = net.blobs['deconv6_0_0'].data[0].argmax(axis=0)
			prediction = np.squeeze(prediction_argmax)
			prediction = np.resize(prediction, (3, output_shape[2], output_shape[3]))
			prediction = prediction.transpose(1, 2, 0).astype(np.uint8)

			Mean_IntersectionOverUnionAccuracy = Metrics.Mean_IntersectionOverUnion(prediction_argmax,gt_image)
			miou = mean_IU(prediction_argmax,gt_image)
			print(" {} MIoU : {} {} {}".format(count, Mean_IntersectionOverUnionAccuracy, miou, Mean_IntersectionOverUnionAccuracy - miou ))
			MIoU.append(Mean_IntersectionOverUnionAccuracy)

			prediction_rgb = np.zeros(prediction.shape, dtype=np.uint8)
			label_colours_bgr = label_colours[..., ::-1]
			cv2.LUT(prediction, label_colours_bgr, prediction_rgb)
			#cv2.imshow("ENet", prediction_rgb)
			#key = cv2.waitKey(0)
			count = count + 1
    print("Mean MIoU for validation dataset: {}".format(sum(MIoU)/float(len(MIoU))))
    mean_value = " mean : {}".format(sum(MIoU)/float(len(MIoU)))
	# An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=MIoU, bins=100, color='#0504aa',alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('MIoU Histogram enet_v1 ' + mean_value )
    stats = "Mean {}".format(sum(MIoU)/float(len(MIoU)))
    plt.text(23, 45, stats)
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()
