import numpy as np
import sys, os

caffe_root = '/home/caffe/'
sys.path.insert(0, caffe_root + 'python')

import caffe

proto = "/home/batNarayan/_CV.ROS_TEMP_Lane_Det_ColTyp_Camera_CNN_Node/camera_lane_detection/models/unified_lane_detection_color_type/deploy.prototxt"
weights = "/home/batNarayan/_CV.ROS_TEMP_Lane_Det_ColTyp_Camera_CNN_Node/camera_lane_detection/models/unified_lane_detection_color_type/weights.caffemodel"
lane_color_net = caffe.Net(proto, weights ,caffe.TRAIN)

for layer in lane_color_net._layer_names:
    print(layer)
print(len(lane_color_net._layer_names))

proto = "/home/batNarayan/scnn_A_60k/prototxts/deploy_scnn_A.prototxt"
weights = "/home/batNarayan/scnn_A_60k/snapshots/solver_iter_500000.caffemodel"
lane_freespace_net = caffe.Net(proto, weights ,caffe.TRAIN)

for layer in lane_freespace_net._layer_names:
    print(layer)
    try:
        W = lane_freespace_net.params[layer][0].data[...]
        b = lane_freespace_net.params[layer][1].data[...]
    except:
        print("no weights for {}".format(layer))

road_net = caffe.Net("/home/batNarayan/roadnet.prototxt",caffe.TRAIN)

print("freespace: ")
print(len(lane_freespace_net._layer_names))
print("color_type: ")
print(len(lane_color_net._layer_names))


print("road net: ")
print(len(road_net._layer_names))

for layer in road_net._layer_names:
    w = None
    b = None
    try:
        w = lane_color_net.params[layer][0].data[...]
        b = lane_color_net.params[layer][1].data[...]
        print("{} -- color".format(layer))
        road_net.params[layer][0].data[...] = w
        road_net.params[layer][1].data[...] = b
    except:
        try:
            w = lane_freespace_net.params[layer][0].data[...]
            b = lane_freespace_net.params[layer][1].data[...]
            print("{} -- freespace".format(layer))
            road_net.params[layer][0].data[...] = w
            road_net.params[layer][1].data[...] = b
        except:
            print("{} -- :( ".format(layer))

road_net.save('/home/batNarayan/roadnet.caffemodel')

