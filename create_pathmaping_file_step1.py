import argparse
import os
import sys

def sizeof_dataset():
  return 945

def get_path(i, directory, extention):
  return directory + '/' + str(i) + '.' + extention

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Create maping file between image and labels")
  parser.add_argument("image_dir",
      help="The directory which contains the images.")
  parser.add_argument("annotations_dir",
      help="The directory which contains the bounding box information.")
  parser.add_argument("segmentation_dir",
      help="The directory which contains the segmenation labels for lanes-freespace.")
  parser.add_argument("output_filename",
      help="Name of output .txt file containing the paths.")

  args = parser.parse_args()

  for arg in vars(args):
    print arg, getattr(args, arg)

  
  with open(args.output_filename, "w+") as f:  
    for j in range(sizeof_dataset()):
      # image index start from 1
      i = j+1
      image_path = get_path(i, args.image_dir, "png")
      segmentation = get_path(i, args.segmentation_dir, "png")
      bounding_box_path = get_path(i, args.annotations_dir, "txt")
      #print(image_path, segmentation, bounding_box_path)
      f.write(image_path + " " + segmentation + " " + bounding_box_path + "\n")




