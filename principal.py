# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:48:09 2019

@author: eduardo
"""
import os
import argparse
from image import ImageManipulation
from preprocess import preprocess
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
parser = argparse.ArgumentParser(description='AstroIMP.')
parser.add_argument("-c", "--crop", action="store_true", help="Specifies if crop images")
parser.add_argument("-p", "--preprocess", action="store_true", help="Specifies if preprocess the images")
parser.add_argument("-ef", "--remove_background", action="store_true", help="Specifies if remove the background")
parser.add_argument("-zs", "--zscale", action="store_true", help="Specifies if applies contrast enhancement, using the algorithm zscale")
parser.add_argument("-pr", "--percentile_range", action="store_true", help="Specifies if applies contrast enhancement, using the algorithm percentile range")
parser.add_argument("-ap", "--arcsin_percentile", action="store_true", help="Specifies if applies contrast enhancement, using the algorithm arcsin percentile")
parser.add_argument("-apr", "--arcsin_percentile_range", action="store_true", help="Specifies if applies contrast enhancement, using the algorithm arcsin percentile range")
parser.add_argument("-pf", "--pfcm", action="store_true", help="Specifies if remove the background using the algorithm PFCM")
parser.add_argument("-ccde", "--ccd_errors", action="store_true", help="Specifies if you want to reduce CCD errors")
parser.add_argument("-d" , "--dir_images", action="store", dest="dir_images", help="Input directory")
parser.add_argument("-r", "--dir_result", action="store", dest="dir_result", help="Output directory")
args = parser.parse_args()
if args.preprocess:
    print("Preprocess...")
    images = os.listdir(args.dir_images)
    type = ""
    use_processed = False
    for img in images:
        print(img)
        path = os.path.splitext(img)
        if path[1] == ".fits":
            pp = preprocess(args.dir_images + "/" + img, False)
        else:
            pp = preprocess(args.dir_images + "/" + img, True)
        pp.save_image_tiff(os.getcwd() + "/" + args.dir_result, img + "_original", True)
        if args.remove_background:
            print("Removing background...")
            pp.remove_background()
            pp.save_image_tiff(os.getcwd() + "/" + args.dir_result, img + "_sin_fondo.tif", True) 
        if args.pfcm:
            pp.pfcm_2(args.dir_result, img)
            use_processed = True
            pp.save_image_tiff(os.getcwd() + "/" + args.dir_result, img + "_pfcm.tif", True)
        if args.ccd_errors:
            pp.ccd_error(args.dir_images, args.dir_result)
            use_processed = True
            
        if args.zscale:
            print("Contrast enhancement with zscale...")
            pp.autocontrast(1, use_processed)
            pp.save_image_tiff(os.getcwd() + "/" + args.dir_result, img + "_zscale_" + type, True)
            #type = "zscale"
        elif args.percentile_range:
            print("Contrast enhancement with percentile range...")
            #type = "percentile_range"
            pp.autocontrast(2, use_processed)
            pp.save_image_tiff(os.getcwd() + "/" + args.dir_result, img + "_percentile_" + type, True)
        elif args.arcsin_percentile:
            print("Contrast enhancement with arcsin percentile...")
            #type = "arcsin_percentile"
            pp.autocontrast(3, use_processed)
            pp.save_image_tiff(os.getcwd() + "/" + args.dir_result, img + "_arcsin_percentile_" + type, True)
        elif args.arcsin_percentile_range:
            print("Contrast enhancement with arcsin percentile range...")
            #type = "arcsin_percentile_range"
            pp.autocontrast(4, use_processed)
            pp.save_image_tiff(os.getcwd() + "/" + args.dir_result, img + "_arcsin_percentile_range_" + type, True)
elif args.crop:
    print("Crop images...")
    iobj = ImageManipulation()
    iobj.crop_images(args.dir_images, args.dir_result, 512)