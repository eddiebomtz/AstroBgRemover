# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 21:00:13 2020

@author: eduardo
"""

import os
from image import image
imageobj = image()
for count, path_name in enumerate(os.listdir("train/fits")): 
    image, header = imageobj.read_fits("train/fits/" + path_name)
    run = header["run"]
    imageid = header["imageid"]
    if not path_name.startswith("r"):
        print("Renaming: " + str(run) + "-" + str(imageid) + "_" + path_name)
        nueva_path_name = "r" + str(run) + "-" + str(imageid) + "_" + path_name
        os.rename("train/fits/" + path_name, "train/fits/" + nueva_path_name) 
    else:
        continue
    #dst ="Hostel" + str(count) + ".jpg"
    #src ='xyz'+ filename 
    #dst ='xyz'+ dst 
    #os.rename(src, dst) 