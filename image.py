# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:24:37 2020

@author: Eduardo
"""
import os
import numpy as np
import skimage.io as io
from astropy.io import fits
import matplotlib.pyplot as plt
from PIL import Image
class ImageManipulation:
    def create_directory(self, path):
        try:
            os.stat(path)
        except:
            os.mkdir(path)
    def read_tiff(self, path):
        im = Image.open(path)
        out = np.array(im)
        img = out.astype('uint16')
        self.image = img
        return img
    def read_fits(self, path):
        im = fits.open(path)
        header = im[1].header
        im = im[1].data
        out = np.array(im)
        out = out.astype('uint16')
        self.image = out
        return out, header
    def histogram(self, path_save, name, image, mask):
        import cv2
        NBINS = 256
        #histogram, bin_edges = np.histogram(image, bins=NBINS, range=(0, 2000))
        image = image / image.max()
        mask = mask / mask.max()
        mask = mask.astype("uint8")
        histogram = cv2.calcHist([image],[0],None,[NBINS],[0,1])
        hist_mask = cv2.calcHist([image],[0],mask,[NBINS],[0,1])
        plt.figure()
        plt.title("Grayscale histogram of " + name)
        plt.xlabel("Grayscale value")
        plt.ylabel("Number of pixels")
        plt.xlim([0.0, 256.0])
        plt.plot(histogram)
        plt.plot(hist_mask)
        plt.savefig(path_save + "/" + name + "_histogram.tif")
        plt.close()
    def crop_images(self, path_img_complete, path_save, crop_size):
        import os
        from PIL import Image
        import skimage.io as io
        os.chdir(os.getcwd())
        images = os.listdir(path_img_complete)
        for img in images:
            imageha = Image.open(path_img_complete + "/" + img)
            width, height = imageha.size
            count = 0
            for i in range(0, height, crop_size):
                for j in range(0, width, crop_size):
                    box = (j, i, j + crop_size, i + crop_size)
                    crop = imageha.crop(box)
                    crop.save(path_save + "/" + img + "_corte_" + str(count) + ".tif")
                    cropped_image = Image.open(path_save + "/" + img + "_corte_" + str(count) + ".tif")
                    cropped_image = np.array(cropped_image)
                    imgint = cropped_image.astype('uint16')
                    io.imsave(path_save + "/" + img + "_corte_" + str(count) + ".tif", imgint)
                    count += 1
    def alphanumeric_order(self, lista):
        import re
        convertir = lambda texto: int(texto) if texto.isdigit() else texto.lower()
        alphanum_key = lambda key: [ convertir(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(lista, key=alphanum_key)
    def paste_images(self, dir_images, dir_save, width, height, tam):
        import skimage.io as io
        self.create_directory(dir_save)
        list_images = self.alphanumeric_order(os.listdir(dir_images))
        i = 1
        path_images_x = []
        images_y = []
        for img in list_images:
            path_images_x += [img]
            if i % (width / tam) == 0:
                concatenate_images = Image.fromarray(
                  np.concatenate(
                    [np.array(Image.open(dir_images + "/" + x)) for x in path_images_x],
                    axis=1 #Concatenate the images horizontal
                  )
                )
                images_y += [concatenate_images]
                path_images_x = []
            if i % ((height / tam) * (width / tam)) == 0:
                concatenate_images = Image.fromarray(
                  np.concatenate(
                    [np.array(images) for images in images_y],
                    axis=0 #Concatenate the images vertical
                  )
                )
                concatenate_images = np.array(concatenate_images)
                concatenate_images = concatenate_images.astype('uint16')
                images_y = []
                img = img.replace(".tif", "")
                io.imsave(os.path.join(dir_save, img + "_image_complete.tif"), concatenate_images)
            i += 1
    def prediction_to_image(self, path, path_save, percentage):
        from imutils import paths
        list_images = self.alphanumeric_order(list(paths.list_images(path)))
        for i, item in enumerate(list_images):
            prediction = self.read_tiff(item)
            prediction = prediction / 65535
            imgbool = prediction.astype('float')
            imgbool[imgbool > percentage] = 1
            imgbool[imgbool <= percentage] = 0
            image = imgbool.astype('uint16')
            image = image * 65535
            image.shape = prediction.shape
            io.imsave(os.path.join(path_save,  str(i) + "_stars_" + str(percentage) + ".tif"), image)
    def image_mask(self, original_path, pathmask, path_save):
        from imutils import paths
        list_images = self.alphanumeric_order(list(paths.list_images(original_path)))
        for i, item in enumerate(list_images):
            image = self.read_tiff(item)
            mask_name = item.replace(original_path + "\\", "")
            mask_name = mask_name.replace("interpolation_1000.tif", "interpolation_1000_predict_50.tif")
            mask = self.read_tiff(os.path.join(pathmask, mask_name))
            image = image / 65535 * 255
            mask = mask / 65535
            without_background = image * mask
            without_background = without_background.astype("uint8")
            io.imsave(os.path.join(path_save, mask_name), without_background)
    def remove_stars(self, original_path, path_predict, path_save, percentage):
        from imutils import paths
        #from preprocess import preprocess
        list_images = self.alphanumeric_order(list(paths.list_images(original_path)))
        for i, item in enumerate(list_images):
            imgoriginal = self.read_tiff(item)
            print(str(i) + item)
            img = item.replace(original_path + "\\", "")
            img = img.replace(".tif", "_predict.tif")
            print(str(i) + " " + path_predict + "/" + img)
            imgbool = self.read_tiff(os.path.join(path_predict, img))
            imgbool = imgbool / 65535
            #percentage_1_0 = percentage / 100
            imgbool[imgbool > percentage] = 1
            imgbool[imgbool <= percentage] = 0
            h, w = imgoriginal.shape 
            img_without_stars = imgoriginal * (1 - imgbool)
            img_without_stars.shape = imgoriginal.shape
            img_without_stars = img_without_stars.astype('uint16')
            img = item.replace(original_path + "\\", "") 
            img = img.replace(".tif.tif", "_predict.tif")
            print(str(i) + " " + path_save + "/" + img)
            io.imsave(os.path.join(path_save, img), img_without_stars)
            '''pp = preprocess(os.path.join(path_save,  str(i) + "_sin_stars_" + str(percentage) + ".tif"), True)
            pp.autocontrast(2, True)
            pp.save_image_tiff(os.getcwd() + "/" + path_save, str(i) + "_sin_stars_" + str(percentage) + "_percentile_range.tif", True)'''
    def paste_images(self, dirimgs, dirguar, num_imgs, width):
        add_width = width
        self.create_directory(dirguar)
        images = self.alphanumeric_order(os.listdir(dirimgs))
        img = Image.new("RGB", (2048,4096))
        xv = 0
        yv = 0
        k = 1
        save = (num_imgs / ((2048 / width) * (4096 / width)) + 1)
        for i in range(1, num_imgs + 1, 1):
            im = self.read_tiff(dirimgs + "/" + images[i - 1])
            if im.max() == 0:
                im = abs(im)
            else:
                im = im / im.max()
            for x in range(width):
                for y in range(width):
                    v = abs(im[x, y])
                    v = round(v * 255)
                    v = v.astype(int)
                    img.putpixel((xv + y, yv + x), (v, v, v))
            if i % (2048 / width) == 0:
                xv = 0
                yv = yv + add_width
            else:
                xv = xv + add_width
            if i % save == 0:
                xv = 0
                yv = 0
                img.save(dirguar + "/" + str(k) + "_complete.tif")
                k = k + 1
'''
import os
iobj = image()
pathfits = "train/masks_cropped"
pathimageres = "train/images_cropped_backup"
pathimage = "train/images_cropped"
images = os.listdir(pathfits)
for img in images:
    image = Image.open(pathimageres + "/" + img)
    io.imsave(pathimage + "/" + img, image)

import os
iobj = image()
pathfits = "train/masks_cropped"
pathimage = "train/images_cropped"
pathimgmask = "train/images_masks"
images = os.listdir(pathfits)
#data = []
#imagesdata = []
for img in images:
    #image = iobj.read_tiff(pathfits + "/" + img)
    image = Image.open(pathfits + "/" + img)
    #imagesdata.append(np.array(image))
    extrema = image.convert("L").getextrema()
    #data.append(extrema)
    print(img)
    if extrema == (0, 0): 
        print("Todos los pixeles son negros")
        image.close()
        os.remove(pathfits + "/" + img)
        os.remove(pathimgmask + "/" + img)
        img2 = img.replace("_predict_50", "")
        os.remove(pathimage + "/" + img2)
    else:
        print("No todos los pixeles son negros")

iobj = image()
path_image = "fits_g139_tif"
image_ha = iobj.read_tiff(path_image + "/r431413-1_PN_G139.0+03.2_Ha_2_600_x_600.tif")
image_i = iobj.read_tiff(path_image + "/r431415-1_PN_G139.0+03.2_i_2_600_x_600.tif")
image_ha_i = image_ha - image_i
io.imsave(path_image + "/r431414_5-1_ha_i.fits.tif", image_ha_i)

image_u = iobj.read_tiff(path_image + "/r763738-1_u.fits.tif")
image_g = iobj.read_tiff(path_image + "/r763739-1_g.fits.tif")
image_u_g = image_u - image_g
io.imsave(path_image + "/r763738_9-1_u_gfits.tif", image_u_g)

iobj = image()
path_image = "fits_g139"
path_result = "fits_g139_tif"
images = os.listdir(path_image)
for img in images:
    image, header = iobj.read_fits(path_image + "/" + img)
    io.imsave(path_result + "/" + img + ".tif", image)
iobj = image()
path_entrenamiento = "train/images_sin_fondo_512_threshold_2000"
path_masks = "train/masks_512_threshold_2000"
path_histogram = "train/images_sin_fondo_512_threshold_2000_histogram"
images = os.listdir(path_entrenamiento)
for img in images:
    image = iobj.read_tiff(path_entrenamiento + "/" + img)
    mask = iobj.read_tiff(path_masks + "/" + img)
    print(image.max())
    iobj.histogram(path_histogram, img, image, mask)

import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
i = image()
img, header = i.read_fits("fits/r750456-1_PN_NAME_LDU_1_PN_K_3-83_Ha.fits")
img2 = img.copy().astype('uint16')
print("El máximo es: " + str(img.max()))
io.imsave("fits/r750456-1_PN_NAME_LDU_1_PN_K_3-83_Ha_original.tif", img2)
img = img / 65535
threshold = 25000 / 65535
mask = img.copy().astype(float)
mask[mask >= threshold] = np.nan
mask2 = mask.copy().astype(float)
mask2 = mask2 * 65535
mask2 = mask2.astype('uint16')
nans = np.count_nonzero(np.isnan(mask))
print("Numero de nans: " + str(nans))
print("El máximo es: " + str(mask2.max()))
io.imsave("fits/r750456-1_PN_NAME_LDU_1_PN_K_3-83_Ha_mask.tif", mask2)
#mask = mask.astype('uint16')
kernel = Gaussian2DKernel(1)
reconstructed_image = interpolate_replace_nans(mask, kernel)
nans = np.count_nonzero(np.isnan(reconstructed_image))
print("Numero de nans: " + str(nans))
reconstructed_image = reconstructed_image * 65535
reconstructed_image = reconstructed_image.astype('uint16')
io.imsave("fits/r750456-1_PN_NAME_LDU_1_PN_K_3-83_Ha.tif", reconstructed_image)

original_path = "train/images_cropped"
pathmask = "train/masks_cropped"
path_save = "train/images_masks"
iobj = image()
iobj.image_mask(original_path, pathmask, path_save)

dir_images = "train/images"
dir_images_512 = "train/images_cropped"
iobj = image()
iobj.crop_images(dir_images, dir_images_512, 512)

dir_images = "train/masks"
dir_images_512 = "train/masks_cropped"
iobj = image()
iobj.crop_images(dir_images, dir_images_512, 512)

dir_images = "train/masks_HII"
dir_images_512 = "train/masks_HII_512_ok"
iobj = image()
iobj.crop_images(dir_images, dir_images_512, 512)

iobj = image()
pathfits = "train/masks_512_ok"
pathtif = "train/masks_512_100_img"
images = os.listdir(pathfits)
for img in images:
    image = iobj.read_tiff(pathfits + "/" + img)
    io.imsave(os.path.join(pathtsi, img + ".tif"), image)

dir_images = "entrenamiento_stars//tsi_original_en_dr2"
dir_images_512 = "entrenamiento_stars//tsi_original_cropped_en_dr2"
iobj = image()
iobj.crop_images(dir_images, dir_images_512, 512)

dir_images = "entrenamiento_stars//tsi_en_dr2_mask"
dir_images_512 = "entrenamiento_stars//tsi_cropped_en_dr2_mask"
iobj.crop_images(dir_images, dir_images_512, 512)

dir_images = "entrenamiento_stars//tsi_original_no_en_dr2"
dir_images_512 = "entrenamiento_stars//tsi_original_cropped_no_en_dr2"
iobj = image()
iobj.crop_images(dir_images, dir_images_512, 512)
iobj = image()
dir_images = "entrenamiento_stars//tsi_en_dr2_mask"
dir_images_512 = "entrenamiento_stars//tsi_cropped_en_dr2_mask"
iobj.crop_images(dir_images, dir_images_512, 512)
dir_images = "entrenamiento_stars//tsi_no_en_dr2_mask"
dir_images_512 = "entrenamiento_stars//tsi_cropped_no_en_dr2_mask"
iobj.crop_images(dir_images, dir_images_512, 512)
dir_images = "train//images_pr"
#dir_images_1024 = "train//images_min_max_1024"
dir_images_512 = "train//images_pr_512"
dir_masks = "train//masks_pr"
#dir_masks_1024 = "train//masks_min_max_1024"
dir_masks_512 = "train//masks_pr_512"
iobj = image()
#iobj.crop_images(dir_images, dir_images_1024, 1024)
iobj.crop_images(dir_images, dir_images_512, 512)
#iobj.crop_images(dir_masks, dir_masks_1024, 1024)
iobj.crop_images(dir_masks, dir_masks_512, 512)
dir_images = "test//images_completes"
dir_images_512 = "test//images_cropped"
iobj = image()
iobj.crop_images(dir_images, dir_images_512, 512)'''