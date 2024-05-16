# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:56:59 2019

@author: eduardo
"""
import os
import cmath
import warnings
import pointarray
import skfuzzy as fuzz
import skimage.io as io
from PFCM import PFCM
from image import image
from scipy import ndimage
import skimage.exposure as skie
from skimage.filters import gaussian
from astropy.stats import mad_std
import numpy as np
class preprocess:
    def __init__(self, path, tiff):
        if path != None:
            self.imageobj = image()
            if tiff:
                self.image = self.imageobj.read_tiff(path)
            else:
                self.image, self.header = self.imageobj.read_fits(path)
            self.processed_image = self.image
    def image_to_process(self, image):
        self.image = image
        self.processed_image = self.image
    def normalize_image(self, type):
        img = self.processed_image
        if type == 1:
            img = 2 * ((img - img.min()) / (img.max() - img.min())) - 1
        elif type == 2:
            img = (img - img.min()) / (img.max() - img.min()) * 65535
            img = img.astype("uint16")
            self.processed_image = img
        return img
    def ccd_error(self, dir_images, dir_result):
        from skimage.morphology import closing, disk
        images = os.listdir(dir_images)
        for img in images:
            print(img)
            image = self.image
            I = image.max() - image
            I = I.astype("uint16")
            io.imsave(dir_result + "/" + img + "_inverted.tif", I)
            new_image = I - image
            new_image = new_image.astype("uint16")
            io.imsave(dir_result + "/" + img + "_subtraction_inverted.tif", new_image)
            selem = disk(8)
            closed = closing(new_image, selem)
            io.imsave(dir_result + "/" + img + "_inverted_close_operation.tif", closed)
            result = closed / image.max() * image
            result = result.astype("uint16")
            io.imsave(dir_result + "/" + img + "_inverted_again_close_operation.tif", result)
            result_subtraction = result - image
            result_subtraction = result_subtraction.astype("uint16")
            io.imsave(dir_result + "/" + img + "_inverted_again_close_operation_subtraction.tif", result_subtraction)
            self.processed_image = result_subtraction / image.max() * image
            new_image = self.processed_image.astype("uint16")
            io.imsave(dir_result + "/" + img + "_inverted_again_close_operation_subtraction_dynamic_range.tif", new_image)
    def remove_background_mask(self, num_sigma):
        '''WARNING THIS ONLY WORKS WITH IPHAS ORIGINAL IMAGES IN FITS FORMAT'''
        from astropy.io import fits
        fits_table = fits.open('iphas-images.fits')
        self.run = self.header["run"]
        self.imageid = self.header["imageid"]
        data = fits_table[1].data
        runs = np.array(data["run"])
        valids = np.array(np.where(runs == self.run))
        idrun = valids.flat[self.imageid - 1]
        original = self.processed_image
        sigma = self.sigma()
        mask = original.copy().astype(float)
        skylevel = data["skylevel"][idrun]
        skynoise = data["skynoise"][idrun]
        threshold = skylevel + skynoise + (sigma * num_sigma)
        mask[mask <= threshold] = 0
        mask[mask > 0] = 65535
        mask = mask.astype('uint16')
        self.processed_image = mask
    def remove_background(self):
        from astropy.io import fits
        fits_table = fits.open('iphas-images.fits')
        self.run = self.header["run"]
        self.imageid = self.header["imageid"]
        data = fits_table[1].data
        runs = np.array(data["run"])
        valids = np.array(np.where(runs == self.run))
        idrun = valids.flat[self.imageid - 1]
        original = self.processed_image
        sigma = self.sigma()
        mask = original.copy().astype(float)
        skylevel = data["skylevel"][idrun]
        skynoise = data["skynoise"][idrun]
        threshold = skylevel + skynoise + (sigma * 3)
        mask[mask <= threshold] = 0
        mask[mask > 0] = 1
        mask = mask.astype('uint16')
        original = self.processed_image * mask
        original = original.astype('uint16')
        self.processed_image = original
    def remove_background_tif(self):
        threshold = 2000
        original = self.processed_image
        mask = original.copy().astype(float)
        mask[mask >= threshold] = threshold
        mask = mask.astype('uint16')
        self.processed_image = mask
    def interpolation_saturated(self, threshold):
        from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
        img = self.processed_image
        img = img / 65535
        threshold = threshold / 65535
        mask = img.copy().astype(float)
        mask[mask >= threshold] = np.nan
        mask2 = mask.copy().astype(float)
        mask2 = mask2 * 65535
        mask2 = mask2.astype('uint16')
        nans = np.count_nonzero(np.isnan(mask))
        kernel = Gaussian2DKernel(1)
        reconstructed_image = interpolate_replace_nans(mask, kernel)
        while nans > 0:
            reconstructed_image = interpolate_replace_nans(reconstructed_image, kernel)
            nans = np.count_nonzero(np.isnan(reconstructed_image))
        reconstructed_image = reconstructed_image * 65535
        reconstructed_image = reconstructed_image.astype('uint16')
        self.processed_image = reconstructed_image
    def remove_blooming(self):
        original = self.image
        mask = original.copy().astype(float)
        mask[mask >= 32500.0] = 0
        mask[mask > 0] = 1
        mask = mask.astype('uint16')
        original = self.image * mask
        self.processed_image = original
    def sigma(self):
        sigma = mad_std(self.processed_image)
        return sigma
    def anisotropic_difussion(self,img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),sigma=0,option=1,ploton=False):
        """
        # Original work: Copyright (c) 1995-2012 Peter Kovesi pk@peterkovesi.com
        # Modified work: Copyright (c) 2012 Alistair Muldal
        #
        # Permission is hereby granted, free of charge, to any person obtaining a copy
        # of this software and associated documentation files (the "Software"), to deal
        # in the Software without restriction, including without limitation the rights
        # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        # copies of the Software, and to permit persons to whom the Software is
        # furnished to do so, subject to the following conditions:
        #
        # The above copyright notice and this permission notice shall be included in all
        # copies or substantial portions of the Software.
        #
        # The software is provided "as is", without warranty of any kind, express or
        # implied, including but not limited to the warranties of merchantability,
        # fitness for a particular purpose and noninfringement. In no event shall the
        # authors or copyright holders be liable for any claim, damages or other
        # liability, whether in an action of contract, tort or otherwise, arising from,
        # out of or in connection with the software or the use or other dealings in the
        # software.
        Anisotropic difussion.
        Usage:
        imgout = anisotropic_difussion(im, niter, kappa, gamma, option)
        Arguments:
                img    - input image
                niter  - number of iterations
                kappa  - conduction coefficient 20-100 ?
                gamma  - max value of .25 for stability
                step   - tuple, the distance between adjacent pixels in (y,x)
                option - 1 Perona Malik difussion equation No 1
                         2 Perona Malik difussion equation No 2
                ploton - if True, the image will be plotted on every iteration
        Returns:
                imgout   - dsifused image.
        kappa controls conduction as a function of gradient.  if kappa is low
        small intensity gradients are able to block conduction and hence difussion
        across step edges.  A large value reduces the influence of intensity
        gradients on conduction.
        gamma controls speed of difussion (you usually want it at a maximum of
        0.25)
        step is used to scale the gradients in case the spacing between adjacent
        pixels dsifers in the x and y axes
        difussion equation 1 favours high contrast edges over low contrast ones.
        difussion equation 2 favours wide regions over smaller ones.
        Reference: 
        P. Perona and J. Malik. 
        Scale-space and edge detection using anisotropic difussion.
        IEEE Transactions on Pattern Analysis and Machine Intelligence, 
        12(7):629-639, July 1990.
        Original MATLAB code by Peter Kovesi
        School of Computer Science & Software Engineering
        The University of Western Australia
        pk @ csse uwa edu au
        <http://www.csse.uwa.edu.au>

        Translated to Python and optimised by Alistair Muldal
        Department of Pharmacology
        University of Oxford
        <alistair.muldal@pharm.ox.ac.uk>

        June 2000  original version.
        March 2002 corrected diffusion eqn No 2.
        July 2012 translated to Python
        """
        if img.ndim == 3:
            warnings.warn("Only grayscale images allowed, converting to 2D matrix")
            img = img.mean(2)
        img = img.astype('float64')
        imgout = img.copy()
        deltaS = np.zeros_like(imgout)
        deltaE = deltaS.copy()
        NS = deltaS.copy()
        EW = deltaS.copy()
        gS = np.ones_like(imgout)
        gE = gS.copy()
        if ploton:
            import pylab as pl
            fig = pl.figure(figsize=(20,5.5),num="Anisotropic difussion")
            ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)
            ax1.imshow(img,interpolation='nearest')
            ih = ax2.imshow(imgout,interpolation='nearest',animated=True)
            ax1.set_title("Original image")
            ax2.set_title("Iteration 0")
            fig.canvas.draw()
        for ii in np.arange(1,niter):
            deltaS[:-1,: ] = np.dsif(imgout,axis=0)
            deltaE[: ,:-1] = np.dsif(imgout,axis=1)
            if 0<sigma:
                deltaSf=gaussian(deltaS,sigma);
                deltaEf=gaussian(deltaE,sigma);
            else: 
                deltaSf=deltaS;
                deltaEf=deltaE;	
            if option == 1:
                gS = np.exp(-(deltaSf/kappa)**2.)/step[0]
                gE = np.exp(-(deltaEf/kappa)**2.)/step[1]
            elif option == 2:
                gS = 1./(1.+(deltaSf/kappa)**2.)/step[0]
                gE = 1./(1.+(deltaEf/kappa)**2.)/step[1]
            E = gE*deltaE
            S = gS*deltaS
            NS[:] = S
            EW[:] = E
            NS[1:,:] -= S[:-1,:]
            EW[:,1:] -= E[:,:-1]
            imgout += gamma*(NS+EW)
            if ploton:
                iterstring = "Iteration %i" %(ii+1)
                ih.set_data(imgout)
                ax2.set_title(iterstring)
                fig.canvas.draw()
        self.processed_image = imgout
        return imgout
    def zscale_range(self, contrast=0.25, num_points=600, num_per_row=120):
        #print("::: Calculando valor minimo y maximo con zscale range :::")
        if len(self.image.shape) != 2:
            raise ValueError("input data is not an image")
        if contrast <= 0.0:
            contrast = 1.0
        if num_points > np.size(self.image) or num_points < 0:
            num_points = 0.5 * np.size(self.image)
        num_per_col = int(float(num_points) / float(num_per_row) + 0.5)
        xsize, ysize = self.image.shape
        row_skip = float(xsize - 1) / float(num_per_row - 1)
        col_skip = float(ysize - 1) / float(num_per_col - 1)
        data = []
        for i in range(num_per_row):
            x = int(i * row_skip + 0.5)
            for j in range(num_per_col):
                y = int(j * col_skip + 0.5)
                data.append(self.image[x, y])
        num_pixels = len(data)
        data.sort()
        data_min = min(data)
        data_max = max(data)
        center_pixel = (num_pixels + 1) / 2
        if data_min == data_max:
            return data_min, data_max
        if num_pixels % 2 == 0:
            center_pixel = round(center_pixel)
            median = data[center_pixel - 1]
        else:
            median = 0.5 * (data[center_pixel - 1] + data[center_pixel])
        pixel_indexes = map(float, range(num_pixels))
        points = pointarray.PointArray(pixel_indexes, data, min_err=1.0e-4)
        fit = points.sigmaIterate()
        num_allowed = 0
        for pt in points.allowedPoints():
            num_allowed += 1
        if num_allowed < int(num_pixels / 2.0):
            return data_min, data_max
        z1 = median - (center_pixel - 1) * (fit.slope / contrast)
        z2 = median + (num_pixels - center_pixel) * (fit.slope / contrast)
        if z1 > data_min:
            zmin = z1
        else:
            zmin = data_min
        if z2 < data_max:
            zmax = z2
        else:
            zmax = data_max
        if zmin >= zmax:
            zmin = data_min
            zmax = data_max
        return zmin, zmax
    def arcsin_percentile(self, min_percent=3.0, max_percent=99.0):
        img = self.processed_image
        limg = np.arcsinh(img)
        limg = limg / limg.max()
        low = np.percentile(limg, min_percent)
        high = np.percentile(limg, max_percent)
        return limg, low, high
    def percentile_range(self, min_percent=3.0, max_percent=99.0, num_points=5000, num_per_row=250):
        #print("::: Calculando valor minimo y maximo con percentile :::")
        if not 0 <= min_percent <= 100:
            raise ValueError("invalid value for min percent '%s'" % min_percent)
        elif not 0 <= max_percent <= 100:
            raise ValueError("invalid value for max percent '%s'" % max_percent)
        min_percent = float(min_percent) / 100.0
        max_percent = float(max_percent) / 100.0
        if len(self.image.shape) != 2:
            raise ValueError("input data is not an image")
        if num_points > np.size(self.image) or num_points < 0:
            num_points = 0.5 * np.size(self.image)
        num_per_col = int(float(num_points) / float(num_per_row) + 0.5)
        xsize, ysize = self.image.shape
        row_skip = float(xsize - 1) / float(num_per_row - 1)
        col_skip = float(ysize - 1) / float(num_per_col - 1)
        data = []
        for i in range(num_per_row):
            x = int(i * row_skip + 0.5)
            for j in range(num_per_col):
                y = int(j * col_skip + 0.5)
                data.append(self.image[x, y])
        data.sort()
        zmin = data[int(min_percent * len(data))]
        zmax = data[int(max_percent * len(data))]
        return zmin, zmax
    def autocontrast(self, type, original):
        #print("::: Processing image :::")
        zmin = 0
        zmax = 0
        limg = 0
        if type == 1:
            zmin, zmax = self.zscale_range()
            if original:
                self.processed_image = np.where(self.image > zmin, self.image, zmin)
                self.processed_image = np.where(self.processed_image < zmax, self.processed_image, zmax)
            else:
                self.processed_image = np.where(self.processed_image > zmin, self.processed_image, zmin)
                self.processed_image = np.where(self.processed_image < zmax, self.processed_image, zmax)
            nonlinearity = 3.0
            nonlinearity = max(nonlinearity, 0.001)
            max_asinh = cmath.asinh(nonlinearity).real
            self.processed_image = (self.image.max() / max_asinh) * (np.arcsinh((self.processed_image - zmin) * (nonlinearity / (zmax - zmin))))
        elif type == 2:
            zmin, zmax = self.percentile_range(min_percent=3.0, max_percent=99.5, num_points=6000, num_per_row=350)
            if original:
                self.processed_image = np.where(self.image > zmin, self.image, zmin)
                self.processed_image = np.where(self.processed_image < zmax, self.processed_image, zmax)
            else:
                self.processed_image = np.where(self.processed_image > zmin, self.processed_image, zmin)
                self.processed_image = np.where(self.processed_image < zmax, self.processed_image, zmax)
            self.processed_image = (self.processed_image - zmin) * (self.image.max() / (zmax - zmin))
        elif type == 3:
            limg, zmin, zmax = self.arcsin_percentile(min_percent=3.0, max_percent=99.5)
            self.processed_image = skie.exposure.rescale_intensity(limg, in_range=(zmin, zmax))
            self.processed_image = self.processed_image * self.image.max()
        elif type == 4:
            zmin, zmax = self.percentile_range(min_percent=3.0, max_percent=99.5, num_points=6000, num_per_row=350)
            if original:
                self.processed_image = np.where(self.image > zmin, self.image, zmin)
                self.processed_image = np.where(self.processed_image < zmax, self.processed_image, zmax)
            else:
                self.processed_image = np.where(self.processed_image > zmin, self.processed_image, zmin)
                self.processed_image = np.where(self.processed_image < zmax, self.processed_image, zmax)
            nonlinearity = 3.0
            nonlinearity = max(nonlinearity, 0.001)
            max_asinh = cmath.asinh(nonlinearity).real
            self.processed_image = (self.image.max() / max_asinh) * (np.arcsinh((self.processed_image - zmin) * (nonlinearity / (zmax - zmin))))
        elif type == 5:
            nonlinearity = 3.0
            nonlinearity = max(nonlinearity, 0.001)
            max_asinh = cmath.asinh(nonlinearity).real
            self.processed_image = (self.image.max() / max_asinh) * (np.arcsinh((self.image - self.image.min()) * (nonlinearity / (self.image.max() - self.image.min()))))
        elif type == 6:
            nonlinearity = 3.0
            nonlinearity = max(nonlinearity, 0.001)
            max_asinh = cmath.asinh(nonlinearity).real
            self.processed_image = (self.image.max() / max_asinh) * (np.arcsinh((self.processed_image - zmin) * (nonlinearity / (zmax - zmin))))
        self.processed_image = self.processed_image.astype('uint16')
        return limg, zmin, zmax
    def histogram(self):
        import cv2
        h = cv2.calcHist([self.image.ravel()], [0], None, [65536], [0,65536]) 
        return h
    def pfcm_2(self, path, name, anisotropic_difussion=True, median=False, gaussian=False):
        im = self.processed_image
        if anisotropic_difussion:
            I = 2 * ((im - im.min()) / (im.max() - im.min())) - 1
            sigma = mad_std(I)
            I = self.anisotropic_difussion(I,100,80,0.075,(1,1),sigma,2)
            I = (I - I.min()) / (I.max() - I.min())
            self.processed_image = self.processed_image.astype("uint16")
            self.save_image_tiff(path, name + "_anisotropic_difussion.tif", True)
        else:
            I = (im - im.min()) / (im.max() - im.min()) 
            if median:
                I = ndimage.median_filter(I, size=3)
            if gaussian:
                I = ndimage.gaussian_filter(I, 2)
        x, y = I.shape
        I = I.reshape(x * y, 1)
        pfcm = PFCM()
        centers, U, T, obj_fcn = pfcm.pfcm(I, 2, a = 1, b = 2, nc = 2)
        
        colors = []
        cluster_colors = {0:np.array([255,0,0]),1:np.array([0,255,0])}
        for n in range(I.shape[0]):
            color = np.zeros([2])
            for c in range(U.shape[0]):
                color += cluster_colors[c]*U[c,n]
            colors.append(color)
        
        labels = np.argmax(U, axis=0).reshape(im.shape[0], im.shape[1])
        I = I.reshape(im.shape[0], im.shape[1])
        label0 = I[labels == 0]
        label1 = I[labels == 1]
        maxlabel0 = np.max(label0)
        maxlabel1 = np.max(label1)
        labels[labels == 0] = 3
        labels[labels == 1] = 4
        if maxlabel0 < maxlabel1:
            labels[labels == 3] = 0
            labels[labels == 4] = 1
        else:
            labels[labels == 3] = 1 
            labels[labels == 4] = 0
        imglabel = labels.astype("uint16")
        imglabel16 = imglabel * 65535
        self.processed_image = imglabel16
        self.save_image_tiff(path, name + "_pfcm_binaria_", True)
        imglabel.shape = im.shape
        binary_image = imglabel * self.image
        binary_image = binary_image.astype("uint16")
        self.processed_image = binary_image
        self.save_image_tiff(path, name + "_pfcm", True)
    def dice(self, pred, true, k = 1):
        intersection = np.sum(pred[true==k]) * 2.0
        dice = intersection / (np.sum(pred) + np.sum(true))
        return dice
    def save_image_png(self, path, name):
        self.autocontrast(2, True)
        self.processed_image = 255 * (self.processed_image - self.processed_image.min()) / (self.processed_image.max() - self.processed_image.min())
        self.processed_image = self.processed_image.astype('uint8')
        io.imsave(path + "/" + name + ".png", self.processed_image)
        return self.processed_image
    def save_image_tiff(self, path, name, save_processed = False):
        if save_processed:
            io.imsave(path + "/" + name + ".tif", self.processed_image)
            return self.processed_image
        else:
            io.imsave(path + "/" + name + ".tif", self.image)
            return self.image
'''dir_images = "nebula_search"
dir_result = "nebula_search_autocontrast"
images = os.listdir(dir_images)
for img in images:
    print(img)
    pp = preprocess(dir_images + "/" + img, True)
    #pp.save_image_tiff(dir_result, img + "_original.tif", True)
    #pp.autocontrast(3, True)
    pp.save_image_png(dir_result, img)'''