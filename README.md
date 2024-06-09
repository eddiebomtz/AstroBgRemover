# Astro Image Manipulation and Preprocess (AstroIMP)

AstroIMP is a tool for automated pre-processing astronomical images, and manipulating them since the quality of the images is usually low compared to other traditional images. This is due to the technical and physical characteristics related to the acquisition process. 

The tool applies some techniques for image manipulation and pre-processing. Some of them are contrast enhancement, crop images, reducing noise, removing CCD errors, removing the background, and staying only with the objects of interest to detect objects automatically with some algorithm for image segmentation. 

With this tool, we analyzed images from the INT Photometric H-Alpha Survey (IPHAS), an astronomical study from the northern plane of our galaxy. They obtained the images with the INT Wide Field Camera (WFC) at the Isaac Newton Telescope. The WFC has an array of four charge-coupled devices (CCD), each of which has 2K x 4K pixels.

AstroIMP it is divided into two modules: The first is image manipulation, and the second one is image pre-processing.  

The module image manipulation has two functions image contrast enhancement and crop images. Image contrast aims to manipulate the data by increasing the dynamic range of intensities in low-contrast images. The second functionality crop images aims to create multiple sub-images from the original image. These functionalities are described in the subsection software functionalities. 

The module pre-processing has three functions, noise reduction, CCD error reduction, and background removal. For noise reduction, we applied the filter anisotropic diffusion to enhance the edges of the point sources and the extended objects. The second functionality is CCD error reduction, this process is only applied to images that were obtained from CCD number 3 of IPHAS since it is the one that presents damage to the CCD columns. We applied a mathematical morphological close operator to reduce CCD errors. Once this process is completed, the image is inverted and returned to the original dynamic range. For the third functionality background removal, we apply first the anisotropic diffusion filter and the PFCM algorithm, the detailed process is described in the documentation.

For now the is no user interface, it has to run with command line.

Usage:

- To specify the input directory of the images: -d or --dir_images
- To specify the output directory of the images: -r or --dir_results
- Specifies if preprocess the images: -p or --preprocess
- To crop images: -c or --crop
- Specifies if applies contrast enhancement, using the algorithm zscale: -zs or --zscale
- Specifies if applies contrast enhancement, using the algorithm percentile range: -pr or --percentile_range
- Specifies if applies contrast enhancement, using the algorithm arcsin percentile: -ap or --arcsin_percentile
- Specifies if applies contrast enhancement, using the algorithm arcsin percentile range: -apr or --arcsin_percentile_range
- Specifies if remove the background using the algorithm PFCM: -pf or --pfcm

Documentation with examples is available in english and spanish versions. 


