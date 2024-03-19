# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:48:09 2019

@author: eduardo
"""
import os
import argparse
from imagen import Imagen
from preprocesamiento import preprocesamiento
parser = argparse.ArgumentParser(description='Segmentación de objetos extendidos.')
parser.add_argument("-c", "--recortar", action="store_true", help="Especifica si está en modo para recortar imágenes")
parser.add_argument("-p", "--preprocesamiento", action="store_true", help="Especifica si está en modo de preprocesamiento de imágenes")
parser.add_argument("-ef", "--elimina_fondo", action="store_true", help="Especifica si se elimina el fondo")
parser.add_argument("-zs", "--zscale", action="store_true", help="Especifica si se hará un ajuste de contraste, utilizando el algoritmo de zscale")
parser.add_argument("-pr", "--percentile_range", action="store_true", help="Especifica si se hará un ajuste de contraste, utilizando el algoritmo de percentile range")
parser.add_argument("-ap", "--arcsin_percentile", action="store_true", help="Especifica si se hará un ajuste de contraste, utilizando el algoritmo de arcsin percentile")
parser.add_argument("-apr", "--arcsin_percentile_range", action="store_true", help="Especifica si se hará un ajuste de contraste, utilizando el algoritmo de arcsin percentile range")
parser.add_argument("-pf", "--pfcm", action="store_true", help="Especifica si desea eliminar el fondo con el algoritmo PFCM")
parser.add_argument("-d" , "--dir_imagenes", action="store", dest="dir_imagenes", help="directorio de entrada")
parser.add_argument("-r", "--dir_resultado", action="store", dest="dir_resultado", help="directorio de salida")
parser.add_argument("-t", "--entrenar", action="store_true", help="Especifica si está en modo para entrenar el modelo")
parser.add_argument("-re", "--reanudar", action="store_true", help="Especifica si está en modo de reanudar el entrenamiento del modelo")
parser.add_argument("-k", "--kfold", action="store", dest="kf", help="Especifica un numero entero para el número de k fold en el que se quedó el entrenamiento.")
parser.add_argument("-s", "--segmentar", action="store_true", help="Especifica si está en modo para segmentar las imágenes de prueba, tomando como base el modelo previamente creado")
parser.add_argument("-o", "--extendidos", action="store_true", help="Especifica si está utilizando el programa para segmentación de objetos extendidos, debe utilizarse junto con -t o -s")
args = parser.parse_args()
#parser.print_help()
if args.preprocesamiento:
    print("Preprocesamiento...")
    imagenes = os.listdir(args.dir_imagenes)
    tipo = ""
    for img in imagenes:
        print(img)
        pp = preprocesamiento(args.dir_imagenes + "/" + img, True)
        pp.guardar_imagen_tif(os.getcwd() + "/" + args.dir_resultado, img + "_original.tif", True)
        if args.elimina_fondo:
            print("Elimina fondo...")
            pp.elimina_fondo()
            pp.guardar_imagen_tif(os.getcwd() + "/" + args.dir_resultado, img + "_sin_fondo.tif", True)
        if args.zscale:
            print("Ajuste de contraste con zscale...")
            pp.autocontraste(1, not args.elimina_fondo)
            tipo = "zscale"
        elif args.percentile_range:
            print("Ajuste de contraste con percentile range...")
            tipo = "percentile_range"
            pp.autocontraste(2, not args.elimina_fondo)
        elif args.arcsin_percentile:
            print("Ajuste de contraste con arcsin percentile...")
            tipo = "arcsin_percentile"
            pp.autocontraste(3, not args.elimina_fondo)
        elif args.arcsin_percentile_range:
            print("Ajuste de contraste con arcsin percentile range...")
            tipo = "arcsin_percentile_range"
            pp.autocontraste(4, not args.elimina_fondo)
        pp.guardar_imagen_tif(os.getcwd() + "/" + args.dir_resultado, img + "_" + tipo + ".tif", True)
        if args.pfcm:
            pp.pfcm_2(args.dir_resultado, img)
            pp.guardar_imagen_tif(os.getcwd() + "/" + args.dir_resultado, img + "_pfcm.tif", True)
elif args.recortar:
    print("Recortar imagenes...")
    iobj = Imagen()
    iobj.recortar_imagenes(args.dir_imagenes, args.dir_resultado, 512)