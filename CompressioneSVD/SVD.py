#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: marcocianciotta
"""

# USAGE
# python compare.py

# import the necessary packages
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from scipy.interpolate import interp1d
from PIL import Image
from scipy import linalg
from skimage import io, img_as_float, img_as_uint

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def compare_images(imageA, imageB, title, number):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB) 
    
    
    s = ssim(imageA, imageB)

    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    
   
    value=[m,s]
    return value
    
def load( original, maxCompression, compressioneAttuale):

    contrast = cv2.imread(maxCompression)
    shopped = cv2.imread(compressioneAttuale)
    
    # convert the images to grayscale
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
    shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)
    
    foto=[original, contrast, shopped]
    return foto
    
def loadFoto(originale, maxCompression, compressioneAttuale):

    # load the images -- the original, the original + contrast,
    # and the original + photoshop
    original = cv2.imread(originale)
    contrast = cv2.imread(maxCompression)
    shopped = cv2.imread(compressioneAttuale)
    
    # convert the images to grayscale
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
    shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)
    
    # initialize the figure
    fig = plt.figure("Images")
    images = ("Original", original), ("Max Compression", contrast) ,("Compressed", shopped)
    
    # loop over the images
    for (i, (name, image)) in enumerate(images):
        # show the image
        ax = fig.add_subplot(1, 3, i + 1)
        ax.set_title(name)
        plt.imshow(image, cmap = plt.cm.gray)
        plt.axis("off")
    
    # show the figure
    plt.show()
    foto=[original, contrast, shopped]
    return foto

def establishMaxMSE(original,contrast, number):
    # compare the images
   
    maxValue = mse(original, contrast)
    print(maxValue)
    print("\nComparazione immagine \n Original vs MaxCompression")
    compare_images(original, contrast, "Original vs. Max Compression", number)
    
def svd_compress_gs(img, k):
	"""Given a matrix representing a grayscale image, compress 
	it by taking the largest k elements from its singular values"""
	U, singular_vals, V = linalg.svd(img)
	rank = len(singular_vals)
	#print ("Image rank %r" % rank)
	if k > rank:
		print ("k is larger than rank of image %r" % rank)
		return img
	# take columns less than k from U
	U_p = U[:,:k]
	# take rows less than k from V
	V_p = V[:k,:]
	# build the new S matrix with top k diagnal elements
	S_p = np.zeros((k, k), img.dtype)
	for i in range(k):
		S_p[i][i] = singular_vals[i]
	#print ("U_p shape {0}, S_p shape {1}, V_p shape {2}".format(U_p.shape, S_p.shape, V_p.shape))
	compressed = np.dot(np.dot(U_p, S_p), V_p)
	ss = ssim(img, compressed,
		dynamic_range=compressed.max()-compressed.min())
	#print ("Strucural similarity: %r" % ss)
    
	return compressed

def compress_images_k(originale, k_values):
    """Compress images with different k values. Compressed format is png."""
    imm = cv2.imread(originale)
    
    img = img_as_float(imm)
    number=0
    i=0
    contrast=None
    compressed = None
    m=[]
    s=[]
    klist=[]
    for k in range(2, k_values, 2):
        compressed = svd_compress_rgb(img, k, k, k)
        if(i==0):
            #io.imsave(os.path.join("../", "maxCompr" + ".jpg"), compressed)
            
            io.imsave('maxCompr.jpg', compressed)
            contrast=compressed
            
            
            
            i=i+1
        io.imsave('compressed.jpg', compressed)
        
    
        foto=load(imm, "maxCompr.jpg", "compressed.jpg" )
        original=foto[0]
        maxCompr=foto[1]
        comprAttuale=foto[2]
        
        #establishMaxMSE(original,maxCompr,number)
        value=compare_images(original, comprAttuale , "Original vs. Compressed", number)
        #print("value of K is: " + str(k))
        m.append(value[0])
        s.append(value[1])
        klist.append(k)
        
        #print("value of MSE is: " + str(m))
        #print("value of SSIM is: " + str(s))
    #imm2 = cv2.imread("compressed.jpg")
   
    #plt.imshow(imm2)
    #plt.show()
        
        
    return m,s,klist   
    
            
            
def svd_compress_rgb(img, k_r, k_g, k_b):
    """Given a matrix representing a RGB image, compress 
    it by taking the largest k elements from its singular values"""
    # split into separate channels
    comp_r = svd_compress_gs(img[:,:,0], k_r)
    comp_g = svd_compress_gs(img[:,:,1], k_g)
    comp_b = svd_compress_gs(img[:,:,2], k_b)
    new_img = np.zeros(img.shape, img.dtype)
    nrows = img.shape[0]
    ncols = img.shape[1]
    nchans = img.shape[2]
    for i in range(nrows):
        for j in range(ncols):
            for c in range(nchans):
                val = 0
                if c == 0:
                    val = comp_r[i][j]
                elif c == 1:
                    val = comp_g[i][j]
                else:
                    val = comp_b[i][j]
                # float64 values must be between -1.0 and 1.0
                if val < -1.0:
                    val = -1.0
                elif val > 1.0:
                    val = 1.0
                new_img[i][j][c] = val
    return new_img


def compressionOptimal(nomeFoto):
    
    originale=nomeFoto
    m,s,k= compress_images_k(originale, 200)
    mcount=len(m)
   
    kesimo=k[mcount-1]
    print("\nKesimo" + str(kesimo))
    
    i=0
    sFoto = float(os.path.getsize(nomeFoto))/1000
    mNormalized=[]
   
    #print(mcount)
    for i in range(0, mcount):
        mNormalized.append(m[i]/m[0])
    
    print("Array M "+ str(mNormalized) + "\nArray S "+ str(s) + "\nArray k " + str(k))
    
   
    position=0
    distance=1
    for i in range(0,mcount):
        if(s[i]>0.73):
            distanceAct= abs(mNormalized[i]-s[i])
            if distanceAct<distance:
                distance=distanceAct
                position=i
    position3=0
    distance3=1
    for i3 in range(0,mcount):
        if(s[i3]>0.90):
            distanceAct3= abs(mNormalized[i3]-s[i3])
            if distanceAct3<distance3:
                distance3=distanceAct3
                position3=i3
    position2=0
    distance2=1
    for i2 in range(0,mcount):
        distanceAct2= abs(mNormalized[i2]-s[i2])
        if distanceAct2<distance2:
            distance2=distanceAct2
            position2=i2
            
    
    #print("\nPosition " + str(position ))
    
    print("Intersezione punti")
    plt.plot(s, k, 'r', label='SSIM')
   
    plt.plot(mNormalized,k, 'b', label='MSE')
    
    plt.plot((s[position]+mNormalized[position])/2, k[position] , 'o', label='Lowest dist.\nSSIM and MSE and\nSSI >0.7')
    plt.plot((s[position2]+mNormalized[position2])/2, k[position2] , 'o', label='Lowest distance\nbet. SSIM and MSE', color='green')

    plt.legend()
    
    plt.axis([0, 1, 0, kesimo])
    plt.autoscale()
    plt.ylabel("k Value")
        
  
    plt.show()
    
    imm = cv2.imread(originale)
    cv2.imwrite
    
    img = img_as_float(imm)
    print("--------------------------------------------------------" )
    print("Image with lowest distance between MSE and SSIM" )
    print( "\n - S value: ")
    print(s[position2])
    print( "\n - M value: ")
    print(mNormalized[position2])
    print("\n - K is: ")
    print(k[position2])
    
    compressed2 = svd_compress_rgb(img, k[position2], k[position2], k[position2])
    io.imsave('compressedExactPoint.jpg', compressed2)
    imm3 = cv2.imread("compressedExactPoint.jpg")
    plt.imshow(imm3)
    s1 = float(os.path.getsize('compressedExactPoint.jpg'))/1000 
    print('\nweight in Kb : ', s1)
    print('\nOriginal foto  weight in Kb : ', sFoto)
    print('\nweight difference: ', (sFoto-s1))
    
    plt.show()
    
    print("--------------------------------------------------------" )
    print("Image with lowest distance between MSE and SSIM and SSIM >0.73" )
    print( "\n - S value: ")
    print(s[position])
    print( "\n - M value: ")
    print(mNormalized[position])
    print("\n - K is: ")
    print(k[position])
    
    compressed = svd_compress_rgb(img, k[position], k[position], k[position])
    io.imsave('compressedKoptimal.jpg', compressed)
    imm2 = cv2.imread("compressedKoptimal.jpg")
    plt.imshow(imm2)
    sCom = float(os.path.getsize('compressedKoptimal.jpg'))/1000 
    print('\nCompressed SSIM>0,73 weight in Kb : ', sCom)
    print('\nOriginal foto  weight in Kb : ', sFoto)
    print('\nweight difference: ', (sFoto-sCom))
    
    
    plt.show()
    
    print("--------------------------------------------------------" )
    print("Image with lowest distance between MSE and SSIM and SSIM >0.90" )
    print( "\n - S value: ")
    print(s[position3])
    print( "\n - M value: ")
    print(mNormalized[position3])
    print("\n - K is: ")
    print(k[position3])
    
    compressed3 = svd_compress_rgb(img, k[position3], k[position3], k[position3])
    io.imsave('compressedKoptimal90.jpg', compressed3)
    imm3 = cv2.imread("compressedKoptimal.jpg")
    plt.imshow(imm3)
    sCom3 = float(os.path.getsize('compressedKoptimal90.jpg'))/1000 
    print('\nCompressed SSIM>0,73 weight in Kb : ', sCom3)
    print('\nOriginal foto  weight in Kb : ', sFoto)
    print('\nweight difference: ', (sFoto-sCom3))
    
    
    plt.show()
    
    

print("---------- MENU' ----------")
scelta= input('1)Eseguire SVD con SSIM e MSE ottimali e SSIM > 0.73 \n2)Eseguire SVD inserendo un limite specifico di peso di Kb \ninserire scelta -> ')
#print(scelta)
scelta= int(scelta)
nomeFoto= input('Inserire il nome della foto es: foto.jpg \n(questa deve essere posizionata nella stessa cartella) \ninserire nome -> ')
#print(nomeFoto)
print("----------  ---  ----------")
nomeFoto=str(nomeFoto)
if scelta is 1:
    print("\nAttendere...L'esecuzione può richiedere qualche minuto")
    compressionOptimal(nomeFoto)
elif scelta is 2:
    kbinput= input('Inserire il limite dei Kb \nEs: 80 \ninserire kb -> ')
    kbinput=int(kbinput)
    print("\nAttendere...L'esecuzione può richiedere qualche minuto")
    conti=True
    k=2
    imm = cv2.imread(nomeFoto)
    cv2.imwrite
    img = img_as_float(imm)
    while(conti):
        compressed = svd_compress_rgb(img, k, k, k)
        io.imsave('compresKb.jpg', compressed)
        kb = float(os.path.getsize('compresKb.jpg'))/1000 
        imm2=cv2.imread('compressKb.jpg')
        
        
        if(kb<kbinput):
            k=k+2
            
        else:
            conti=False
        
    print("\nImmagine Compressa")
    print("\nKb image: ", str(kb))
    print("Value of K: ", str(k))
    
    immCompressedKb = cv2.imread("compresKb.jpg")
    plt.imshow(immCompressedKb)
    