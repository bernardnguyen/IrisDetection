#!/usr/bin/env python
#from segmentation import *
import time
import math
# import pycuda.autoinit
# from pycuda.compiler import SourceModule
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure
# from pycuda import driver, compiler, gpuarray, tools
print ("Eye detection")

#Device Source
# mod = SourceModule("""
# #include <math.h>
# #define PI 3.14159265

	# __global__ void grayscale(float* data_r, float* data_g, float* data_b, float* grayscaled, int numCols){
		# int tx = threadIdx.x;
		# int ty = threadIdx.y;
		# int row = blockIdx.y*blockDim.y + ty;
		# int col = blockIdx.x*blockDim.x + tx;
		
		# int index = row*numCols + col;
		
		# grayscaled[index] = 0.2989*data_r[index] + 0.5870*data_g[index] + 0.1140*data_b[index];
		
	# }
	
	# __global__ void convolution_2D_tiled(float* data, float* mask, float* z, int M, int N, int dilation, int mask_size){
	
	# //THIS CODE IS FOR AN UNPADDED INPUT, JUST USE PADDED INPUT AKA CONVERT MY PYTHON CODE INTO KERNEL SHIT
		# int tx = threadIdx.x;
		# int ty = threadIdx.y;
		# int row = blockIdx.y*blockDim.y + ty;
		# int col = blockIdx.x*blockDim.x + tx;
		
		# //this works for smaller matrices (input matrix < 20000 elements)
		# int output = 0;
		# if ((row >= dilation) && (row < dilation + M) && (col >= dilation) && (col < dilation + N)){
			# for (int j = 0; j < mask_size; j++){
				# for (int k = 0; k < mask_size; k++){
					# output += mask[j*mask_size +k] * data[(row-dilation + dilation*j)*(N+dilation*(mask_size - 1)) + (col-dilation + dilation*k)];
				# }
			# }
			# z[(row - dilation)*N + (col - dilation)] = output;
		# }
	# }

	# __global__ void gradient(float* data_x, float* data_y, float* gradient, int numCols){
		# int tx = threadIdx.x;
		# int ty = threadIdx.y;
		# int row = blockIdx.y*blockDim.y + ty;
		# int col = blockIdx.x*blockDim.x + tx;
		
		# int index = row*numCols + col;
		
		# gradient[index] = sqrt(data_x[index]*data_x[index] + data_y[index]*data_y[index]);
		
	# }
	
	# __global__ void orientation(float* data_x, float* data_y, float* orientation, int numCols){
		# int tx = threadIdx.x;
		# int ty = threadIdx.y;
		# int row = blockIdx.y*blockDim.y + ty;
		# int col = blockIdx.x*blockDim.x + tx;
		
		# int index = row*numCols + col;
		
		# float value = atan2(-data_y[index],data_x[index]);
		# if (value < 0){
			# value = value + PI;
		# }
		# orientation[index] =  value * 180 / PI;	
	# }
	
	# __global__ void adjgamma(float* gradient_in, float* gradient_out, float min, float max, int numCols){
		# int tx = threadIdx.x;
		# int ty = threadIdx.y;
		# int row = blockIdx.y*blockDim.y + ty;
		# int col = blockIdx.x*blockDim.x + tx;
		
		# int index = row*numCols + col;
		# float gamma = 1/1.9;
		
		# float value = (gradient_in[index] - min) / max;
		# gradient_out[index] = pow(value, gamma);

	# }
	
	# __global__ void nonmaxsuppression(float* im, float* inimage, int* orient, float* xoff, float* yoff, float* hfrac, float* vfrac, int iradius, int numCols, int numRows){
		# int tx = threadIdx.x;
		# int ty = threadIdx.y;
		# int row = blockIdx.y*blockDim.y + ty + iradius;
		# int col = blockIdx.x*blockDim.x + tx + iradius;
		
		# if ((row < numCols) && (col < numCols)){
			# int index = row*numCols + col;
			# int ori = orient[index];
			
			# double x = col + xoff[ori];
			# double y = row -yoff[ori];
			
			# int fx = (int) x;
			# int cx = (int)(x+0.5);
			# int fy = (int) y;
			# int cy = (int)(y+0.5);
			# float tl = inimage[fy*numCols + fx];
			# float tr = inimage[fy*numCols + cx];
			# float bl = inimage[cy*numCols + fx];
			# float br = inimage[cy*numCols + cx];
			
			# float upperavg = tl + hfrac[ori]*(tr-tl);
			# float loweravg = bl + hfrac[ori]*(br-bl);
			# float v1 = upperavg + vfrac[ori]*(loweravg - upperavg);
			
			# if (inimage[index] > v1){
				# x = col - xoff[ori];
				# y = row + yoff[ori];
				# fx = (int)(x);
				# cx = (int)(x+0.5);
				# fy = (int)(y);
				# cy = (int)(y+0.5);
				# tl = inimage[fy*numCols + fx];
				# tr = inimage[fy*numCols + cx];
				# bl = inimage[cy*numCols + fx];
				# br = inimage[cy*numCols + cx];
			
				# upperavg = tl + hfrac[ori]*(tr-tl);
				# loweravg = bl + hfrac[ori]*(br-bl);
				# float v2 = upperavg + vfrac[ori]*(loweravg - upperavg);
				
				# if (inimage[index] > v2){
					# im[index] = inimage[index];
				# }
				
			# }
			
		# }
	# }		
		
	
	
# """)
			
# #function calls (ALL FOR CANNY.M)
# grayscaled = mod.get_function("grayscale") 
# conv = mod.get_function("convolution_2D_tiled")
# grad = mod.get_function("gradient")
# orient = mod.get_function("orientation")
# adjg = mod.get_function("adjgamma")
# nonmaxsuppression = mod.get_function("nonmaxsuppression")

#converting image to grayscale
img = Image.open("eye.jpg")
img.load()
data_raw = np.asarray(img, dtype="float32")
numCols = data_raw.shape[1]
numRows = data_raw.shape[0]
r, g, b = data_raw[:,:,0], data_raw[:,:,1], data_raw[:,:,2]
print(data_raw.shape)
times = []
for w in range(1,4):
	start = time.time()
	gray = 0.2989*r + 0.5870*g + 0.1140*b
	times.append(time.time() - start)
print ('python grayscaling time: ', np.average(times))
gray = np.floor(gray)

M = gray.shape[0]
N = gray.shape[1]

# r_gpu = gpuarray.to_gpu(r)
# g_gpu = gpuarray.to_gpu(g)
# b_gpu = gpuarray.to_gpu(b)
# grayscaled_gpu = gpuarray.empty(gray.shape, np.float32)

# for w in range(1,4):
	# start = time.time()
	# grayscaled(r_gpu, g_gpu, b_gpu, grayscaled_gpu, np.int32(numCols), block=(8, 8, 1), grid = (N/8 + 1, M/8 + 1, 1))
	# times.append(time.time() - start)
# print 'pyCUDA grayscaling time: ', np.average(times)

#filtering with gaussian
dilation = 1
mask_size = 13
sigma = 6
padding = ((mask_size - 1)/2)*dilation
ind = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
xmask = []
for i in range(13):
	xmask.append(ind)
xmask_np = np.array(xmask, dtype = 'float32')
ymask_np = np.transpose(xmask_np)
h = np.exp(-(xmask_np**2 + ymask_np**2)/(2*sigma*sigma))
h = h/np.sum(h)
#h is the gaussian filter, with sigma = 2, it is also symmetric

#data = np.pad(grayscaled_gpu.get().astype(np.float32), (int(padding),int(padding)), 'constant', constant_values = 0)
# data_gpu = gpuarray.to_gpu(data)
mask = h.astype(np.float32)
# mask_gpu = gpuarray.to_gpu(mask)
# z_gpu = gpuarray.empty(gray.shape, np.float32)

#python 2d conv
times = []
for q in range(1,4):
	start = time.time()
	dst = cv2.filter2D(gray,-1,mask)
	times.append(time.time() - start)
print ('python filter time: ', np.average(times))

#CUDA 2d conv
# times = []
# for q in range(1,4):
	# start = time.time()
	# conv(data_gpu, mask_gpu, z_gpu, np.int32(M), np.int32(N), np.int32(dilation), np.int32(mask_size), block=(8, 8, 1), grid = (N/8 + 1, M/8 + 1, 1))
	# times.append(time.time() - start)
# print 'pyCUDA filter time: ', np.average(times)

# calculating gradient 
h1 = np.c_[dst[:,1:numCols], np.zeros(numRows)]
h2 = np.c_[np.zeros(numRows), dst[:,0:numCols-1]]
h = h1 - h2

v1 = np.r_[dst[1:numRows,:], [np.zeros(numCols)]]
v2 = np.r_[[np.zeros(numCols)], dst[0:numRows-1, :]]
v = v1 - v2

d1a = np.c_[dst[1:numRows,1:numCols], np.zeros(numRows-1)] #add col
d1b = np.pad(dst[1:numRows, 1:numCols], (1,1), 'constant', constant_values = 0)
d1b = d1b[1:numRows+1, 1:numCols+1]
d1c = np.pad(dst[0:numRows-1, 0:numCols-1], (1,1), 'constant', constant_values = 0)
d1c = d1c[0:numRows, 0:numCols]
d1 = d1b - d1c

d2a = np.pad(dst[0:numRows-1, 1:numCols], (1,1), 'constant', constant_values = 0)
d2a = d2a[0:numRows, 1:numCols+1]
d2b = np.pad(dst[1:numRows, 0:numCols-1], (1,1), 'constant', constant_values = 0)
d2b = d2b[1:numRows+1, 0:numCols]
d2 = d2a-d2b

X = (h + (d1+d2)/2.0)
Y = (v + (d1-d2)/2.0)

times = []
for g in range(1,4):
	start = time.time()
	gradient = np.sqrt(np.multiply(X,X) + np.multiply(Y,Y))
	times.append(time.time() - start)
print ('python gradient time: ', np.average(times))

# X_gpu = gpuarray.to_gpu(X.astype(np.float32))
# Y_gpu = gpuarray.to_gpu(Y.astype(np.float32))
# gradient_gpu = gpuarray.empty(X_gpu.shape, np.float32)

# times = []
# for g in range(1,4):
	# start = time.time()
	# grad(X_gpu, Y_gpu, gradient_gpu, np.int32(numCols), block=(8, 8, 1), grid = (N/8 + 1, M/8 + 1, 1))
	# times.append(time.time() - start)
# print 'pyCUDA gradient time: ', np.average(times)

# find orientation
for g in range(1,4):
	start = time.time()
	orientation = np.arctan2(-Y,X) #confirmed same as matlab
	for i in range(orientation.shape[0]):
		for j in range(orientation.shape[1]):
			if orientation[i,j] < 0:
				orientation[i,j]  = orientation[i,j] + math.pi

	orientation = orientation * 180 / math.pi
	times.append(time.time() - start)
print ('python orientation time: ', np.average(times))

# X_gpu = gpuarray.to_gpu(X.astype(np.float32))
# Y_gpu = gpuarray.to_gpu(Y.astype(np.float32))
# orientation_gpu = gpuarray.empty(X_gpu.shape, np.float32)

# times = []
# for g in range(1,4):
	# start = time.time()
	# orient(X_gpu, Y_gpu, orientation_gpu, np.int32(numCols), block = (8,8,1), grid = (N/8 + 1, M/8 + 1, 1))
	# times.append(time.time() - start)
# print 'pyCUDA orientation time: ', np.average(times)

#i3 = adjgamma(i2, 1.9)
#find minimum, maximum, element-wise scaling
newim = gradient
times = []
for g in range(1,4):
	start = time.time()
	newim = newim - np.min(np.min(newim))
	newim = newim / np.max(np.max(newim))
	newim = newim ** (1/1.9)
	adjgamma = newim
	times.append(time.time() - start)
print ('python adjgamma time: ', np.average(times))

# times = []
# for g in range(1,4):
	# start = time.time()
	# min = pycuda.gpuarray.min(gradient_gpu)
	# max = pycuda.gpuarray.max(gradient_gpu)
	# adjgamma_gpu = gpuarray.empty(X_gpu.shape, np.float32)
	# adjg(gradient_gpu, adjgamma_gpu, np.float32(min.get()), np.float32(max.get()), np.int32(numCols), block = (8,8,1), grid = (N/8 + 1, M/8 + 1, 1))
	# times.append(time.time() - start)
# print 'pyCUDA adjgamma time: ', np.average(times)

#i4 = nonmaxsup(i3, or, 1.5)
inimage = adjgamma
#inimage = adjgamma_gpu.get()
orient = orientation
radius = 1.5
rows,cols = inimage.shape
im = np.zeros((rows,cols))        #Preallocate memory for output image for speed
iradius = np.ceil(radius).astype(np.int32)

#Precalculate x and y offsets relative to centre pixel for each orientation angle 
angle = np.array(range(181))*np.pi/180    # Array of angles in 1 degree increments (but in radians).
xoff = radius*np.cos(angle)   # x and y offset of points at specified radius and angle
yoff = radius*np.sin(angle)   # from each reference position.
hfrac = xoff - np.floor(xoff) # Fractional offset of xoff relative to integer location
vfrac = yoff - np.floor(yoff) # Fractional offset of yoff relative to integer location

orient = np.fix(orient).astype(np.int32)


times = []
for w in range(1,2):
	start = time.time()
	for row in range (iradius,(rows - iradius)):
		for col in range (iradius,(cols - iradius)):
			ori = orient[row,col]   # Index into precomputed arrays

			x = col + xoff[ori]     # x, y location on one side of the point in question
			y = row - yoff[ori]

			fx = np.floor(x).astype(np.int32)          # Get integer pixel locations that surround location x,y
			cx = np.ceil(x).astype(np.int32)
			fy = np.floor(y).astype(np.int32)
			cy = np.ceil(y).astype(np.int32)
			tl = inimage[fy,fx]   # Value at top left integer pixel location.
			tr = inimage[fy,cx]    # top right
			bl = inimage[cy,fx]   # bottom left
			br = inimage[cy,cx]   # bottom right

			upperavg = tl + hfrac[ori] * (tr - tl)  # Now use bilinear interpolation to
			loweravg = bl + hfrac[ori] * (br - bl)  # estimate value at x,y
			v1 = upperavg + vfrac[ori] * (loweravg - upperavg)

			if inimage[row, col] > v1 : # We need to check the value on the other side...
				x = col - xoff[ori]     # x, y location on the `other side' of the point in question
				y = row + yoff[ori]

				fx = np.floor(x).astype(np.int32)
				cx = np.ceil(x).astype(np.int32)
				fy = np.floor(y).astype(np.int32)
				cy = np.ceil(y).astype(np.int32)
				tl = inimage[fy,fx]   # Value at top left integer pixel location.
				tr = inimage[fy,cx]    # top right
				bl = inimage[cy,fx]    # bottom left
				br = inimage[cy,cx]    # bottom right
				upperavg = tl + hfrac[ori] * (tr - tl)
				loweravg = bl + hfrac[ori] * (br - bl)
				v2 = upperavg + vfrac[ori] * (loweravg - upperavg)

				if inimage[row,col] > v2:            # This is a local maximum.
					im[row, col] = inimage[row, col] # Record value in the output image.
	times.append(time.time() - start)
print ('python nonmax suppression time: ', np.average(times))

# im_gpu = gpuarray.to_gpu(im.astype(np.float32))	
# inimage_gpu = gpuarray.to_gpu(inimage.astype(np.float32))
# orient_gpu = gpuarray.to_gpu(orient.astype(np.int32))
# xoff_gpu = gpuarray.to_gpu(xoff.astype(np.float32))
# yoff_gpu = gpuarray.to_gpu(yoff.astype(np.float32))
# hfrac_gpu = gpuarray.to_gpu(hfrac.astype(np.float32))
# vfrac_gpu = gpuarray.to_gpu(vfrac.astype(np.float32))

# times = []
# for w in range(1,4):
	# start = time.time()
	# nonmaxsuppression(im_gpu, inimage_gpu, orient_gpu, xoff_gpu, yoff_gpu, hfrac_gpu, vfrac_gpu, np.int32(iradius), np.int32(numCols), np.int32(numRows), block = (8,8,1), grid = (N/8 + 1, M/8 + 1, 1))
	# times.append(time.time() - start)
# print 'pyCUDA nonmax suppression time: ', np.average(times)
	
#hyst 
T1 = 0.20
T2 = 0.19
#im = im_gpu.get()
rows, cols = im.shape    # Precompute some values for speed and convenience.
rc = rows*cols
rcmr = rc - rows
rp1 = rows+1
imx,imy = im.shape
bw = np.reshape(im,(imx*imy))                # Make image into a column vector
pix = np.nonzero(bw > T1)       # Find indices of all pixels with value > T1
pix = pix[0] #tuple to array

npix = pix.shape[0]
stack = np.zeros(rows*cols) # Create a stack array (that should never overflow

stack = pix        # Put all the edge points on the stack
stp = npix                 # set stack pointer
for k in range (npix):
    bw[pix[k]] = -1        # mark points as edges


# Precompute an array, O, of index offset values that correspond to the eight 
# surrounding pixels of any point. Note that the image was transformed into
# a column vector, so if we reshape the image back to a square the indices 
# surrounding a pixel with index, n, will be:
#              n-rows-1   n-1   n+rows-1
#
#               n-rows     n     n+rows
#                     
#              n-rows+1   n+1   n+rows+1

O = [-1, 1, -rows-1, -rows, -rows+1, rows-1, rows, rows+1]

while stp != -1 :           # While the stack is not empty
    v = stack[stp-1]         # Pop next index off the stack
    stp = stp - 1
    
    if v > rp1 and v < rcmr:   # Prevent us from generating illegal indices
    # Now look at surrounding pixels to see if they
                            # should be pushed onto the stack to be
                            # processed as well.
       index = O+v     # Calculate indices of points around this pixel.     
       for l in range(8):
        ind = index[l]
        if bw[ind-1] > T2:   # if value > T2,
            stp = stp+1  # push index onto the stack.
            stack[stp-1] = ind
            bw[ind-1] = -1 # mark this as an edge point

bw = (bw == -1)            # Finally zero out anything that was not an edge 
bw = np.reshape(bw,(rows,cols)) # and reshape the image

# plt.figure(1)
# plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.figure(2)
# plt.imshow(grayscaled_gpu.get()),plt.title('grayscaling')
# plt.xticks([]), plt.yticks([])
# plt.figure(3)
# plt.imshow(z_gpu.get()), plt.title('filtered')
# plt.xticks([]), plt.yticks([])
# plt.figure(4)
# plt.imshow(gradient_gpu.get()), plt.title('gradient')
# plt.xticks([]), plt.yticks([])
# plt.figure(5)
# plt.imshow(adjgamma_gpu.get()), plt.title('adjgamma')
# plt.xticks([]), plt.yticks([])
# plt.figure(6)
# plt.imshow(im_gpu.get()), plt.title('non-max suppression')
# plt.xticks([]), plt.yticks([])
# plt.figure(7)
# plt.imshow(bw), plt.title('edge map')
# plt.xticks([]), plt.yticks([])
# plt.show()	

plt.figure(7)
plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.figure(6)
plt.imshow(gray),plt.title('grayscaling')
plt.xticks([]), plt.yticks([])
plt.figure(5)
plt.imshow(dst), plt.title('filtered')
plt.xticks([]), plt.yticks([])
plt.figure(4)
plt.imshow(gradient), plt.title('gradient')
plt.xticks([]), plt.yticks([])
plt.figure(3)
plt.imshow(adjgamma), plt.title('adjgamma')
plt.xticks([]), plt.yticks([])
plt.figure(2)
plt.imshow(im), plt.title('non-max suppression')
plt.xticks([]), plt.yticks([])
plt.figure(1)
plt.imshow(bw), plt.title('edge map')
plt.xticks([]), plt.yticks([])
plt.show()





	
		
















