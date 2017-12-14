from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter as g_f
from scipy.misc import imresize

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
	
def segmentiris(eyeimage):
	##imsize double 
    lpupilradius = 28
    upupilradius = 40
    lirisradius = 80
    uirisradius = 150

    #eyeimage = rgb2gray(np.array(Image.open(eyeimage)))
    # define scaling factor to speed up Hough transform
    scaling = 0.4

    reflecthres = 240

    # Output message
    print('    (1/2) Finding iris boundary...')
	
    #find the iris boundary
    row, col, r = findcircle(eyeimage, lirisradius, uirisradius, scaling, 6, 0.6, 0.59, 1.00, 1.00)
    
    circleiris = [row, col, r]
    irl = row-r
    iru = row+r
    icl = col-r
    icu = col+r

    #imgsize = size(eyeimage);
    imgsize = eyeimage.shape

    if irl < 1 :
    	irl = 1

    if icl < 1:
    	icl = 1

    if iru > imgsize[0]:
    	iru = imgsize[0]

    if icu > imgsize[1]:
    	icu = imgsize[1]

    # to find the inner pupil, use just the region within the previously
    # detected iris boundary
    imagepupil = eyeimage [irl:iru,icl:icu]

	# Output message
    print('    (2/2) Finding pupil boundary...')

    #find pupil boundary
    rowp, colp, r = findcircle(imagepupil, lpupilradius, upupilradius ,0.6,6,0.25,0.25,1.00,1.00)

    row = irl + rowp
    col = icl + colp

    circlepupil = [row,col,r]

    # set up array for recording noise regions
    #noise pixels will have NaN values
    imagewithnoise = eyeimage

    return circleiris,circlepupil,imagewithnoise

def findcircle(image,lradius,uradius,scaling, sigma, hithres, lowthres, vert, horz):
	#around where
	lradsc = np.around(lradius*scaling)
	uradsc = np.around(uradius*scaling)
	rd = np.around(uradius*scaling - lradius*scaling).astype(np.int32)

	#generate the edge imag e

	# Output message
	print('        (1/5) Canny edge detection...')

	I2, ori = canny(image, sigma, scaling, vert, horz)
	
	# Output message
	print('        (2/5) Gamma adjustment...')
	
	I3 = adjgamma(I2, 1.9)
	
	# Output message
	print('        (3/5) Non-max suppression...')
	
	I4 = nonmaxsup(I3, ori, 1.5)

	# Output message
	print('        (4/5) Hysteresis thresholding...')

	edgeimage = hysthresh(I4, hithres, lowthres)

	plt.figure(1)
	plt.imshow(image),plt.title('grayscaled')
	plt.xticks([]), plt.yticks([])	
	plt.figure(2)
	plt.imshow(I2),plt.title('gradient')
	plt.xticks([]), plt.yticks([])
	plt.figure(3)
	plt.imshow(I3), plt.title('gamma adjusted')
	plt.xticks([]), plt.yticks([])
	plt.figure(4)
	plt.imshow(I4), plt.title('non-max suppression')
	plt.xticks([]), plt.yticks([])
	plt.figure(5)
	plt.imshow(edgeimage), plt.title('edge map')
	plt.xticks([]), plt.yticks([])
	plt.show()

	# Output message
	print('        (5/5) Hough circle transform...')

	#perform the circular Hough transform
	h = houghcircle(edgeimage, lradsc, uradsc)

	maxtotal = 0

	# find the maximum in the Hough space, and hence
	#the parameters of the circle
	for i in range(1+rd):
		layer = h[:,:,i]
		maxlayer = np.max(np.max(layer))

		if maxlayer > maxtotal :
			maxtotal = maxlayer
			r = ((lradsc+i) / scaling).astype(np.int32) 
			row,col =  np.nonzero(layer == maxlayer)
			row = row[0]
			col = col[0] 
			# returns only first max value
			row = (row / scaling).astype(np.int32)
			col = (col / scaling).astype(np.int32)
	return row, col, r

def houghcircle(edgeim, rmin, rmax):
	#nonzero
	rows,cols = edgeim.shape
	nradii = (rmax-rmin+1).astype(np.int32)
	h = np.zeros((rows,cols,nradii))

	y,x = np.nonzero(edgeim!=0)

	#%for each edge point, draw circles of different radii
	for index in range(y.shape[0]):
		print('        Edge point: ({}/{})'.format(index,y.shape[0]), end='\r')
		cx = x[index]
		cy = y[index]
		for n in range(nradii):
			h[:,:,n] = addcircle(h[:,:,n],[cx,cy],n+rmin)
	return h

def canny(im, sigma, scaling, vert, horz):
	#arctan2
	xscaling = vert
	yscaling = horz

	im = g_f(im,sigma)       # Smoothed image.
	im = imresize(im, scaling)

	rows, cols = im.shape	

	# h =  np.pad(im[:,1:cols],((0,0),(0,1)),'constant') - np.pad(im[:,:cols-1],((0,0),(1,0)),'constant' )
	# v =  np.pad(im[1:rows,:], ((1,0),(0,0)),'constant') - np.pad(im[:rows-1,:],((0,1),(0,0)),'constant')
	# d1 = np.pad(im[1:rows,1:cols],((0,1),(0,1)),'constant' ) - np.pad( im[:rows-1,:cols-1], ((1,0),(1,0)),'constant')
	# d2 = np.pad( im[:rows-1,1:cols], ((1,0),(0,1)),'constant') - np.pad(im[1:rows,:cols-1],((0,1),(1,0)),'constant')
	
	h1 = np.c_[im[:,1:cols], np.zeros(rows)]
	h2 = np.c_[np.zeros(rows), im[:,0:cols-1]]
	h = h1 - h2
	
	v1 = np.r_[im[1:rows,:], [np.zeros(cols)]]
	v2 = np.r_[[np.zeros(cols)], im[0:rows-1, :]]
	v = v1 - v2

	d1a = np.c_[im[1:rows,1:cols], np.zeros(rows-1)] #add col
	d1b = np.pad(im[1:rows, 1:cols], (1,1), 'constant', constant_values = 0)
	d1b = d1b[1:rows+1, 1:cols+1]
	d1c = np.pad(im[0:rows-1, 0:cols-1], (1,1), 'constant', constant_values = 0)
	d1c = d1c[0:rows, 0:cols]
	d1 = d1b - d1c

	d2a = np.pad(im[0:rows-1, 1:cols], (1,1), 'constant', constant_values = 0)
	d2a = d2a[0:rows, 1:cols+1]
	d2b = np.pad(im[1:rows, 0:cols-1], (1,1), 'constant', constant_values = 0)
	d2b = d2b[1:rows+1, 0:cols]
	d2 = d2a-d2b

	X = ( h + (d1 + d2)/2.0 ) * xscaling
	Y = ( v + (d1 - d2)/2.0 ) * yscaling

	gradient = np.sqrt(X**2 + Y**2) # Gradient amplitude.

	ori = np.arctan2(-Y, X)            # Angles -pi to + pi.
	neg = ori<0;                   # Map angles to 0-pi.
	ori = ori*~neg + (ori+np.pi)*neg 
	ori = ori*180 / np.pi               # Convert to degrees
	return gradient, ori

def adjgamma(im,g):
	# error ,isa deleted
	#    im     - image to be processed.
	#    g      - image gamma value.
	#         Values in the range 0-1 enhance contrast of bright
	#         regions, values > 1 enhance contrast in dark
	#         regions.

	newim = im
	newim = newim-np.min(np.min(newim))
	newim = newim/np.max(np.max(newim))

	newim =  newim**(1/g)

	return newim

def nonmaxsup(inimage, orient, radius):
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

	# Now run through the image interpolating grey values on each side
	# of the centre pixel to be used for the non-maximal suppression.

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

	return im

def hysthresh(im, T1, T2):
	rows, cols = im.shape    # Precompute some values for speed and convenience.
	rc = rows*cols
	rcmr = rc - rows
	rp1 = rows+1
	imx,imy = im.shape
	bw = np.reshape(im,(imx*imy))                # Make image into a column vector
	pix = np.nonzero(bw > T1)       # Find indices of all pixels with value > T1
	pix = pix[0]	#tuple to array

	npix = pix.shape[0]
	#stack = np.zeros(rows*cols) # Create a stack array (that should never overflow

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

	while stp > 0 :           # While the stack is not empty
	    v = stack[stp-1]         # Pop next index off the stack
	    stp = stp - 1
	    
	    if v > rp1 and v < rcmr:   # Prevent us from generating illegal indices
	    # Now look at surrounding pixels to see if they
	                            # should be pushed onto the stack to be
	                            # processed as well.
	       index = O+v	    # Calculate indices of points around this pixel.	    
	       for l in range(8):
	       	ind = index[l]
	       	if bw[ind] > T2:   # if value > T2,
	       		stp = stp+1  # push index onto the stack.
	       		stack[stp-1] = ind
	       		bw[ind] = -1 # mark this as an edge point




	bw = (bw == -1)            # Finally zero out anything that was not an edge 
	bw = np.reshape(bw,(rows,cols)) # and reshape the image

	return bw

def addcircle(h, c, radius, weight=1):
	#find, and, append
	hr, hc = h.shape
	h = np.reshape(h,(hr*hc))
	x = range(np.fix(radius/np.sqrt(2)).astype(np.int32))
	x = np.array(x) # list to array

	costheta = np.sqrt(1 - (x**2 / radius**2))
	y = np.around(radius*costheta).astype(np.int32)
	# Now fill in the 8-way symmetric points on a circle given coords 
	# [px py] of a point on the circle.

	px = c[1] + np.append(x, [y,  y,  x, -x, -y, -y, -x])
	py = c[0] + np.append(y, [x, -x, -y, -y, -x,  x,  y])

	# Cull points that are outside limits
	validx1 = px>=1
	validx2 = px<=hr
	validy1 = py>=1 
	validy2 = py<=hc

	validxy = validx1*validx2*validy1*validy2

	valid = np.nonzero(validxy)

	px = px[valid]
	py = py[valid]

	ind = px +(py-1) *hr - 1
	h[ind] = h[ind] + weight
	h = np.reshape(h,(hr,hc))

	return h

def createiristemplate(eyeimage_filename):
	#imread? load save

	# path for writing diagnostic images
	
	#global DIAGPATH
	#DIAGPATH = 'diagnostics'

	#normalisation parameters
	radial_res = 20
	angular_res = 240
	# with these settings a 9600 bit iris template is
	# created

	#feature encoding parameters
	nscales=1
	minWaveLength=18
	mult=1 # not applicable if using nscales = 1
	sigmaOnf=0.5
	eyeimage = eyeimage_filename

	# Output message
	print('(1/4) Grayscaling image...')

	eyeimage = rgb2gray(np.array(Image.open(eyeimage)))

	# Output message
	print('(2/4) Segmenting iris...')

	circleiris,circlepupil,imagewithnoise = segmentiris(eyeimage)
	print('circleiris:',circleiris)
	print('circlepupil:',circlepupil)
	#np.save(savefile,'circleiris','circlepupil','imagewithnoise')
	# WRITE NOISE IMAGE
	imagewithnoise2 = imagewithnoise.astype(np.int32)
	imagewithcircles = eyeimage.astype(np.int32)
	
	# Output message
	print('(3/4) Finding circle coordinates for iris...')

	#get pixel coords for circle around iris
	x,y = circlecoords([circleiris[0],circleiris[1]],circleiris[2],eyeimage.shape)
	print('x:',x,'y:',y)
	print('shape:',eyeimage.shape)
	ind2 = x*eyeimage.shape[1]+y - 1

	# Output message
	print('(4/4) Finding circle coordinates for pupil...')

	#get pixel coords for circle around pupil
	xp,yp = circlecoords([circlepupil[0],circlepupil[1]],circlepupil[2],eyeimage.shape)
	ind1 = xp*eyeimage.shape[1]+yp - 1

	# Write noise regions
	inx,iny = imagewithnoise2.shape
	imagewithnoise2 = np.reshape(imagewithnoise2,(inx*iny))

	imagewithnoise2[ind2] = 0
	imagewithnoise2[ind1] = 0
	# Write circles overlayed
	inx,iny = imagewithcircles.shape
	imagewithcircles = np.reshape(imagewithcircles,(inx*iny))

	imagewithcircles[ind2] = 155
	imagewithcircles[ind1] = 155
	
	imagewithcircles = np.reshape(imagewithcircles,(inx,iny))

		# plot
	print('imagewithcircles')
	showresult(imagewithcircles)

	# perform feature encoding
	template = 0

	return template

def circlecoords(c, r, imgsize,nsides = 600):
	nsides = round(nsides)
	a = np.linspace(0,2*np.pi,2*nsides)
	xd = r*np.cos(a)+ c[0]
	yd = r*np.sin(a)+ c[1] 

	xd = np.around(xd).astype(np.int32)
	yd = np.around(yd).astype(np.int32)
	#get rid of -ves
	#get rid of values larger than image
	xd2 = xd
	coords = np.nonzero(xd>imgsize[0])
	coords = coords[0]
	xd2[coords] = imgsize[0]
	coords = np.nonzero(xd<=0)
	coords = coords[0]
	xd2[coords] = 1

	yd2 = yd
	coords = np.nonzero(yd>imgsize[1])
	coords = coords[0]
	yd2[coords] = imgsize[1]
	coords = np.nonzero(yd<=0)
	coords = coords[0]
	yd2[coords] = 1
	x = xd2.astype(np.int32)
	y = yd2.astype(np.int32)

	return x , y

def showresult(picture):
	plt.imshow(picture, interpolation='nearest')
	plt.show()