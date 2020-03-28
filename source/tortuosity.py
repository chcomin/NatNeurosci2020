'''Script for measuring blood vessel tortuosity'''

import numpy as np
import matplotlib.pyplot as plt
import os
import readimg
import tifffile
import natsort
import multiscale_tortuosity
import scipy.ndimage as nd
import file_tree
import time

def createAlphaImage(imgBack, indChange, colors, ABack=1., AFeature=0.7):
	'''Used for visualizing results'''

	img_f = imgBack.astype(float)
	AOut = AFeature + ABack*(1.-AFeature)

	for j in range(3):
		img_f[indChange[0], indChange[1],j] = (colors[:,j]*AFeature + img_f[indChange[0], indChange[1],j]*ABack*(1.-AFeature))/AOut
	img_f = np.round(img_f).astype(np.uint8)

	return img_f
	
	
def drawResults(img, featureImg, color=(255, 0, 0), cmap=None, outputFile=None, saturation=1., saturationFeature=1., AFeature=0.5):
	'''Used for visualizing results'''
	
	size_z, size_x, size_y = img.shape
	
	imgNorm = img - np.min(img)
	imgNorm = 255.*imgNorm/float(np.max(imgNorm))
	imgNorm = imgNorm*saturation
	imgNorm[imgNorm>255] = 255
	imgNorm = np.round(imgNorm).astype(np.uint8)
	imgMarked = np.zeros([size_z,size_x,size_y,3],dtype=np.uint8)
	imgMarked[:,:,:,0] = imgNorm.copy()
	imgMarked[:,:,:,1] = imgNorm.copy()
	imgMarked[:,:,:,2] = imgNorm.copy()
	
	imgFeatureNorm = featureImg - np.min(featureImg)
	imgFeatureNorm = 255.*imgFeatureNorm/float(np.max(imgFeatureNorm))	
	imgFeatureNorm = imgFeatureNorm*saturationFeature
	imgFeatureNorm[imgFeatureNorm>255] = 255	
	imgFeatureNorm = np.round(imgFeatureNorm).astype(np.uint8)
	ind = np.nonzero(imgFeatureNorm)
	
	if cmap==None:
		imgMarked[ind[0],ind[1],ind[2],:] = (255,0,0)
	else:
		cmap = plt.cm.get_cmap(cmap)
		colors = cmap(imgFeatureNorm[ind], bytes=True)
		colors = colors[:,:3]
		imgMarked[ind[0],ind[1],ind[2],:] = colors	
	
	imgProj = np.max(img,axis=0)
	imgProj = imgProj - np.min(imgProj)
	imgProj = 255*imgProj/float(np.max(imgProj))
	imgProj = imgProj*saturation
	imgProj[imgProj>255] = 255
	imgProj = np.round(imgProj).astype(np.uint8)
	imgMarkedProj = np.zeros([size_x,size_y,3],dtype=np.uint8)
	imgMarkedProj[:,:,0] = imgProj.copy()
	imgMarkedProj[:,:,1] = imgProj.copy()
	imgMarkedProj[:,:,2] = imgProj.copy()
	indProj = ind[1:]
	
	if cmap==None:
		imgMarkedProj[indProj[0],indProj[1],:] = (255,0,0)
	else:
		cmap = plt.cm.get_cmap(cmap)
		colors = cmap(imgFeatureNorm[ind], bytes=True)
		colors = colors[:,:3]
		imgFeatureProj = np.zeros([size_x,size_y,3],dtype=np.uint8)
		imgFeatureProj[indProj[0],indProj[1],:] = colors
		imgMarkedProj = createAlphaImage(imgMarkedProj, indProj, colors, ABack=1., AFeature=AFeature)
		#imgMarkedProj[indProj[0],indProj[1],:] = colors

	if outputFile!=None:
		fileNameLeft, fileNameRight = outputFile[0:-5], outputFile[-5:]
		fileNamePiece, fileExtension = fileNameRight.split('.')
		fileName = fileNameLeft+fileNamePiece
		
		#tifffile.imsaveWrapper(outputFile, imgMarked, scale=[1.,1.,1.], toUint8=True)			
		tifffile.imsaveWrapper(fileName+'_proj.'+fileExtension, imgMarkedProj, scale=[1.,1.,1.], toUint8=True)	

	return imgMarked, imgMarkedProj
	

batchName = 'Experiment #2 (P14 set #1)_samples'

inFolderNetwork = '../data/Network/'
inFolderOriginal = '../data/Original/'
outFolder = '../results/'

shouldDrawResults = False

L0 = 20				# Scale to calculate tortuosity

T = 2*L0/np.sqrt(2)

dataFolder = inFolderNetwork+batchName+'/core/numpy/'

filesTree, files = file_tree.get_files(inFolderOriginal, batchName)

allTortuosity = []
allTortuosityWholeEdge = []
edgeSizes = []
for q in range(len(files)):	
	print(q)
	t = time.time()

	splitIndex = files[q].rfind('/')
	filePath, file = files[q][:splitIndex]+'/', files[q][splitIndex+1:]
	dotIndex = file.rfind('.')
	fileName = file[:dotIndex]
	
	img, scale_or = readimg.ReadImg(inFolderOriginal+filePath+file, 0)
	scale = [np.min(scale_or)]*3		
	
	if shouldDrawResults:
		zoomRatio = scale_or/np.min(scale_or)
		img_interp = nd.interpolation.zoom(img,zoomRatio,order=1)	# 2nd order is slightly better than 1st	
		featureImage = np.zeros_like(img_interp, dtype=float)	

	g = np.load(inFolderNetwork+filePath+'/core/numpy/'+fileName+'.npy').item()		

	allTortuosity.append([])
	allTortuosityWholeEdge.append([])
	for edgeIndex in range(g.ecount()):

		ed_pixels = np.array(g.es['pos_ed'][edgeIndex])
		ed = scale*ed_pixels
		
		edgeSize = np.sqrt(np.sum((ed[-1]-ed[0])**2))
		edgeSizes.append(edgeSize)

		if ed.shape[0]>(T/scale[0]):
			edgeTortuosity, leftBorderIndex, rightBorderIndex = multiscale_tortuosity.getEdgeTortuosity(ed, L0)		
			if rightBorderIndex>leftBorderIndex:
				validEdgeTortuosity = edgeTortuosity[leftBorderIndex:rightBorderIndex]		

				allTortuosityWholeEdge[-1].append(validEdgeTortuosity)
				allTortuosity[-1].append(np.median(validEdgeTortuosity))

				edgeSegment = ed[leftBorderIndex:rightBorderIndex]				
				if shouldDrawResults:
					featureImage[edgeSegment[:,0],edgeSegment[:,1],edgeSegment[:,2]] = validEdgeTortuosity				
				
	if shouldDrawResults:	
		imgMarked, imgMarkedProj = drawResults(imgOrInterp, featureImage, color=(255, 0, 0), cmap='jet', outputFile=None, saturation=1., saturationFeature=1., AFeature=0.5)	

	print('Took %d seconds'%(time.time()-t))

tortThreshold = 3

numFiles = len(allTortuosity)
avgTortMean = np.zeros(numFiles)
fracTortMax = np.zeros(numFiles)
for imgIndex, imgTort in enumerate(allTortuosityWholeEdge):
	tortMean = np.zeros(len(imgTort))
	tortMax = np.zeros(len(imgTort))
	for edgeIndex, edgeTort in enumerate(imgTort):
		tortMean[edgeIndex] = np.mean(edgeTort)
		
	avgTortMean[imgIndex] = np.mean(tortMean)
	
	
strData = ",Sample,Tortuosity\n"	
for i in range(len(files)):
	filePieces = files[i].split('/')
	folders = filePieces[1:-1]
	fileName = filePieces[-1]
	dotIndex = fileName.rfind('.')
	fileName = fileName[:dotIndex]
	
	strData += ','.join(folders)+','+fileName+','
	strData += '%.3f\n'%(avgTortMean[i])

fd = open(outFolder+'results_'+batchName+'_tortuosity.txt', 'w')
fd.write(strData)
fd.close()



