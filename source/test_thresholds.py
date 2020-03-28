import numpy as np
import scipy.ndimage.measurements as me
import scipy.ndimage as nd
import scipy.ndimage.filters as fil
import matplotlib.pyplot as plt
import os
import readimg
import natsort

def make_folders(outFolder,batchName,files):

	if (os.path.isdir(outFolder+batchName)==False):	
		os.mkdir(outFolder+batchName)

def remove_conn_comp_v2(imgBin, tamThreshold=20):

    imgLab,numComp = nd.label(imgBin)
    tamComp = nd.sum(imgBin,imgLab,range(numComp+1))

    mask = tamComp>tamThreshold
    mask[0] = False

    img_f = mask[imgLab]

    return img_f	


batchName = 'Experiment #2 (P14 set #1)_samples'

inFolder = inFolder = '../data/Original/'+batchName+'/'
outFolder = '../data/tests/threshold/'

C_list = range(1, 8)

dataTree = os.walk(inFolder)
files = []
for Pfolder, subFolders, subFolderFiles in dataTree:
	if len(subFolderFiles)>0:
		ind = Pfolder.rfind(batchName)+len(batchName)+1
		folders = Pfolder[ind:].split('\\')
		
		if (folders[-1]!='ROI') and (folders[-1]!='maximum_projection') and (folders[-1]!='equalized'):
			if (len(folders[0])>0):
				folders = [folder+'/' for folder in folders]
					
			for subFolderFile in subFolderFiles:
				#if '10X' in subFolderFile:
				files.append(folders+[subFolderFile])


make_folders(outFolder,batchName,files)
files = natsort.natsorted(files)

for q in range(len(files)):	

	file = inFolder+''.join(files[q])
	ind = files[q][-1].rfind('.')
	fileName = files[q][-1][0:ind]
	fileFolder = files[q][:-1]

	dictKey = ''
	for filFol in fileFolder:
		if filFol != '':
			dictKey += '%s-'%(filFol.strip('/'))
	dictKey += fileName	
	
	outFolderComplete = outFolder+batchName+'/'
		
	print("\nProcessing image \""+str(dictKey)+"\" ("+str(q+1)+" of "+str(len(files))+")")
	
	img_np_or,scale = readimg.ReadImg(file, 0)

	zoomRatio = scale/np.min(scale)
	img_np = nd.interpolation.zoom(img_np_or,zoomRatio,order=2)	
	newScale = [np.min(scale)]*3	
	
	size_z,size_x,size_y = img_np.shape

	sigma = np.array([0.1,1.,1.])
	img_np_diffused = fil.gaussian_filter(img_np,sigma=sigma/scale) 

	radius = 40
	comp_size = 500
	img_corr = np.zeros([size_z,size_x,size_y])
	for i in range(size_z):
		img_blurred = fil.gaussian_filter(img_np_diffused[i],sigma=radius/2.)
		img_corr[i] = img_np_diffused[i] - img_blurred


	for C in C_list:
		img_T = img_corr > C	

		img_T_larg_comp = remove_conn_comp_v2(img_T, comp_size)

		[img_lab, num_comp] = me.label(1-img_T_larg_comp)
		tam_comp = me.sum(1-img_T_larg_comp, labels=img_lab, index=range(1,num_comp+1))
		ind = np.argmax(tam_comp)+1
		img_T_final = img_lab!=ind

		img_diff = np.logical_xor(img_T, img_T_final)
		img_final_diff = img_T_final.astype(np.uint8)*2
		img_final_diff[img_diff] = 1


		outFolderComplete = outFolder+batchName+'/'
		plt.imsave(outFolderComplete+dictKey+'_%.1f.png'%(C),np.max(img_final_diff,axis=0),cmap='hot')
	

	
