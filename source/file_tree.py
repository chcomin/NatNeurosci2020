'''Return file tree from a directory containing images separated into distinct
   folders.'''
   
import os
from igraph import Graph
import natsort

def get_files(inFolder, batchName, condStr=''):
	''' Get all files inside a root directory. '''

	forbiddenFolders = ['ROI', 'maximum_projection', 'equalized']   # Do not consider files inside these folders

	if os.name=='nt':    # If Windows
		if inFolder[-1]!='\\':
			if inFolder[-1]=='/':
				properInFolder = inFolder[:-1] + '\\'
				inFolder = inFolder[:-1] + '/'
			else:
				properInFolder = inFolder + '\\'
				inFolder = inFolder + '/'
	else:
		properInFolder = inFolder	

	foldersTree = Graph(directed=True)
	foldersTree.add_vertex(name=batchName, fullName=batchName+'/', isFile=0)
	nameMap = {batchName:0}
	dataTree = os.walk(properInFolder+batchName)

	k = 1
	for fullPfolder, folders, files in dataTree:
		if os.name=='nt':
			Pfolder = fullPfolder.split('\\')[-1]
		else:
			Pfolder = fullPfolder.split('/')[-1]			
		if Pfolder not in forbiddenFolders:
			sortedFiles = natsort.natsorted(files)
			sortedFolders = natsort.natsorted(folders)
			for file in sortedFiles:
				if condStr in file:
					foldersTree.add_vertex(name=file, fullName=foldersTree.vs[nameMap[Pfolder]]['fullName']+file, isFile=1)
					nameMap[file] = k
					k += 1
					foldersTree.add_edge(nameMap[Pfolder], nameMap[file])
			for folder in sortedFolders:
				if folder not in forbiddenFolders:
					foldersTree.add_vertex(name=folder, fullName=foldersTree.vs[nameMap[Pfolder]]['fullName']+folder+'/', isFile=0)
					nameMap[folder] = k
					k += 1			
					foldersTree.add_edge(nameMap[Pfolder], nameMap[folder])

	files = foldersTree.vs.select(isFile_eq=1)['fullName']
	
	return foldersTree, files