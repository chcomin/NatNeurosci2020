'''Detect neurons and calculate their density as a function of distance to surface'''

import numpy as np
import matplotlib.pyplot as plt
import PIL as pil
import os
from igraph import Graph
import natsort
import scipy.ndimage as ndi
from scipy.interpolate import splprep, splev, spalde
import skimage
from skimage.feature import blob_log
import cv2
import json

def getFiles(inFolder, batchName):
    
    forbiddenFolders = []

    if os.name=='nt':
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

def find_background_comps(img_ROI_inv):
    '''Detect bachground of image (everything outside the tissue)'''

    img_lab, num_comp = ndi.label(img_ROI_inv)
    tam_comp = ndi.sum(img_ROI_inv, img_lab, range(1, num_comp+1))
    k = np.argsort(tam_comp)[-2:]
    ROI_inv_comp1 = img_lab==k[0]+1
    ROI_inv_comp2 = img_lab==k[1]+1
    cm1 = ndi.center_of_mass(ROI_inv_comp1)
    cm2 = ndi.center_of_mass(ROI_inv_comp2)
    dist1 = np.sqrt(cm1[0]**2+cm1[1]**2)
    dist2 = np.sqrt(cm2[0]**2+cm2[1]**2)
    if dist1<dist2:
        surf_comp = ROI_inv_comp1
        deep_comp = ROI_inv_comp2
    else:
        surf_comp = ROI_inv_comp2
        deep_comp = ROI_inv_comp1
        
    return surf_comp, deep_comp

def open_images(root_folder, ROI_folder, files, file_idx, channel_order, plot=False):
    '''Open images taking care of selecting the correct channels for NeuN and TBR1.'''
    
    img_c1 = pil.Image.open(root_folder+files[file_idx])
    img_c1 = np.array(img_c1)
    img_c2 = pil.Image.open(root_folder+files[file_idx+1])
    img_c2 = np.array(img_c2)
    filename = files[file_idx].split('/')[-1].split('.')[0][:-3]
    try:
        img_ROI = pil.Image.open(ROI_folder+f'{filename}_C1.png')
    except FileNotFoundError:
        try:
            img_ROI = pil.Image.open(ROI_folder+f'{filename}_C2.png')
        except FileNotFoundError:
            img_ROI = pil.Image.open(ROI_folder+f'{filename}_C3.png')
    img_ROI = np.array(img_ROI)>0
    
    found_condition_all = False
    for case in channel_order:
        found_condition = True
        for condition in case[0]:
            if condition not in files[file_idx]:
                found_condition = False
        if found_condition:
            NeuN_channel = case[1]
            if found_condition_all:
                raise ValueError("Error, more than one condition found.")
            else:
                found_condition_all = True
                
    
    if NeuN_channel==1:
        img_temp = img_c1.copy()
        img_c1 = img_c2.copy()
        img_c2 = img_temp
    
    if plot:
        plt.figure(figsize=[10,8])
        plt.subplot(2, 2, 1)
        plt.imshow(img_c1, 'gray')
        plt.subplot(2, 2, 2)
        plt.imshow(img_c2, 'gray')
        plt.subplot(2, 2, 3)
        plt.imshow(img_ROI, 'gray')
            
    return img_c1, img_c2, img_ROI, filename

def get_file_key(path, first_level=1, last_level=-2):
    
    folders = path.split('/')[first_level:last_level+1]
    key = '-'.join(folders)
    return key

def find_surface_cont(surf_comp):
    '''Find contour of the surface of the cortex'''
    
    _, contours, _ = cv2.findContours(surf_comp.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    contour = contours[0][:,0]
    surface_cont = []
    for point in contour:
        if point[0]>0 and point[1]>0 and point[0]<surf_comp.shape[1]-1 and point[1]<surf_comp.shape[0]-1:
            surface_cont.append(point)
    surface_cont = np.array(surface_cont)
            
    return surface_cont

def get_point_info(u, tck, orient_factor=10000):
    '''Get position, tangent and normal of interpolated point'''
    
    derivatives = spalde(u, tck)
    p = (derivatives[0][0], derivatives[1][0])
    tangent = (derivatives[0][1], derivatives[1][1])
    tangent = tangent/np.sqrt(tangent[0]**2+tangent[1]**2)

    if np.abs(tangent[0])<1e-4:
        normal = np.array([0, 1])
    else:
        normal = np.array([-tangent[1]/tangent[0], 1])
        normal = normal/np.sqrt(normal[0]**2+normal[1]**2)

    if np.minimum(p[0]+orient_factor*normal[0], p[1]+orient_factor*normal[1])<0:
        normal = -normal
        
    return p, tangent, normal

def get_normal_line(u, tck, shape):
    '''Get normal to a inteprolated curve at point u'''
    
    p, _, n = get_point_info(u, tck)

    if np.abs(n[0])>1e-5:
        fac_max_c = (shape[1]+5 - p[0])/n[0]
    else:
        fac_max_c = 0
    if np.abs(n[1])>1e-5:
        fac_max_r = (shape[0]+5 - p[1])/n[1]    
    else:
        fac_max_r = 0
    fac_max = np.minimum(fac_max_c, fac_max_r)

    if np.abs(n[0])>1e-5:
        fac_min_c = (5 + p[0])/n[0]
    else:
        fac_min_c = 0
    if np.abs(n[1])>1e-5:
        fac_min_r = (5 + p[1])/n[1]    
    else:
        fac_min_r = 0
    fac_min = np.minimum(fac_min_c, fac_min_r)
    
    
    rr, cc = skimage.draw.line(int(p[1]-fac_min*n[1]), int(p[0]-fac_min*n[0]), int(p[1]+fac_max*n[1]), int(p[0]+fac_max*n[0]))
    ind = np.nonzero((rr>=0) & (rr<shape[0]) & (cc>=0) & (cc<shape[1]))
    rr = rr[ind]
    cc = cc[ind]

    return rr, cc

def get_quantification_ROI(img_ROI, s_factor=10, frac_rem=0.2, plot=False, debug=False):
    '''Get ROI for quantification. Defined by the surface of the cortex and the maximal cortical depth and,
       at the borders, the normals to the cortical surface.'''

    img_ROI_inv = np.logical_not(img_ROI)
    surf_comp, deep_comp = find_background_comps(img_ROI_inv)
    
    # Fit spline to surface
    surface_cont = find_surface_cont(surf_comp)
    tck, u = splprep([surface_cont[:,0], surface_cont[:,1]], s=s_factor*surface_cont.shape[0])
    
    ## Draw lines defined by normals
    # Get normals
    img_del = np.zeros_like(img_ROI)
    rr0, cc0 = get_normal_line(frac_rem, tck, img_ROI.shape)
    img_del[rr0,cc0] = 1
    rr1, cc1 = get_normal_line(1-frac_rem, tck, img_ROI.shape)
    img_del[rr1,cc1] = 1
    
    # Find component inside lines
    middle_point = np.mean(surface_cont, axis=0)[::-1]

    middle_point = (int(middle_point[0]), int(middle_point[1]))

    img_lab, num_comp = ndi.label(np.logical_not(img_del))
    img_ROI_quant = img_lab == img_lab[middle_point[0], middle_point[1]]
    img_ROI_quant = img_ROI_quant*img_ROI
    
   
    # Plot one of the normals
    if plot:
        new_points = splev(u, tck)
        p0, _, n0 = get_point_info(frac_rem, tck)
        
        plt.figure()
        plt.subplot(2, 2, 1, aspect='equal')
        plt.plot(surface_cont[:,0], surf_comp.shape[0]-surface_cont[:,1], '-o')
        plt.subplot(2, 2, 2, aspect='equal')
        plt.plot(new_points[0], surf_comp.shape[0]-new_points[1], '-o')
        plt.plot(p0[0], surf_comp.shape[0]-p0[1], 'o')
        plt.plot(p0[0]+50*n0[0], surf_comp.shape[0]-(p0[1]+50*n0[1]), 'o')
        plt.subplot(2, 2, 3)
        plt.imshow(img_ROI_quant)    
        plt.subplot(2, 2, 4)
        plt.imshow(img_del) 

    return img_ROI_quant, surf_comp

def detect_nuclei(img, img_dist, img_sub=None, sub_factor=1, min_sigma=3, max_sigma=6, num_sigma=4, log_threshold=0.05, overlap=0.25,
                  sig_x0=100, sig_k=0.05, apply_sigmoid=True, return_img=True):
    
    if img_sub is None:
        img_diff = img.copy()
        img_diff -= np.min(img_diff)
        img_diff = img_diff/float(np.max(img_diff))
    else:
        img_diff = img.astype(float)-sub_factor*img_sub.astype(float)
        img_diff[img_diff<0] = 0
        img_diff = img_diff/np.max(img_diff)

    if apply_sigmoid:
        img_sig = 1./(1+np.exp(-sig_k*(img_dist-sig_x0)))
        img_nuclei = img_sig*img_diff
    else:
        img_nuclei = img_diff

    blobs = blob_log(img_nuclei, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, 
                     threshold=log_threshold, overlap=overlap)
    
    blobs_pos = blobs[:, :2].astype(int)
    blobs_radius = blobs[:, 2]*np.sqrt(2)

    if return_img:
        img_nuclei = np.zeros_like(img_nuclei, dtype=np.uint8)
        img_nuclei[blobs_pos[:,0], blobs_pos[:,1]] = 1
        return img_nuclei
    else:
        return blobs_pos, blobs_radius

def mark_nuclei_centers(nuclei, img_diff, radius=1, bright_factor=2):
        
    if nuclei.shape[1]==2 or nuclei.shape[1]==3:
        img_nuclei = np.zeros_like(img_diff, dtype=np.uint8)
        img_nuclei[nuclei[:,0], nuclei[:,1]] = 1
    else:
        img_nuclei = nuclei.copy()
        
    img_diff_brighter = img_diff.copy()
    img_diff_brighter = bright_factor*img_diff_brighter
    img_diff_brighter[img_diff_brighter>1] = 1
    
    img_diff_brighter = (255*img_diff_brighter).astype(np.uint8)

    img_mark = np.zeros((img_diff.shape[0], img_diff.shape[1], 3), dtype=np.uint8)
    img_mark[:,:,0] = img_diff_brighter.copy()
    img_mark[:,:,1] = img_diff_brighter.copy()
    img_mark[:,:,2] = img_diff_brighter.copy()
    
    ind = np.nonzero(img_nuclei>0)
    for i in range(len(ind[0])):
        rr, cc = skimage.draw.circle(ind[0][i], ind[1][i], radius=radius, shape=img_diff.shape)
        img_mark[rr, cc, 0] = 255
        img_mark[rr, cc, 1] = 0
        img_mark[rr, cc, 2] = 0

    return img_mark

def get_shell(img, img_dist, shell_width, index):
    '''Get image inside specific shell (distance larger than x and smaller than x+shell_width)'''
    
    first_dist = shell_width*index
    last_dist = shell_width*(index+1)
    img_shell = (img_dist>first_dist) & (img_dist<=last_dist)
    img_in_shell = img*img_shell
    
    return img_in_shell, img_shell
    
def plot_nuclei_centers(nuclei_list, img_diff, bright_factor=2):
        
    img_diff_brighter = img_diff.copy()
    img_diff_brighter = bright_factor*img_diff_brighter
    img_diff_brighter[img_diff_brighter>1] = 1

    plt.figure(figsize=[10,10])
    ax = plt.subplot(111)
    ax.imshow(img_diff_brighter, 'gray')
    for blob in nuclei_list:
        y, x, r = blob
        plt.scatter(x, y, s=1, c='red')
        
def plot_contours(img_c1, img_c2, img_dist, figsize=[9,5], levels=20, fig=None):
    
    if fig is None:
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
    else:
        ax1, ax2 = fig.axes
        ax1.clear()
        ax2.clear()
    
    ax1.imshow(img_c1, 'gray')
    ax1.contour(img_dist, levels=levels)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    
    ax2.imshow(img_c2, 'gray')
    ax2.contour(img_dist, levels=levels)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    fig.tight_layout()
    
def plot_marks(img_mark_c1, img_mark_c2, figsize=[9,5], fig=None):

    if fig is None:
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
    else:
        ax1, ax2 = fig.axes
        ax1.clear()
        ax2.clear()

    ax1.imshow(img_mark_c1, 'gray')
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax2.imshow(img_mark_c2, 'gray')
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    fig.tight_layout()
    
def write_data(nuc_density_all, filename, folder=None):
    '''Write results as csv'''
    
    if folder is None:
        folder = './'
    elif folder[-1]!='/':
        folder += '/'
    
    shell_width = nuc_density_all['info']['shell_width_um']
    max_num_shells = 0
    for k, v in nuc_density_all.items():
        if k is not 'info':
            s = len(v['C1'])
            if s>max_num_shells:
                max_num_shells = s

    for channel in ['C1', 'C2']:
        fd = open(f'{folder}{filename}_{channel}.csv', 'w')
        line = 'Sample name;'
        for i in range(max_num_shells-1):
            line += f'{i*shell_width}-{(i+1)*shell_width};'
        line += f'{(max_num_shells-1)*shell_width}-{(max_num_shells)*shell_width}\n'
        fd.write(line)
        for k, v in nuc_density_all.items():
            if k is not 'info':
                #file = k.replace('-', ';')
                line = f'{k};'
                c_data = v[f'{channel}']
                for value in c_data[:-1]:
                    line += f'{value:.1f};'
                line += f'{c_data[-1]:.1f}\n'
                fd.write(line)
        fd.close()


data_folder = '../data/Neuron/Original/'
ROI_folder = '../data/Neuron/ROI/'
out_folder = '../data/Neuron/'
batch_name = 'Neuronal density quantifications'
# 0 means NeuN (large oval-shaped neurons) are in first channel, 1 means that TBR1 (small, round neurons)
# are in first channel
channel_order = [[["Conditional mutants", "P50 females"], 0], [["Conditional mutants", "P50 males"], 0],
                 [["P50 (females)"], 0], [["P14 (males)"], 1], [["P50 (males)"], 1], [["P0"], 1],
                 [["P14 females"], 0]]
tree, files = getFiles(data_folder, batch_name)
files = natsort.natsorted(files)

scale = 0.908            # microns/px
shell_width_um = 50      # microns
log_threshold_c2 = 0.06


shell_width = shell_width_um/scale

fig1 = plt.figure(figsize=[9,5])
fig1.add_subplot(1, 2, 1)
fig1.add_subplot(1, 2, 2)
fig2 = plt.figure(figsize=[9,5])
fig2.add_subplot(1, 2, 1)
fig2.add_subplot(1, 2, 2)

nuc_density_all = {'info':{'scale':scale, 'shell_width_um':shell_width_um}}
for file_idx in range(0, len(files), 2):
    print(file_idx)
    file_key = get_file_key(files[file_idx], first_level=1, last_level=-2)

    img_c1, img_c2, img_ROI, filename = open_images(data_folder, ROI_folder, files, file_idx, channel_order)

    # Surface distance
    img_ROI_quant, surf_comp = get_quantification_ROI(img_ROI, s_factor=100)
    surf_comp_inv = np.logical_not(surf_comp)
    img_dist = ndi.distance_transform_edt(surf_comp_inv)
    img_dist_ROI = img_dist*img_ROI_quant
    
    img_c1_s = ndi.gaussian_filter(img_c1, sigma=4)
    img_c2_s = ndi.gaussian_filter(img_c2, sigma=4)
    
    img_c1_eq = skimage.exposure.equalize_adapthist(img_c1, kernel_size=151, clip_limit=0.03)

    # Nuclei detection
    nuclei_centers_c1 = detect_nuclei(img_c1_eq, img_dist, min_sigma=4, max_sigma=6, num_sigma=4, log_threshold=0.06, overlap=0.25,
                      sig_x0=100, sig_k=0.05, return_img=True)
    nuclei_centers_c1 = nuclei_centers_c1*img_ROI
    img_mark_c1 = mark_nuclei_centers(nuclei_centers_c1, img_c1/255, radius=2.5, bright_factor=2)
    nuclei_centers_c2 = detect_nuclei(img_c2, img_dist, img_c1_s, sub_factor=0.5, min_sigma=4, max_sigma=6, num_sigma=4, log_threshold=log_threshold_c2, overlap=0.25,
                      sig_x0=100, sig_k=0.05, return_img=True)
    nuclei_centers_c2 = nuclei_centers_c2*img_ROI
    img_mark_c2 = mark_nuclei_centers(nuclei_centers_c2, img_c2/255, radius=2.5, bright_factor=2)
    
    num_full_shells = int(np.floor(np.max(img_dist_ROI)/shell_width))
    if np.max(img_dist_ROI)/shell_width-num_full_shells>0:
        num_shells = num_full_shells + 1
    else:
        num_shells = num_full_shells

    nuc_density_c1 = []
    nuc_density_c2 = []
    for i in range(num_shells):
        img_nuc_shell, img_shell = get_shell(nuclei_centers_c1, img_dist_ROI, shell_width, i)
        area = np.sum(img_shell)*scale**2*1e-6
        nuc_density_c1.append(np.sum(img_nuc_shell)/area)
        img_nuc_shell, img_shell = get_shell(nuclei_centers_c2, img_dist_ROI, shell_width, i)
        nuc_density_c2.append(np.sum(img_nuc_shell)/area)

    nuc_density_all[file_key] = {'C1':nuc_density_c1, 'C2':nuc_density_c2}
    
    # Save data
    levels = np.arange(0, shell_width*num_full_shells+0.1, shell_width)
    plot_contours(img_c1, img_c2, img_dist_ROI, fig=fig1, levels=levels)
    fig1.savefig(f"{out_folder}detection/surface/{file_key}.png")

    plot_marks(img_mark_c1, img_mark_c2, fig=fig2)
    fig2.savefig(f"{out_folder}detection/nuclei/{file_key}.png", dpi=128)
    
#json.dump(nuc_density_all, open('nuclei_density.json', 'w'), indent=2)
write_data(nuc_density_all, 'neuron_density', '../results/')

