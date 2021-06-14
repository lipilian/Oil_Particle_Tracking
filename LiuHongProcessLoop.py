# %% import the package
import cv2
print(cv2.__version__)
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import pims
import trackpy as tp
import numpy as np
import math
from tqdm import tqdm
import os
import glob
from skimage.feature import canny
from skimage.draw import line
from skimage.transform import probabilistic_hough_line
from itertools import combinations
from openpiv import tools, scaling, pyprocess, validation, process, filters
import multiprocessing
from joblib import Parallel, delayed

# %% glob all file path and preprocessed
fileList = glob.glob('./calibration/*.avi')
for fullPath in fileList:
    fileName = fullPath.split('\\')[-1]
    fileName = fileName.split('.av')[0]
    SavePath = fileName
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
    tempPath = os.path.join(SavePath,'temp')
    if not os.path.exists(tempPath):
        os.mkdir(tempPath)
    vid = cv2.VideoCapture(fullPath)
    frames = []

    while(vid.isOpened()):
        ret,frame = vid.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        else:
            break
    vid.release()
    frames = np.array(frames)
    median = np.max(frames, axis = 0)
    i = 0
    for frame in frames:
        frameName = str(i)
        frameName = frameName.zfill(5)
        frame = median - frame
        frame[frame < 0] = 0
        cv2.imwrite(os.path.join(tempPath, frameName + '.tif'), frame)
        i = i + 1
      
# %%  piv proprecessed
pathAll = glob.glob('*S00*')
for path in pathAll:
    calibrationFactor = 1
    resultPath = os.path.join('result', path)
    if not os.path.exists(resultPath):
        os.mkdir(resultPath)
    imagenames = glob.glob(path + '/temp/*.tif')
    fileNumber = len(imagenames)
    N = fileNumber
    print('fileNums is {}'.format(fileNumber))    
    fps = 6400
    startFrame = 100
    DeltaFrame = 2
    frame_a = cv2.imread(imagenames[startFrame], 0)
    frame_b = cv2.imread(imagenames[startFrame + DeltaFrame], 0)
    winsize = 20 # pixels
    searchsize = 20#pixels
    overlap = 10 # piexels
    dt = DeltaFrame*1./fps # piexels
    u0, v0, sig2noise = process.extended_search_area_piv(frame_a.astype(np.int32), frame_b.astype(np.int32), window_size=winsize, overlap=overlap, dt=dt, search_area_size=searchsize, sig2noise_method='peak2peak' )
    x, y = process.get_coordinates( image_size=frame_a.shape, window_size=winsize, overlap=overlap )  
    u1, v1, mask = validation.sig2noise_val( u0, v0, sig2noise, threshold = 1.3)
    u2, v2 = filters.replace_outliers( u1, v1, method='localmean', max_iter=5, kernel_size=5)
    u3, v3, mask1 = validation.local_median_val(u2,v2,3,3,1)
    u4, v4 = filters.replace_outliers(u3, v3, method='localmean', max_iter=5, kernel_size=5)
    mask1[:] = False
    tools.save(x, y, u1, v1, mask, './result/' + path + '/test.txt')
    
    
    frames_color = pims.MoviePyReader('./calibration/' + path + '.avi')
    
    fig, ax = plt.subplots(2,1,figsize=(18,9))
    ax[0].imshow(frames_color[startFrame], cmap = plt.cm.gray, aspect = 'auto')

    tools.display_vector_field('./result/' + path + '/test.txt',ax = ax[1], scale=2000, width=0.0005)
    [m,n] = frame_a.shape
    ax[1].set_xlim([0, n])
    ax[1].set_ylim([m, 0])
    
    
    
    
    break



# %%
for fullPath in fileList:
    frames_color = pims.MoviePyReader(fullPath)
    fileName = fullPath.split('\\')[-1]
    fileName = fileName.split('.av')[0]
    SavePath = fileName
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
    
    
    
    @pims.pipeline
    def as_grey(frame):
        red = frame[:, :, 0]
        green = frame[:, :, 1]
        blue = frame[:, :, 2]
        return 0.2125 * red + 0.7154 * green + 0.0721 * blue
    frames_gray = as_grey(frames_color)    
    
    m, n, _  = frames_gray.frame_shape
    t = len(frames_gray)
    temp = np.zeros((m,n,t), dtype = frames_gray.pixel_type)
    for kk in range(t):
        temp[:,:,kk] = np.array(frames_gray[kk])
        
    median = np.max(temp, axis = 2)
   
    @pims.pipeline
    def removeBack(frame, median):
        frame = median - frame
        frame[frame < 0] = 0
        return frame
    frames = removeBack(frames_gray, median)
    break

# %%----------------------------------------------------------------
fps = 6400
startFrame = 100
DeltaFrame = 2
frame_a = np.array(frames[startFrame])
frame_b = np.array(frames[startFrame + DeltaFrame])
winsize = 20 # pixels
searchsize = 20#pixels
overlap = 10 # piexels
dt = DeltaFrame*1./fps # piexels
u0, v0, sig2noise = process.extended_search_area_piv(frame_a.astype(np.int32), frame_b.astype(np.int32), window_size=winsize, overlap=overlap, dt=dt, search_area_size=searchsize, sig2noise_method='peak2peak' )
x, y = process.get_coordinates( image_size=frame_a.shape, window_size=winsize, overlap=overlap )  
u1, v1, mask = validation.sig2noise_val( u0, v0, sig2noise, threshold = 1.3)
u2, v2 = filters.replace_outliers( u1, v1, method='localmean', max_iter=5, kernel_size=5)
u3, v3, mask1 = validation.local_median_val(u2,v2,3,3,1)
u4, v4 = filters.replace_outliers(u3, v3, method='localmean', max_iter=5, kernel_size=5)
mask1[:] = False
tools.save(x, y, u1, v1, mask1, './test.txt')
# %%
fig, ax_array = plt.subplots(2,1,figsize=(18,9))
ax_array[0].imshow(frame_a, cmap = plt.cm.gray, aspect = 'auto')
tools.display_vector_field('./test.txt',ax = ax_array[1], scale=2000, width=0.0005)

# %%

fileList = glob.glob('./calibration/*/*.tif')
for fullPath in fileList:
    
    frames = pims.open(fullPath)
    img = np.array(frames[0]); img = (img/256).astype('uint8')
    edges = canny(img, 3,5,20)
    lines = probabilistic_hough_line(edges, threshold=30, line_length=400,
                                    line_gap=200)
    fig = plt.figure()
    for line1, line2 in combinations(lines, 2):
        #p0, p1 = line
        #plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
        x0, y0 = line1[0]; x1, y1 = line1[1]
        a0, b0 = line2[0]; a1, b1 = line2[1]
        theta0 = (y1 - y0)/(x1 - x0); theta1 = (b1 - b0)/(a1 - a0)
        if abs(theta1) > 0.1 or abs(theta0) > 0.1:
            print('line detection wrong')
            break
        distance = abs((b0 + b1)/2 - (y1 + y0)/2)
        if distance > 40:
            break
    plt.imshow(img,cmap='gray')
    plt.plot((x0,x1), (y0,y1))
    plt.plot((a0,a1), (b0,b1))

   
    yMax = np.max(np.array([y1,y0,b1,b0]))
    yMin = np.min(np.array([y1,y0,b1,b0]))
    plt.xlabel('crop with ymin = {}, ymax = {}'.format(yMin, yMax))
    
    fileName = fullPath.split('\\')[-1]
    fileName = fileName.split('_MMStack')[0]
    SavePath = fileName
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
    
    print(fileName)
    estimateFeatureSize = 5
    minMass = 20000
    
    
    @pims.pipeline
    def cropImage(frame):
        frame[0:yMin,:] = 0
        frame[yMax:,:] = 0    
        return frame
    frames = pims.open(fullPath)
    
    m, n = frames.frame_shape
    t = len(frames)
    temp = np.zeros((m,n,t), dtype = frames.pixel_type)
    for kk in range(t):
        temp[:,:,kk] = np.array(frames[kk])
    median = np.median(temp, axis = 2)
    
    @pims.pipeline
    def removeBack(frame, median):
        frame = frame - median
        return frame
    frames = removeBack(frames, median)
    
    #break
    #frames = cropImage(frames)
    
    fig.savefig(SavePath + '/Line.jpg')
    
    
    
    f = tp.batch(frames, estimateFeatureSize, minmass = minMass)    
    t = tp.link(f, 60, memory=80)
    
    t1 = tp.filter_stubs(t, 10)
    tp.plot_traj(t1)

    Ntrajs = np.max(np.array(t1['particle'])) + 1

    minMoveDistance = 300
    print('there are %s trajectories' % Ntrajs)
    t2 = t1[0:0]
    for i in range(Ntrajs):
        tNew = t1[t1['particle']==i]
        if(len(tNew) < 30):
            continue
        #distData = tp.motion.msd(tNew,1,1,len(tNew))
            #dist = distData.iloc[-1,:]['msd']
        x0 = tNew.iloc[0,:]['x']
        y0 = tNew.iloc[0,:]['y']
        xend = tNew.iloc[-1,:]['x']
        yend = tNew.iloc[-1,:]['y']
        dist = np.sqrt((xend - x0)**2 + (yend - y0)**2)
        print('partile index:' , i ,' traveling distance: ', dist)
        if dist > minMoveDistance:
            t2 = t2.append(tNew)

    k,ax1 = plt.subplots(1, figsize=(60,20))
    #ax.plot(x1,y,'-',linewidth = 2, color='red')
    #ax.plot(x2,y,'-',linewidth = 2, color='red')
    tp.plot_traj(t2)
    ax1.imshow(img, cmap= 'gray')
    k.savefig(SavePath + '/images.jpg')
    t2.to_csv(SavePath + '/pointsData.csv')
    
   
# %%


# %% test
fig = plt.figure()
f = tp.locate(np.array(frames[1318]), estimateFeatureSize)

fig, ax = plt.subplots()
ax.hist(f['mass'], bins=20)
ax.set(xlabel='mass', ylabel='count');

minMass = 20000
fig = plt.figure()
f = tp.locate(frames[1318], estimateFeatureSize, minmass= minMass)
tp.annotate(f, frames[1318])
    
# %%