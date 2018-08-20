import numpy as np
import pylab as pl
import pandas as pd
from PIL import Image
import scipy.misc as smp
import matplotlib.pyplot as plt


# IBD dataset
# Normalized Data
disease_list = ['cir','col','obe','t2d','wt2d','ibd']
for disease in disease_list:

	data = pd.read_csv('./datasets/'+disease+'phy_x.csv')
	data = data.iloc[:,1:] # added
	data = data.T # added

	# np.save('./dataset_abundance_'+disease+'.npy',data)

	flat_data = data.values.flatten()
	mini = np.unique(sorted(flat_data))[1]
	maxi = np.unique(sorted(flat_data))[-1]

	n, bins, patches = plt.hist(flat_data, bins=np.logspace(np.log10(mini), np.log10(maxi+0.0001), 10)) # no of bins to play with
	print bins
	pl.gca().set_xscale("log")

	cm = plt.cm.get_cmap('RdYlBu_r')
	col = (n-n.min())/(n.max()-n.min())
	color_bins = []


	for c, p in zip(col, patches): 
	    pl.setp(p, 'facecolor', cm(c))
	    color_bins.append(list(cm(c)))


	rows, cols = data.shape
	if int(np.sqrt(cols))==np.sqrt(cols):
		length = np.sqrt(cols)
	else:
		length = int(np.sqrt(cols))+1

	indices = np.digitize(flat_data,bins)
	k = 0
	rdf_data = np.zeros(shape=[rows,cols])
	for i in range(rows):
		image_pixels = np.ones(shape=[length*length,3], dtype=np.uint8)
		image_pixels = image_pixels*255
		for j in range(cols):
			if indices[k]-1>=0:
				values = list(color_bins[indices[k]-1])
				values = [int(val * 255) for val in values]
				image_pixels[j,:] = values[0:3]
				rdf_data[i,j] = indices[k]
			k+=1

		img = image_pixels.reshape(length, length, 3)
		img = Image.fromarray(img)
		img.save('./images/'+disease+'/'+str(k/cols)+'_'+str(length)+'.png')
		np.save('./dataset'+disease+'.npy',rdf_data)
		
