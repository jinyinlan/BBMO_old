import numpy as np
import pandas as pd
import os
from PIL import Image

def compute_mean(csv_file, root):

	fileName = os.path.join(root, csv_file)
	print str(fileName)

	enhance_param_frame = pd.read_csv(fileName)

	
	
	all_R = np.zeros(( 224*224, (len(enhance_param_frame)) ))
	all_G = np.zeros(( 224*224, (len(enhance_param_frame)) ))
	all_B = np.zeros(( 224*224, (len(enhance_param_frame)) ))
	
	all_mean = np.zeros((len(enhance_param_frame), 3))
	all_std = np.zeros((len(enhance_param_frame), 3))

	"""
	for index in range( len(enhance_param_frame) ):
		#img_name = str(root)+'/'+str(enhance_param_frame.ix[index, 0])+'-s.png'
		img_name = str(root)+'/'+str(enhance_param_frame.ix[index, 0])
			
		image = Image.open(img_name, 'r').resize((224, 224))
		npimage = np.array( image )/float(255)

		tempR = npimage[:,:,0].flatten()
		tempG = npimage[:,:,1].flatten()
		tempB = npimage[:,:,2].flatten()

		all_R[:,index] = tempR
		all_G[:,index] = tempG
		all_B[:,index] = tempB

	
		all_mean[index] = np.mean( (np.mean(npimage, axis=0)), axis=0 )
		#all_std[index] = np.std( (np.std(npimage, axis=0)), axis=0 )


	print np.mean(all_mean, axis=0)
	#print np.std(all_std, axis=0)
	#print np.std(all_R), np.std(all_G), np.std(all_B)
	"""
	for index in range( len(enhance_param_frame)/2 ):
		#img_name = str(root)+'/'+str(enhance_param_frame.ix[index, 0])+'-s.png'
		img_name = str(root)+'/'+str(enhance_param_frame.ix[index, 0])
			
		image = Image.open(img_name, 'r').resize((224, 224))
		npimage = np.array( image )/float(255)

		tempR = npimage[:,:,0].flatten()
		tempG = npimage[:,:,1].flatten()
		tempB = npimage[:,:,2].flatten()

		all_R[:,index] = tempR
		all_G[:,index] = tempG
		all_B[:,index] = tempB

	
		all_mean[index] = np.mean( (np.mean(npimage, axis=0)), axis=0 )
		all_std[index] = np.std( (np.std(npimage, axis=0)), axis=0 )


	print('mean', np.mean( all_mean, axis=0))
	print('std', np.std( all_std, axis=0))



	del all_R, all_G, all_B, all_mean, all_std

	all_R = np.zeros(( 224*224, (len(enhance_param_frame)) ))
	all_G = np.zeros(( 224*224, (len(enhance_param_frame)) ))
	all_B = np.zeros(( 224*224, (len(enhance_param_frame)) ))
	
	all_mean = np.zeros((len(enhance_param_frame), 3))
	all_std = np.zeros((len(enhance_param_frame), 3))


	for index in range( len(enhance_param_frame)/2,  len(enhance_param_frame) ):
		#img_name = str(root)+'/'+str(enhance_param_frame.ix[index, 0])+'-s.png'
		img_name = str(root)+'/'+str(enhance_param_frame.ix[index, 0])
			
		image = Image.open(img_name, 'r').resize((224, 224))
		npimage = np.array( image )/float(255)

		tempR = npimage[:,:,0].flatten()
		tempG = npimage[:,:,1].flatten()
		tempB = npimage[:,:,2].flatten()

		all_R[:,index] = tempR
		all_G[:,index] = tempG
		all_B[:,index] = tempB

	
		all_mean[index] = np.mean( (np.mean(npimage, axis=0)), axis=0 )
		all_std[index] = np.std( (np.std(npimage, axis=0)), axis=0 )


	print('mean', np.mean( all_mean, axis=0))
	print('std', np.std( all_std, axis=0))
	#print np.std(all_R), np.std(all_G), np.std(all_B)



if __name__ == '__main__':

	compute_mean("flickr-parameters-train.csv", "/home/omiya/data/flickr_teacherData0731/train")
	
