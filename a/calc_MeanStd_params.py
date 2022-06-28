import numpy as np
import pandas as pd

def paramReader(paramfileName):
	f = open(paramfileName, 'r')
	for input_string in f:
		if input_string.split(' ')[0]=='exposure':
			exposure_black = input_string.split(' ')[2]
			exposure_exposure = input_string.split(' ')[3]
		if input_string.split(' ')[0]=='shadhi':
			shadhi_shadow = input_string.split(' ')[2]
			shadhi_whitepoint = input_string.split(' ')[3]
			shadhi_highlight = input_string.split(' ')[4]
			shadhi_shadowSat = input_string.split(' ')[5]
			shadhi_highlightSat = input_string.split(' ')[6]
		if input_string.split(' ')[0]=='colisa':
			colisa_contrast = input_string.split(' ')[2]
			colisa_light = input_string.split(' ')[3]
			colisa_saturation = input_string.split(' ')[4]
		if input_string.split(' ')[0]=='temperature':
			temperature_red = input_string.split(' ')[2]
			temperature_green = input_string.split(' ')[3]
			temperature_blue = input_string.split(' ')[4]
		if input_string.split(' ')[0]=='vibrance':
			vibrance_vibrance = input_string.split(' ')[2]
		if input_string.split(' ')[0]=='colorcorrection':
			colorcorrection_highX = input_string.split(' ')[2]
			colorcorrection_highY = input_string.split(' ')[3]
			colorcorrection_shadX = input_string.split(' ')[4]
			colorcorrection_shadY = input_string.split(' ')[5]
			colorcorrection_saturation = input_string.split(' ')[6]
		if input_string.split(' ')[0]=='colorcontrast':
			colorcontrast_GM = input_string.split(' ')[2]
			colorcontrast_BY = input_string.split(' ')[3]
	f.close()
	return np.array([ exposure_black, exposure_exposure, shadhi_shadow, shadhi_whitepoint, shadhi_highlight, shadhi_shadowSat, shadhi_highlightSat, colisa_contrast, colisa_light, colisa_saturation, temperature_red, temperature_green, temperature_blue, vibrance_vibrance, colorcorrection_highX, colorcorrection_highY, colorcorrection_shadX, colorcorrection_shadY, colorcorrection_saturation, colorcontrast_GM, colorcontrast_BY ]).astype(np.float64).astype(np.float64)


csvFileName = '/home/omiya/data/flickr_teacherData0814/train/flickr-parameters-train.csv'
csvFile = pd.read_csv(csvFileName)
fileName = csvFile['file name']
all_params = np.zeros(( len(fileName), 21))
for i in range(len(fileName)):
	root_dirName = '/home/omiya/data/flickr_teacherData0814/parameters/'
	num = fileName[i]
	name = num[:22]
	train_param_fileName = root_dirName+name+'_parameter_nm.txt'
	train_params = paramReader(train_param_fileName)
	np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
	#print fileName[i], train_params
	all_params[i,:] = train_params
mean = np.mean(all_params, axis = 0)
std = np.std(all_params, axis = 0)
print('mean:'+str(mean))
print('std:'+str(std))
