import torch
import os
import time
import sys
import datetime
import numpy as np
#import cv2

from PIL import Image
from argparse import ArgumentParser

from torch.optim import SGD, Adam, Adadelta
from torch.autograd import Variable
from torchvision.transforms import Compose, Scale, CenterCrop, Normalize, RandomHorizontalFlip, RandomCrop
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import DataLoader

from network import myAlexNet, myVGG, myVGG_bn, myVGG19, myVGG19_bn, myResNet18
from dataset import FlickrInteresting
from normalize import denormalize
#import param_to_photoEnhance2
import param_to_photoEnhance2test

#for python3, torch0.4.0
"""
def mse(imageA, imageB):
	err = np.mean((imageA.astype("float")/255 - imageB.astype("float")/255) ** 2)
	#err /= float(imageA.shape[0] * imageA.shape[1])
	
	return err

"""

def mse_pil(imageA, imageB):
	err = np.mean( (imageA.astype("float")/255 - imageB.astype("float")/255)**2 )
	return err



def train(trainLoader, model, criterion, optimizer, save_dir, max_iteration, iteration):
	model.train()

	train_start = time.time()

	cnt = 0
	loss_ave = 0

	while(True):

		for step, (images, enhance_params) in enumerate(trainLoader):

			images = images.cuda()
			inputs = Variable(images)
			
			enhance_params = enhance_params.cuda() #double -> float
			targets = Variable(enhance_params)
		
			outputs = model(inputs)
			
			#reset gradient
			optimizer.zero_grad()

			#define loss function and optimizer
			loss = criterion(outputs, targets) #criterion = MSE
			loss_print = loss.item()
			loss_ave += loss.data[0]
			loss.backward()
			
			#update parameters
			optimizer.step()
		
			cnt += 1
			batch = images.size()[0] #???


			#print( 'iteration:'+str(iteration+cnt) )
			#print( 'loss:'+str(loss.data[0]) )
			f_loss = open(save_dir+'/error.log','a')
			f_loss.write(str(iteration+cnt)+' '+str(loss_print)+'\n')
			f_loss.close()

			if cnt == max_iteration:
				loss_ave /= max_iteration

				train_time = time.time() - train_start
				print('1train time', train_time)

				return loss_ave

def validation(val_loader, model, criterion, data_dir, save_dir, iteration, resultSave=True):
	model.eval()

	val_start = time.time()

	itedir = save_dir+'/valResults/'+str(iteration)
	if not os.path.exists(itedir):
		os.mkdir(itedir)
	

	loss_val = 0
	model = model.cuda()
	#print 'def validation'
	for step, (image, enhance_param, fileName) in enumerate(val_loader):
		
		image = image.cuda()
		enhance_param = enhance_param.cuda()
		
		#inputs = Variable(image, volatile=True)
		with torch.no_grad():
			inputs = Variable(image)
		enhance_param = enhance_param.float() #double -> float
		#targets = Variable(enhance_param, volatile=True)
		with torch.no_grad():
			targets = Variable(enhance_param)

		outputs = model(inputs)

		#print('targets', targets)
		#print('outputs', outputs)
		
		loss = criterion(outputs, targets)
		loss_print = loss.item()
		#loss_val += loss.data[0]
		loss_val += loss_print
		#print('loss:'+str(loss.data[0]))
		
		loss_val_MSE = 0
		if resultSave:
			normalized_params = (outputs.data).cpu().numpy()
			denormalized_params = denormalize( normalized_params )

			param_to_photoEnhance_start = time.time()
			param_to_photoEnhance2test.param_to_photoEnhance( denormalized_params.flatten(), fileName[0], data_dir, itedir, 'test' )
			param_to_photoEnhance_end = time.time()

			#loss_im = torch.nn.MSELoss()
			target_im_name = data_dir.split('/val')[0]+'/val-target/'+fileName[0].split('_bad')[0]+'_new_nm.jpg'
			estimated_im_name =  itedir +'/'+ fileName[0].split('_bad')[0]+'_bad_cnn.jpg'

			#target_im = cv2.imread(target_im_name)
			#estimated_im = cv2.imread(estimated_im_name)
			target_img_array = np.array( Image.open(target_im_name) )
			estimated_img_array = np.array( Image.open(estimated_im_name) )
		
			loss_MSE_start = time.time()
			loss_MSE = mse_pil(target_img_array, estimated_img_array)
			loss_val_MSE += loss_MSE
			loss_MSE_end = time.time()
			#loss_MSE = loss_im(target_im, estimated_im)
			#print("fileName", fileName[0].split('_bad')[0], loss_MSE)
			print('param_to_photoEnhance', param_to_photoEnhance_end-param_to_photoEnhance_start)
			print('loss_MSE', loss_MSE_end-loss_MSE_start)
		

	loss_val /= len(val_loader)
	loss_val_MSE /= len(val_loader)

	print('average val loss:'+str(loss_val)+' '+str(loss_val_MSE))

	val_time = time.time() - val_start
	print('validation time', val_time)

	return loss_val, loss_val_MSE


def test(test_loader, model, save_dir, iteration, image_transform):
	model.eval()

	itedir = os.path.join(save_dir, 'testResults', str(iteration))
	if not os.path.exists(itedir):
		os.mkdir(itedir)

	
	model = model.cuda()

	print('def test')
	

	for step, (images, enhance_params, fileName) in enumerate(test_loader):
		
		images = images.cuda()
		
		inputs = Variable(images, volatile=True)
		outputs = model(inputs)

		#print (fileName[0]) #a0001
		#print ( outputs ) #variable data
		#print( (outputs.data).cpu().numpy() ) variable -> ndarray

		
		
		#scaled_params = (outputs.data).cpu().numpy()
		normalized_params = (outputs.data).cpu().numpy()
		#descaled_params = descaling( scaled_params )
		denormalized_params = denormalize( normalized_params )
		#save descaled params, enhanced photos
		param_to_photoEnhance2.param_to_photoEnhance( denormalized_params.flatten(), fileName[0], itedir, 'test' )
		

		#scaled params -> descaled params, save
		#descaled params -> xmp file
		#enhanced_image, save


def main(args):
	todaydetail = datetime.datetime.today()

	NUM_PARAMETERS = 21

	#input_mean = [ 0.43589211, 0.41897088, 0.36703317 ] #calculate from train dataset
	#input_std = [ 0.280604211202, 0.266653047603, 0.275418424552 ] # use "calc_MeanStd_image.py"

	input_mean = [ 0.215806275, 0.20634507, 0.18388370999999998 ] #calculate from train dataset
	input_std = [ 0.026666969999999998, 0.02542865, 0.02622691 ] # use "calc_MeanStd_image.py"

	traindir = os.path.join(args.datadir, 'train')
	valdir = os.path.join(args.datadir, 'val')
	testdir = os.path.join(args.datadir, 'test')

	image_transform = ToPILImage()
	input_transform_train = Compose([
		Scale((256,256)),
		RandomCrop(224),
		RandomHorizontalFlip(),
		ToTensor(),
		Normalize(input_mean, input_std)
	])
	input_transform_testval = Compose([
		Scale((224,224)),
		ToTensor(),
		Normalize(input_mean, input_std)
	])

	if args.model == 'AlexNet':
		model = myAlexNet( NUM_PARAMETERS )
	elif args.model == 'VGG16':
		model = myVGG( NUM_PARAMETERS )
	elif args.model == 'VGG16_bn':
		model = myVGG_bn( NUM_PARAMETERS )
	elif args.model == 'VGG19':
		model = myVGG19( NUM_PARAMETERS )
	elif args.model == 'VGG19_bn':
		model = myVGG19_bn( NUM_PARAMETERS )
	elif args.model == 'ResNet18':
		model = myResNet18( NUM_PARAMETERS )
	else:
		print('check the model name')

	#model = tempModel( NUM_PARAMETERS )
	print(model)
	model.cuda()

	if args.multiGpu == 1:
		model = torch.nn.DataParallel(model)
	
	weight = torch.ones(NUM_PARAMETERS)
	criterion = torch.nn.MSELoss().cuda()
	


	if args.mode == 'train':
		print('train...')

		dirName = todaydetail.strftime('%Y%m%d%H%M')+'_model_'+str(args.model)
		dst = args.savedir+dirName
		os.system('mkdir '+dst)
		os.system('mkdir '+dst+'/trainResults')
		os.system('mkdir '+dst+'/valResults')

		f_ec = open(dst+'/experimentsCondition.txt','a')
		f_ec.write('-------------------------------------------------------\n')
		f_ec.write(str(dirName)+'\n')
		f_args = open(dst+'/args.txt','a')

		for arg in vars(args):
			f_ec.write(str(arg)+':'+str(getattr(args, arg))+'\n')
			f_args.write(str(arg)+':'+str(getattr(args, arg))+'\n')

		f_ec.write('-------------------------------------------------------\n')
		f_ec.close()
		f_args.close()
		#lr_list = []
		if args.optim == 'SGD_momentum':
			print( 'SGD_momentum')

			if args.multiGpu:
				print( 'multi GPU' )
			else:
				print( 'single GPU ')

			if args.multiLR:
				print( 'multi LR' )

				if args.model=='VGG19_bn':
					optimizer = SGD([
						{'params': model.module.features.parameters() },
						{'params': model.module.classifier._modules['0'].parameters() },
						{'params': model.module.classifier._modules['1'].parameters() },
						{'params': model.module.classifier._modules['2'].parameters() },
						{'params': model.module.classifier._modules['3'].parameters() },
						{'params': model.module.classifier._modules['4'].parameters() },
						{'params': model.module.classifier._modules['5'].parameters() },
						{'params': model.module.classifier._modules['6'].parameters(), 'lr': args.lastLR }
					], lr=args.LR, momentum = 0.9, weight_decay = 1e-4 )

				elif args.model=='AlexNet':
					print("hogehoge")

			else:
				print( 'single LR' )
				optimizer = SGD(model.parameters(), lr=args.LR, momentum = 0.9, weight_decay = 1e-4)
				#optimizer = SGD(filter(lambda x: x.requires_grad, model.parameters()) , lr=1e-2, momentum = 0.9, weight_decay = 1e-4)

		elif args.optim == 'ADADELTA':
			print('optim = ADADELTA')
			optimizer = Adadelta(model.parameters())
			#optimizer = Adadelta(filter(lambda x: x.requires_grad, model.parameters())) #if required_grad=True, filter the parameters so that only the parameters that requires gradient are passed to the optimizer.
		else:
			print('check the optimizer name')

		if args.trainCon:
			conState = torch.load(args.conParamsPath)
			iteration = conState['iteration']
			model.load_state_dict(conState['state_dict'])
			optimizer.load_state_dict(conState['optimizer'])
		else:
			iteration = 0


		train_dataset = FlickrInteresting('flickr-parameters-train.csv', traindir, input_transform_train, mode='train')
		train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.trainBSize, shuffle=True, drop_last=True)
		val_dataset = FlickrInteresting('flickr-parameters-val.csv', valdir, input_transform_testval, mode='val')
		val_loader = DataLoader(val_dataset, num_workers=args.num_workers, batch_size=1, shuffle=False, drop_last=False)

		f_loss = open(dst+'/error.log','a')
		f_loss.write('%iteration %trainerror\n')
		f_loss.close()
		f_loss_ave = open(dst+'/ave_error.log','a')
		f_loss_ave.write('%iteration %trainError %valError %valMSE_Error\n')
		f_loss_ave.close()
		while(True):
			#if args.optim == 'SGD_momentum':
			#	adjust_learning_rate(optimizer, lr_list, iteration, args.steps_devLR)
			
			loss_ave = train(train_loader, model, criterion, optimizer, dst, args.steps_validation, iteration)
			
			iteration += args.steps_validation
			loss_val, loss_val_MSE = validation(val_loader, model, criterion, valdir, dst, iteration, resultSave=args.resultSave)
			f_loss_ave = open(dst+'/ave_error.log','a')
			f_loss_ave.write(str(iteration)+' '+str(float(loss_ave))+' '+str(loss_val)+' '+str(loss_val_MSE)+'\n')
			f_loss_ave.close()
			state ={
				'iteration': iteration,
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict(),
			}
			torch.save(state, dst+'/ite_'+str(iteration)+'_net.pth') #save model

"""
	elif args.mode == 'test':
		print('test...')

		dst = args.savedir+args.directry_date+'_model_'+args.model
		print(dst)
		savedData = dst+'/ite_'+str(args.iteration)+'_net.pth'
		
		if not os.path.exists(dst+'/testResults'):
			os.mkdir(dst+'/testResults')
		
		state = torch.load(savedData)
		if 'state_dict' in state and 'iteration' in state:
			model.load_state_dict(state['state_dict'])
			iteration = state['iteration']
			#downSize = state['downSize']
		else:
			model.load_state_dict(state)
			iteration = args.iteration
			#downSize = args.down_size

		
		test_dataset = FlickrInteresting('myphoto-test.csv', testdir, input_transform, 'test')
		test_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=1, shuffle=False, drop_last=True)
		test(test_loader, model, dst, iteration, image_transform)
"""		

if __name__ == '__main__':
	parser = ArgumentParser()

	subparsers = parser.add_subparsers(dest='mode')
	subparsers.required = True

	parser_train = subparsers.add_parser('train')
	parser_train.add_argument('--datadir', type=str, required=True)
	parser_train.add_argument('--savedir', type=str, required=True)
	parser_train.add_argument('--model', type=str, required=True)
	parser_train.add_argument('--optim', type=str, default='ADADELTA')
	#parser_train.add_argument('--pretrainedParamsPath', type=str, default='none')
	#parser_train.add_argument('--iteration', type=int, default=0)
	parser_train.add_argument('--num-workers', type=int, default=0)
	parser_train.add_argument('--trainBSize', type=int, default=16)
	#parser_train.add_argument('--steps-trainSave', type=int, default=100)
	parser_train.add_argument('--steps-validation', type=int, default=500)
	parser_train.add_argument('--multiGpu', type=int, default=0) 
	parser_train.add_argument('--trainCon', action='store_true')
	#parser_train.add_argument('--steps-devLR', type=int, default=50000)
	parser_train.add_argument('--conParamsPath', type=str, default='none')
	parser_train.add_argument('--LR', type=float, default=0.01) #1e-2
	parser_train.add_argument('--lastLR', type=float, default=0.01) #1e-2
	parser_train.add_argument('--multiLR', type=int, default=0) #if 1 -> multi Learning rate
	parser_train.add_argument('--resultSave', type=int, default=1)

	parser_val = subparsers.add_parser('val')
	#parser_val.add_argument('--dataType', type=str, default='crop', required=True)
	parser_val.add_argument('--datadir', type=str, required=True)
	parser_val.add_argument('--pretrainedParamsPath', type=str, default='none')
	parser_val.add_argument('--directry-date', required=True)
	parser_val.add_argument('--iteration', type=int, required=True)
	parser_val.add_argument('--model', type=str, required=True)
	parser_val.add_argument('--num-workers', type=int, default=4)
	parser_val.add_argument('--multiGpu', type=int, default=0) 
	"""
	parser_test = subparsers.add_parser('test')
	#parser_test.add_argument('--dataset', type=str, default='EvLab', required=True)
	parser_test.add_argument('--datadir', type=str, required=True)
	parser_test.add_argument('--savedir', type=str, required=True)
	parser_test.add_argument('--pretrainedParamsPath', type=str, default='none')
	parser_test.add_argument('--directry-date', required=True)
	parser_test.add_argument('--iteration', type=int, required=True)
	#parser_test.add_argument('--recNum', type=int, required=True)
	parser_test.add_argument('--model', type=str, required=True)
	#parser_test.add_argument('--initType', type=str, default='default')
	#parser_test.add_argument('--num-workers', type=int, default=4)
	parser_test.add_argument('--num-workers', type=int, default=0)
	#parser_test.add_argument('--down-size', type=int, default=8)
	parser_test.add_argument('--multiGpu', type=int, default=0) 
	"""
	main(parser.parse_args())


