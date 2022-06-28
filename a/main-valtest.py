import torch
import os
import time
import sys
import datetime

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
import param_to_photoEnhance2test

def test(test_loader, model, data_dir, save_dir, iteration):
	model.eval()

	itedir_out = os.path.join(save_dir, 'valResults', str(iteration))
	if not os.path.exists( os.path.join(save_dir, 'valResults') ):
		os.mkdir( os.path.join(save_dir, 'valResults') )
	if not os.path.exists(itedir_out):
		os.mkdir(itedir_out)

	
	model = model.cuda()

	print('def valtest')
	

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
		print('nyaan')
		param_to_photoEnhance2test.param_to_photoEnhance( denormalized_params.flatten(), fileName[0], data_dir, itedir_out, 'test' )
		

		#scaled params -> descaled params, save
		#descaled params -> xmp file
		#enhanced_image, save

def adjust_learning_rate(optimizer, lr_list, iteration, devStep):
	#Sets the learning rate to the initial LR divided by 10 every specified iterations
	for i, param_group in enumerate(optimizer.param_groups):
		param_group['lr'] = lr_list[i] * (0.1 ** (iteration // devStep))
		print(i, param_group['lr'])

def main(args):
	todaydetail = datetime.datetime.today()

	NUM_PARAMETERS = 21

	input_mean = [ 0.215806275, 0.20634507, 0.18388370999999998 ] #calculate from train dataset
	input_std = [ 0.26666969999999998, 0.2542865, 0.2622691 ] # use "calc_MeanStd_image.py"

	traindir = os.path.join(args.datadir, 'train')
	valdir = os.path.join(args.datadir, 'val')
	testdir = os.path.join(args.datadir, 'test')

	image_transform = ToPILImage()
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
	


	if args.mode == 'test':
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

		
		#test_dataset = FlickrInteresting('test-data.csv', testdir, input_transform, 'test')
		#test_dataset = FlickrInteresting('flickr-parameters_2017-07-20_2017-09-30-val.csv', valdir, input_transform_testval, 'test')
		test_dataset = FlickrInteresting('flickr-parameters-1208-val.csv', valdir, input_transform_testval, 'test')
		test_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=1, shuffle=False, drop_last=False)
		test(test_loader, model, valdir, dst, iteration)
		

if __name__ == '__main__':
	parser = ArgumentParser()
	

	subparsers = parser.add_subparsers(dest='mode')
	subparsers.required = True

	parser_train = subparsers.add_parser('train')
	parser_train.add_argument('--datadir', type=str, required=True)
	parser_train.add_argument('--savedir', type=str, required=True)
	#parser_train.add_argument('--dataset', type=str, default='EvLab', required=True)
	#parser_train.add_argument('--recNum', type=int)
	parser_train.add_argument('--model', type=str, required=True)
	parser_train.add_argument('--optim', type=str, default='ADADELTA')
	parser_train.add_argument('--pretrainedParamsPath', type=str, default='none')
	parser_train.add_argument('--iteration', type=int, default=0)
	#parser_train.add_argument('--initType', type=str, default='default')
	parser_train.add_argument('--num-workers', type=int, default=0)
	#parser_train.add_argument('--trainBSize', type=int, default=10)
	parser_train.add_argument('--trainBSize', type=int, default=16)
	#parser_train.add_argument('--pSize', type=int, default=256)
	parser_train.add_argument('--steps-trainSave', type=int, default=100)
	parser_train.add_argument('--steps-validation', type=int, default=500)
	#parser_train.add_argument('--down-size', type=int, default=8)
	"""
	parser_train.add_argument('--DABright', type=int, default=0)
	parser_train.add_argument('--DASharp', type=int, default=0)
	parser_train.add_argument('--DAColor', type=int, default=0)
	parser_train.add_argument('--DACont', type=int, default=0)
	parser_train.add_argument('--DAGauss', type=int, default=0)
	parser_train.add_argument('--scaleMin', type=float, default=0.9)
	parser_train.add_argument('--scaleMax', type=float, default=1.1)
	parser_train.add_argument('--scaleMode', type=str, default='normal')
	"""
	parser_train.add_argument('--multiGpu', type=int, default=0) 
	parser_train.add_argument('--trainCon', action='store_true')
	parser_train.add_argument('--steps-devLR', type=int, default=50000)
	parser_train.add_argument('--conParamsPath', type=str, default='none')
	parser_train.add_argument('--encoderLR', type=float, default=0.001)
	parser_train.add_argument('--recurrentLR', type=float, default=0.1)
	parser_train.add_argument('--decoderLR', type=float, default=0.1)
	parser_train.add_argument('--LR', type=float, default=0.01) #1e-2

	parser_val = subparsers.add_parser('val')
	#parser_val.add_argument('--dataset', type=str, default='EvLab', required=True)
	parser_val.add_argument('--dataType', type=str, default='crop', required=True)
	parser_val.add_argument('--datadir', type=str, required=True)
	parser_val.add_argument('--pretrainedParamsPath', type=str, default='none')
	parser_val.add_argument('--directry-date', required=True)
	parser_val.add_argument('--iteration', type=int, required=True)
	#parser_val.add_argument('--recNum', type=int, required=True)
	parser_val.add_argument('--model', type=str, required=True)
	#parser_val.add_argument('--initType', type=str, default='default')
	parser_val.add_argument('--num-workers', type=int, default=4)
	#parser_val.add_argument('--down-size', type=int, default=8)
	parser_val.add_argument('--multiGpu', type=int, default=0) 

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

	main(parser.parse_args())


