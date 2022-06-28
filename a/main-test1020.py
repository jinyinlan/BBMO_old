import torch
import os
import time
import sys
import datetime

from PIL import Image
from argparse import ArgumentParser
from collections import OrderedDict

from torch.optim import SGD, Adam, Adadelta
from torch.autograd import Variable
from torchvision.transforms import Compose, Scale, CenterCrop, Normalize, RandomHorizontalFlip, RandomCrop
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import DataLoader

from network import myAlexNet, myVGG, myVGG_bn, myVGG19, myVGG19_bn, myResNet18
from dataset1020 import FlickrInteresting
from normalize import denormalize
import param_to_photoEnhance2test

def test(test_loader, model, data_dir, save_dir, iteration):
	model.eval()

	itedir_out = os.path.join(save_dir, 'testResults', str(iteration))
	if not os.path.exists( os.path.join(save_dir, 'testResults') ):
		os.mkdir( os.path.join(save_dir, 'testResults') )
	if not os.path.exists(itedir_out):
		os.mkdir(itedir_out)

	
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
	#input_std = [ 0.026666969999999998, 0.02542865, 0.02622691 ] # use "calc_MeanStd_image.py"
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
		weight = state['state_dict']
		weight_m = OrderedDict()
		keys = weight.keys()

		#print('state')
		#print(state.keys())

		if args.multiGpu:
			print('multi GPU in train')

			for k in keys:
				weight_m[k.split('module.')[1]] = weight[k]

			if 'state_dict' in state and 'iteration' in state:
				print('with iteration')
				state_m = {
					'iteration' : state['iteration'],
					'optimizer' : state['optimizer'],
					'state_dict' : weight_m,
				}
			else:
				print('without iteration')
				state_m = {
					'iteration' : args.iteration,
					'optimizer' : state['optimizer'],
					'state_dict' : weight_m,
				}
		
			#print('state_m')
			#print(state_m['state_dict'].keys())

		else:
			print('single GPU in train')

			if 'state_dict' in state and 'iteration' in state:
				model.load_state_dict(state['state_dict'])
				iteration = state['iteration']
			else:
				model.load_state_dict(state)
				iteration = args.iteration
		
		#test_dataset = FlickrInteresting('test-data.csv', testdir, input_transform, 'test')
		#test_dataset = FlickrInteresting('flickr-test.csv', '/home/omiya/data/test-set0811/degraded', input_transform_testval, 'test')
		#test_dataset = FlickrInteresting('flickr-test.csv', '/home/omiya/data/test-set0811/test-set0820/degraded', input_transform_testval, 'test')
		#test_dataset = FlickrInteresting('flickr-val.csv', '/home/omiya/data/test-set0811/degraded-val', input_transform_testval, 'test')
		test_dataset = FlickrInteresting('flickr-1020.csv', '/home/omiya/data/test-set1020/degraded_photos/temp', input_transform_testval, 'test')
		
		test_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=1, shuffle=False, drop_last=False)
		datadir = args.datadir
		test(test_loader, model, datadir, dst, args.iteration)
		

if __name__ == '__main__':
	parser = ArgumentParser()
	

	subparsers = parser.add_subparsers(dest='mode')
	subparsers.required = True

	parser_test = subparsers.add_parser('test')
	parser_test.add_argument('--datadir', type=str, required=True)
	parser_test.add_argument('--savedir', type=str, required=True)
	parser_test.add_argument('--directry-date', required=True)
	parser_test.add_argument('--iteration', type=int, required=True)
	parser_test.add_argument('--model', type=str, required=True)
	parser_test.add_argument('--num-workers', type=int, default=0)
	parser_test.add_argument('--multiGpu', type=int, required=True) 

	main(parser.parse_args())


