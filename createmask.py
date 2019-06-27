from __future__ import print_function
import argparse
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from os import listdir
from os.path import join

from moviepy.editor import *

model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
people_class = 15

model.eval()
print ("Model Loaded")

blur = torch.FloatTensor([[[[1.0, 2.0, 1.0],[2.0, 4.0, 2.0],[1.0, 2.0, 1.0]]]]) / 16.0

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
	model.to('cuda')
	blur = blur.to('cuda')
	
import urllib
from torchvision import transforms

preprocess = transforms.Compose([
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def makeSegMask(img):
	frame_data = torch.FloatTensor( img ) / 255.0

	input_tensor = preprocess(frame_data.permute(2, 0, 1))
	input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

	# move the input and model to GPU for speed if available
	if torch.cuda.is_available():
		input_batch = input_batch.to('cuda')


	with torch.no_grad():
		output = model(input_batch)['out'][0]

	segmentation = output.argmax(0)

	bgOut = output[0:1][:][:]
	a = (1.0 - F.relu(torch.tanh(bgOut * 0.30 - 1.0))).pow(0.5) * 2.0

	people = segmentation.eq( torch.ones_like(segmentation).long().fill_(people_class) ).float()

	people.unsqueeze_(0).unsqueeze_(0)
	
	for i in range(3):
		people = F.conv2d(people, blur, stride=1, padding=1)

	# combined_mask = F.hardtanh(a * b)
	combined_mask = F.relu(F.hardtanh(a * (people.squeeze().pow(1.5)) ))
	combined_mask = combined_mask.expand(1, 3, -1, -1)

	res = (combined_mask * 255.0).cpu().squeeze().byte().permute(1, 2, 0).numpy()

	return res

def processMovie(args):
	print("Processing {}... This will take some time.".format(args.input))

	if args.width != 0:
		target=[args.width, None]
	else:
		target=None

	realityClip = VideoFileClip(args.input, target_resolution=target)

	realityMask = realityClip.fl_image(makeSegMask)
	realityMask.write_videofile(args.output)

def main():
	parser = argparse.ArgumentParser(description='BGRemove')
	parser.add_argument('--input', metavar='N', required=True,
						help='input movie path')	
	parser.add_argument('--output', metavar='N', required=True,
						help='output movie path')							
	parser.add_argument('--width', metavar='N', type=int, default=0,
						help='target width (optional, omit for full width)')													

	args = parser.parse_args()

	processMovie(args)

if __name__ == '__main__':
	main()
