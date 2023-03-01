import os
import shutil
import numpy as np

import torch
from torch import nn

from torchvision import transforms, datasets
from torchvision.models import resnet50, vgg16, inception_v3

from tqdm.auto import tqdm

from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'


dims = (3, 224, 224)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
								 std=[0.229, 0.224, 0.225])

class DefectDataset(datasets.ImageFolder):
	def __init__(self, root, dims=None, return_tensors=True,
				 normalize_transform=None, **kwargs):
		self.root = root
				
		t_list = []
		if dims:
			if len(dims) == 3 and dims[0] == 1:
				t_list.append(transforms.Grayscale())
			t_list.append(transforms.Resize(dims[1:]))
		
		if return_tensors:
			t_list.append(transforms.ToTensor()) 
		
		if normalize_transform:
			t_list.append(normalize_transform)
				
		self.transforms = transforms.Compose(t_list)
		super().__init__(root, transform=self.transforms)

def get_resnet50_embeddings(dataset, tqdm_suppress = False):
	model = resnet50(pretrained=True).to(device)
	model.eval()

	extraction_layer = model._modules.get('avgpool')

	embeddings = get_embeddings(dataset, model, extraction_layer, tqdm_suppress)
	return embeddings

def get_vgg16_embeddings(dataset, tqdm_suppress = False):
	model = vgg16(pretrained=True).to(device)
	model.eval()

	extraction_layer = model._modules.get('avgpool')

	embeddings = get_embeddings(dataset, model, extraction_layer, tqdm_suppress)
	return embeddings

def get_inception_embeddings(dataset, tqdm_suppress = False):
	model = inception_v3(pretrained=True).to(device)
	model.eval()

	extraction_layer = model._modules.get('avgpool')
	
	embeddings = get_embeddings(dataset, model, extraction_layer, tqdm_suppress)
	return embeddings

def get_embeddings(dataset, model, extraction_layer, tqdm_suppress):
	reprs = []
	
	def copy_data(m, i, o):
		reprs.append(o.data.flatten())

	h = extraction_layer.register_forward_hook(copy_data)
	with torch.no_grad():
		for image, _ in tqdm(dataset, disable = tqdm_suppress):
			h_x = model(image[None, :].to(device))
	h.remove()

	return torch.vstack(reprs).cpu().numpy()
	