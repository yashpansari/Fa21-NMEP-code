import numpy as np
from PIL import Image
import glob
import torch
from torch.utils.data.dataset import Dataset

'''
Pytorch uses datasets and has a very handy way of creatig dataloaders in your main.py
Make sure you read enough documentation.
'''

class Data(Dataset):
	def __init__(self, data_dir):
		#gets the data from the directory
        self.data = data_dir
        self.image_list = os.listdir(data_dir)
        #calculates the length of image_list
        self.len = len(self.image_list)

	def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_path = self.image_list[index]
        # Open image
        image = Image.open(os.path.join(self.data_dir, single_image_path))

        # Do some operations on image
        # Convert to numpy, dim = 28x28
        image_np = np.asarray(image)/255
        # Add channel dimension, dim = 1x28x28
        # Note: You do not need to do this if you are reading RGB images
        # or i there is already channel dimension

        image_np = (np.asarray(image)/255).transpose(2, 0, 1) 
        '''
		#TODO: Convert your numpy to a tensor and get the labels
		'''
        image_tensor = torch.from_numpy(image_np).float()
        class_indicator_location = single_image_path.rfind('_c')
        label = int(single_image_path[class_indicator_location+2:class_indicator_location+3])
        image90 = torch.from_numpy(np.flip(image_np.transpose(0, 2, 1), axis=0).copy()).float()
        image180 = torch.from_numpy(np.flip(np.flip(image_np, axis=0), axis=1).copy()).float()
        image270 = torch.from_numpy(np.flip(image_np, axis=0).transpose(0, 2, 1).copy()).float()
        return (image_tensor,image90,image180,image270), (label,0,1,2,3)

    def __len__(self):
        return self.data_len