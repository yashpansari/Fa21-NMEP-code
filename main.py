import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from data_pytorch import Data
from rotnet import RotNet
import time
import shutil
import yaml

parser = argparse.ArgumentParser(description='Configuration details for training/testing rotation net')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--train', action='store_true')
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--image', type=str)
parser.add_argument('--model_number', type=str, required=True)

args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    total = 0
    for i, (input, target) in enumerate(train_loader):
    	x, x90, x180, x270 = input
        label, y0, y90, y180, y270 = target
        optimizer.zero_grad()
		pred = model(torch.vstack([x, x90, x180, x270]))
		loss = criterion(pred, torch.hstack([y0, y90, y180, y270]).T)
		loss.backward()
		optimizer.step()
		total += loss
	print("Epoch {0}: {1}".format(epoch, total)) 

def validate(val_loader, model, criterion):
	model.eval()
	test_accuracy = 0
    for i, (input, target) in enumerate(train_loader):
    	x, x90, x180, x270 = input
        label, y0, y90, y180, y270 = target
        optimizer.zero_grad()
		pred = model(torch.vstack([x, x90, x180, x270]))
		test_accuracy = torch.sum(test_accuracy,(torch.sum(torch.hstack([y0, y90, y180, y270]).T == torch.argmax(pred, dim=1), dtype=torch.double) / 4))
		print ("Test Accuracy {0}".format(test_accuracy))
	return test_accuracy/i
		

def save_checkpoint(state, best_one, filename='rotationnetcheckpoint.pth.tar', filename2='rotationnetmodelbest.pth.tar'):
	torch.save(state, filename)
	#best_one stores whether your current checkpoint is better than the previous checkpoint
    if best_one:
        shutil.copyfile(filename, filename2)

def main():
	n_epochs = config["num_epochs"]
	model = RotNet(block=BasicBlock, num_blocks=[2, 2, 2, 2], rot_classes=4, inference_classes=10)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

	train_dataset = Data(os.path.join(data_dir, "train"))
	train_data, val_data = torch.utils.data.random_split(train_set, lengths=[round(len(train_set) * 0.8), len(train_set) - round(len(train_set) * 0.8)])
	train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
	val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
	val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=True)
	test_dataset = Data(os.path.join(data_dir, "test"))
	test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)
	
	max_acc = 0
	for epoch in range(n_epochs):
	 	#TODO: make your loop which trains and validates. Use the train() func
		loss = train(train_loader, model, criterion, optimizer, epoch)
		acc = validate(val_loader, model, rot_or_class)
		print("Validation Accuracy {0}".format(acc))

		if acc > max_acc:
			max_acc = acc
			train(train_loader(), model, criterion, optimizer, epoch)
		acc = validate(val_loader(), model, criterion)
	 	#TODO: Save your checkpoint
		best_one = False
		if acc>max_acc:
			max_acc = acc
			best_one = True
			print("Best one yet")
		save_checkpoint({'epoch': epoch + 1,'max_acc': best_acc,'state_dict': model.state_dict()}, best_one, filename=os.path.join(checkpoint_dir, rot_or_class + 'checkpoint.pth.tar'), filename2=os.path.join(checkpoint_dir, rot_or_class + 'modelbest.pth.tar'))
	test_acc = validate(test_loader, model)
	print("Final Test Accuracy:", test_acc)



if __name__ == "__main__":
    main()
