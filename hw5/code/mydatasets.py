import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####

def load_seizure_dataset(path, model_type):
	"""
	:param path: a path to the seizure data CSV file
	:return dataset: a TensorDataset consists of a data Tensor and a target Tensor
	"""
	# TODO: Read a csv file from path.
	# TODO: Please refer to the header of the file to locate X and y.
	# TODO: y in the raw data is ranging from 1 to 5. Change it to be from 0 to 4.
	# TODO: Remove the header of CSV file of course.
	# TODO: Do Not change the order of rows.
	# TODO: You can use Pandas if you want to.
	raw_data = pd.read_csv(path, header=0)
	raw_data['y'] = raw_data['y'] - 1

	if model_type == 'MLP':
		data = torch.tensor(np.array(raw_data.drop(['y'], axis=1)))
		data = data.type(torch.FloatTensor)
		target = torch.tensor(raw_data['y'])
		dataset = TensorDataset(data, target)
	elif model_type == 'CNN':
		x_data = np.array(raw_data.drop(['y'], axis=1))
		x_data = np.expand_dims(x_data[:, 0:178].astype(float), axis=2)
		data = torch.tensor(x_data)
		data = data.permute(0, 2, 1)
		data = data.type(torch.FloatTensor)
		target = torch.tensor(raw_data['y'])
		dataset = TensorDataset(data, target)
	elif model_type == 'RNN':
		x_data = np.array(raw_data.drop(['y'], axis=1))
		x_data = np.expand_dims(x_data[:, 0:178].astype(float), axis=2)
		data = torch.tensor(x_data)
		data = data.permute(0, 2, 1)
		data = data.type(torch.FloatTensor)
		target = torch.tensor(raw_data['y'])
		dataset = TensorDataset(data, target)
	else:
		raise AssertionError("Wrong Model Type!")

	return dataset


def calculate_num_features(seqs):
	"""
	:param seqs:
	:return: the calculated number of features
	"""
	# TODO: Calculate the number of features (diagnoses codes in the train set)
	temp = []
	for i in range(len(seqs)):
		for j in range(len(seqs[i])):
			temp.extend(seqs[i][j])
	drop_dup = list(set(temp))
	return len(drop_dup)


class VisitSequenceWithLabelDataset(Dataset):
	def __init__(self, seqs, labels, num_features):
		"""
		Args:
			seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
			labels (list): list of labels (int)
			num_features (int): number of total features available
		"""

		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")

		self.labels = labels

		# TODO: Complete this constructor to make self.seqs as a List of which each element represent visits of a patient
		# TODO: by Numpy matrix where i-th row represents i-th visit and j-th column represent the feature ID j.
		# TODO: You can use Sparse matrix type for memory efficiency if you want.
		self.seqs = []
		for i in range(len(labels)):
			temp_x = []
			temp_y = []
			for j in range(len(seqs[i])):
				for k in range(len(seqs[i][j])):
					temp_x.append(j)
					temp_y.append(seqs[i][j][k])
			values = [1 for _ in range(len(temp_x))]
			mat = sparse.coo_matrix((values, (temp_x, temp_y)),shape=(len(seqs[i]), num_features))
			mat = mat.toarray()
			self.seqs.append(mat) # replace this with your implementation.

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		# returns will be wrapped as List of Tensor(s) by DataLoader
		return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
	"""
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq_1, label_1), (seq_2, label_2), ... , (seq_N, label_N)]
	where N is minibatch size, seq_i is a Numpy (or Scipy Sparse) array, and label is an int value

	:returns
		seqs (FloatTensor) - 3D of batch_size X max_length X num_features
		lengths (LongTensor) - 1D of batch_size
		labels (LongTensor) - 1D of batch_size
	"""

	# TODO: Return the following two things
	# TODO: 1. a tuple of (Tensor contains the sequence data , Tensor contains the length of each sequence),
	# TODO: 2. Tensor contains the label of each sequence

	x = []
	y = []
	temp = []
	for i in batch:
		x.append(len(i[0]))
		y.append(len(i[0]))
	x.sort(reverse=True)
	for j in x:
		temp.append(batch[y.index(j)])
		y[y.index(j)] = []

	seq = []
	lengths = []
	labels = []
	max_length = 0

	for i in range(len(temp)):
		seq.append(temp[i][0])
		lengths.append(len(temp[i][0]))
		labels.append(temp[i][1])
		if temp[i][0].shape[0] > max_length:
			max_length = temp[i][0].shape[0]

	seqs = []
	for j in seq:
		if j.shape[0] < max_length:
			zeros = np.zeros((max_length-j.shape[0], j.shape[1]))
			fillzero = np.append(j, zeros, axis=0)
			seqs.append(fillzero)
		else:
			seqs.append(j)
	seqs_ls = []
	for i in range(len(seqs)):
		seqi = seqs[i].tolist()
		seqs_ls.append(seqi)

	# temp = batch.sort(key=lambda x: x[0].shape[0], reversed=True)
	seqs_tensor = torch.FloatTensor(seqs_ls)
	lengths_tensor = torch.LongTensor(lengths)
	labels_tensor = torch.LongTensor(labels)

	return (seqs_tensor, lengths_tensor), labels_tensor