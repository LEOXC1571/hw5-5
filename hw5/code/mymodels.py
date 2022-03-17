import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####


class MyMLP(nn.Module):
	def __init__(self, input_dim=178, hidden_dim=16, output_dim=5):
		super(MyMLP, self).__init__()
		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):
		sigmoid = nn.Sigmoid()
		x = sigmoid(self.fc1(x))
		x = sigmoid(self.fc2(x))
		x = sigmoid(self.fc3(x))
		return x


class MyCNN(nn.Module):
	def __init__(self):
		super(MyCNN, self).__init__()

	def forward(self, x):
		return x


class MyRNN(nn.Module):
	def __init__(self):
		super(MyRNN, self).__init__()

	def forward(self, x):
		return x


class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()
		# You may use the input argument 'dim_input', which is basically the number of features

	def forward(self, input_tuple):
		# HINT: Following two methods might be useful
		# 'pack_padded_sequence' and 'pad_packed_sequence' from torch.nn.utils.rnn

		seqs, lengths = input_tuple

		return seqs