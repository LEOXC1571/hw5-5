import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####


class MyMLP(nn.Module):
	def __init__(self, input_dim=178, hidden_dim=128, output_dim=5):
		super(MyMLP, self).__init__()
		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, 64)
		self.fc4 = nn.Linear(64, output_dim)

	def forward(self, x):
		sigmoid = nn.Sigmoid()
		x = sigmoid(self.fc1(x))
		x = sigmoid(self.fc2(x))
		x = sigmoid(self.fc3(x))
		x = sigmoid(self.fc4(x))
		return x


class MyCNN(nn.Module):
	def __init__(self):
		super(MyCNN, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv1d( # input tensor: [16, 1, 178]
				in_channels=1, #in_channels = 1
				out_channels=6,
				kernel_size=5),
			nn.ReLU(), # output tensor: [16, 6, 174]
			nn.MaxPool1d(kernel_size=2) #output tensor: [16, 6, 87]
		)
		# print(self.conv1.shape)
		self.conv2 = nn.Sequential(
			nn.Conv1d( # input tensor: [16, 6, 87]
				in_channels=6,
				out_channels=16,
				kernel_size=5),
			nn.ReLU(), #output tensor: [16, 16, 83]
			nn.MaxPool1d(kernel_size=2) #output tensor: [16, 16, 41]
		)
		self.fc1 = nn.Linear(656, 128)
		self.fc2 = nn.Linear(128, 128)
		self.fc3 = nn.Linear(128, 5)

	def forward(self, x):
		relu = nn.ReLU()
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.shape[0], 16*41) # transform x [16, 16, 41] into [16, 656]
		x = relu(self.fc1(x))
		x = relu(self.fc2(x))
		x = self.fc3(x)
		return x


class MyRNN(nn.Module):
	def __init__(self, input_size=178, hidden_units=16, num_layers=1):
		super(MyRNN, self).__init__()
		self.gru = nn.GRU(
			input_size=input_size,
			hidden_size=hidden_units,
			num_layers=num_layers,
			batch_first=True)
		self.fc1 = nn.Linear(16, 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, 5)
	def forward(self, x):
		sigmoid = nn.Sigmoid()
		x = x.permute(0, 2, 1)
		x, _ = self.gru(x)
		x = x.view(x.shape[0], 1 * 16)
		x = sigmoid(self.fc1(x))
		x = sigmoid(self.fc2(x))
		x = sigmoid(self.fc3(x))
		return x


class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()
		self.fc1 = nn.Linear(dim_input, 32)
		self.gru = nn.GRU(
			input_size=32,
			hidden_size=16,
			num_layers=1,
			batch_first=True
		)
		self.fc2 = nn.Linear(16, 2) # 16 hidden size and binary classification
		# You may use the input argument 'dim_input', which is basically the number of features

	def forward(self, input_tuple):
		# HINT: Following two methods might be useful
		seqs, lengths = input_tuple
		lengths = lengths.cpu()
		tanh = nn.Tanh()
		x = tanh(self.fc1(seqs))
		pack = pack_padded_sequence(x, lengths, batch_first=True)
		out, _ = self.gru(pack)
		# y = self.fc2(out.data)
		unpacked, _ = pad_packed_sequence(out, batch_first=True)
		#x = unpacked.reshape(unpacked.shape[0], unpacked.shape[1]*unpacked.shape[2])
		x = self.fc2(unpacked)
		return torch.sum(x, dim=1)