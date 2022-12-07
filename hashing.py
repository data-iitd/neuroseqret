import argparse
import random
import time
import pickle
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pdb
from collections import defaultdict

class HashCodeGenerator(nn.Module):
	def __init__(self, args, num_corp):
		super(HashCodeGenerator, self).__init__()
		self.dev = args.device 
		self.num_seq = num_corp
		self.corp_self = nn.Embedding(self.num_seq, args.hidden_dim)
		self.corp_query = nn.Embedding(self.num_seq, args.hidden_dim)

		# self.non_nbr_mat = cudavar(self.av,torch.zeros(self.gr.get_num_nodes(),self.gr.get_num_nodes()))
		self.hash_linear1 = nn.Linear(args.hidden_dim, args.hash_dim)
		self.hash_tanh1 = nn.Tanh()
		nn.init.normal_(self.hash_linear1.weight)

	def init_embeddings(self, corpus_embs, corp_query):
		self.corp_self.weight = nn.Parameter(torch.from_numpy(corpus_embs).to(self.dev), requires_grad=False)
		self.corp_query.weight = nn.Parameter(torch.from_numpy(corp_query).to(self.dev), requires_grad=False)
	
	def init_non_nbr_mat(self,list_training_edges):
		for (a,b) in list_training_edges:
			self.non_nbr_mat[a][b] = 1
		z = cudavar(self.av,torch.zeros(self.gr.get_num_nodes(),self.gr.get_num_nodes()))
		o = cudavar(self.av,torch.ones(self.gr.get_num_nodes(),self.gr.get_num_nodes()))
		reverse = torch.where(self.non_nbr_mat==0,o,z)
		self.non_nbr_mat = reverse

	def forward(self, corp_batch):
		cs_embs = self.corp_self(torch.LongTensor(corp_batch).to(self.dev))
		cq_embs = self.corp_query(torch.LongTensor(corp_batch).to(self.dev))
		cs_hashcodes = self.hash_tanh1(self.hash_linear1(cs_embs))
		return cs_hashcodes, cq_embs

	def get_hash(self, seq_emb):
		hashcodes = self.hash_tanh1(self.hash_linear1(torch.from_numpy(seq_emb).to(self.dev)))
		return hashcodes

	def computeLoss(self, corp_batch):
		loss1 = loss2 = loss3 = 0
		cs_hashcodes, cq_embs = self.forward(corp_batch)
		num_nodes = len(corp_batch)
		for i in range(num_nodes):
			selfcode = cs_hashcodes[i]
			cq_code = cq_embs[i]
			loss1 = loss1 + torch.abs(torch.sum(selfcode))
			loss2 = loss2 + torch.norm(torch.abs(selfcode)-1,p=1)
			loss3 += torch.square(torch.norm(selfcode - cq_code))

		'''Implemntation of other losses'''
		# indices = cudavar(av,torch.tensor(nodes))
		# non_nbrs = torch.index_select(torch.index_select(self.non_nbr_mat,0,indices),1,indices)
		# similarity_mat = torch.mul(torch.abs(torch.mm(all_hashcodes,torch.transpose(all_hashcodes,0,1))),non_nbrs)
		# loss3 = torch.sum(similarity_mat) - torch.sum(torch.diagonal(similarity_mat))
		return loss1, loss2, loss3, num_nodes

def train_hash_codes(args):
	[query_embs, corpus_embs] = pickle.load(open("Hash/"+args.dataset+"_Embs.p", "rb"))
	cq_embs = corpus_embs
	query_embs, corpus_embs, cq_embs = order_np_array(query_embs), order_np_array(corpus_embs), order_np_array(cq_embs)
	
	num_corp = len(corpus_embs)
	corpus = list(range(num_corp))

	model = HashCodeGenerator(args, num_corp).to(args.device)
	model.init_embeddings(corpus_embs, cq_embs)
	sgd_opt = torch.optim.SGD(model.parameters(), lr=0.005)
	
	epoch_start_idx = 1
	for epoch in range(epoch_start_idx, args.num_epochs + 1):
		for i in range(0, num_corp, args.batch):
			corp_batch = corpus[i:i+args.batch]
			model.zero_grad()
			loss1, loss2, loss3, num_nodes = model.computeLoss(corp_batch)
			# loss = (args.const_1/num_nodes)*loss1 + (args.const_2/num_nodes)*loss2 + ((1-(args.const_1+args.const_2))/(num_nodes**2))*loss3
			loss = (args.const_1/num_nodes)*loss1 + (args.const_2/num_nodes)*loss2 + (args.const_3/num_nodes)*loss3
			print(loss)
			loss.backward()
			sgd_opt.step()
		epoch += 1

	query_codes = []
	corpus_codes = []
	# Making hash codes for queries and corpus
	for k in range(len(query_embs)):
		hash_emb = np.sign(model.get_hash(query_embs[k]).detach().cpu().numpy())
		query_codes.append(hash_emb)

	for k in range(len(corpus_embs)):
		hash_emb = np.sign(model.get_hash(corpus_embs[k]).detach().cpu().numpy())
		corpus_codes.append(hash_emb)

	pickle.dump([query_codes, corpus_codes], (open("Hash/"+args.dataset+"_Codes.p", "wb")))

def assign_bucket(hash_code, max_ind):
	binary_maps = 1 << np.arange(max_ind - 1, -1, -1)
	bucket = sum(binary_maps * hash_code)
	return bucket

def bucketify(query_codes, corpus_codes, args, num_pos_neg):
	all_hash_tables = []
	for func_id in range(args.tables):
		hash_table = {}
		for id in range(2**args.subset):
			hash_table[id] = []
		for node in range(num_pos_neg):
			hash_table[self.assign_bucket(func_id, self.hashcode_mat[node])].append(node)
		all_hash_tables.append(hash_table)

def order_np_array(arr):
	arr = np.asarray(arr)
	return arr.reshape(arr.shape[0], arr.shape[2])

def minus_10(arr):
	return ((arr + 1)/2).astype(int)

def return_unique(seq):
	return np.unique(seq)

def normalize(seq):
	if len(seq) <= 1:
		return seq
	seq.sort(key=lambda x: x[1])
	new_seq = []
	min_t = seq[0][1]
	if min_t == 0.0:
		min_t = 0.001
	max_t = seq[-1][1]
	for i in seq:
		if i[1] == 0.0:
			new_seq.append([int(i[0]), 0])
		else:
			new_seq.append([int(i[0]), 1.01 - (i[1] - min_t)/(max_t - min_t)])

	return new_seq

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', required=True)
	parser.add_argument('--hidden_dim', default=16, type=int)
	parser.add_argument('--hash_dim', default=16, type=int)
	parser.add_argument('--batch', default=200, type=int)
	parser.add_argument('--const_1', default=0.1, type=float)
	parser.add_argument('--const_2', default=0.1, type=float)
	parser.add_argument('--const_3', default=0.1, type=float)
	parser.add_argument('--num_epochs', default=20, type=int)
	parser.add_argument('--tables', default=10, type=int)
	parser.add_argument('--subset', default=0, type=int)
	parser.add_argument('--device', default='cpu', type=str)
	args = parser.parse_args()

	train_hash_codes(args)

	dump = pickle.load(open("Data/Config.p", "rb"))
	num_pos_neg, num_marks, num_pos = dump[args.dataset][0], dump[args.dataset][1], dump[args.dataset][2]
	[query_codes, corpus_codes] = pickle.load(open("Hash/"+args.dataset+"_Codes.p", "rb"))
	hash_tables = []
	c4q = [[] for i in range(len(query_codes))]

	for i in range(args.tables):
		temp = defaultdict(list)
		indices = np.arange(args.hash_dim)
		np.random.shuffle(indices)
		indices = indices[:args.subset]
		for j in range(len(corpus_codes)):
			corp_hash = minus_10(corpus_codes[j][indices])
			bucket = assign_bucket(corp_hash, args.subset)
			temp[bucket].append(j)

		hash_tables.append(temp)

		for k in range(len(query_codes)):
			min_val = k*num_pos_neg
			true_temp = []
			pred_temp = []
			max_val = (k+1)*num_pos_neg
			q_hash = minus_10(query_codes[k][indices])
			bucket = assign_bucket(q_hash, args.subset)

			for l in temp[bucket]:
				if l > min_val and l < max_val:
					c4q[k].append(l)

	pickle.dump(c4q, (open("Hash/"+args.dataset+"_Hashed.p", "wb")))