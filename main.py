import os, 
import time
import pickle
import argparse
import pdb
import torch
from model import NeuroSeqRet
from tqdm import tqdm
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--batch_size', default=5, type=int)
parser.add_argument('--maxlen', default=20, type=int)
parser.add_argument('--hidden_units', default=16, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=20, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--eval_batch', default=10, type=int)
parser.add_argument('--pos_neg', default=40, type=int)
parser.add_argument('--device', default='cuda', type=str)

args = parser.parse_args()

dataset = data_partition(args.dataset)
[Query_ev, Query_ti, Train_ev, Train_ti, num_queries, num_marks, num_pos, num_pos_neg] = dataset

num_batch = num_queries // args.batch_size

sampler = ParallelSampler(Query_ev, Query_ti, Train_ev, Train_ti, num_queries, num_marks, \
	num_pos, num_pos_neg, batch_size=args.batch_size, n_workers=3)
model = NeuroSeqRet(num_marks, args).to(args.device)

for name, param in model.named_parameters():
	try:
		torch.nn.init.xavier_uniform_(param.data)
	except:
		pass

model.train()
epoch_start_idx = 1
bce_criterion = torch.nn.BCEWithLogitsLoss()
mse_criterion = torch.nn.MSELoss()
adam_optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98))

for epoch in range(epoch_start_idx, args.num_epochs + 1):
	for step in tqdm(range(num_batch), desc='  - Epoch: ' + str(epoch)+' ', leave=False):
		seq_id, query_ev, query_ti, pos_corpus_ev, pos_corpus_ti, neg_corpus_ev, neg_corpus_ti = sampler.next_batch()
		query_ev, query_ti, pos_corpus_ev, pos_corpus_ti, neg_corpus_ev, neg_corpus_ti = \
		to_array(query_ev, query_ti, pos_corpus_ev, pos_corpus_ti, neg_corpus_ev, neg_corpus_ti)

		adam_optimizer.zero_grad()
		dist_pos, dist_neg, query_logits, pos_logits, index_wise_pos, index_wise_neg, seq_end_pos, seq_end_neg = model(seq_id, query_ev, query_ti, pos_corpus_ev, pos_corpus_ti, neg_corpus_ev, neg_corpus_ti)
		
		wd_zero, wd_ones = torch.zeros(dist_pos.shape, device=args.device), torch.ones(dist_pos.shape, device=args.device)
		itr_zero, itr_ones = torch.zeros(index_wise_pos.shape, device=args.device), torch.ones(index_wise_pos.shape, device=args.device)
		end_zero, end_ones = torch.zeros(seq_end_pos.shape, device=args.device), torch.ones(seq_end_pos.shape, device=args.device)

		# Increase for positive and decrease for negative
		wd_loss = mse_criterion(dist_pos, 100*wd_ones) + mse_criterion(dist_neg, wd_zero)
		itr_loss = bce_criterion(index_wise_pos, itr_ones) + bce_criterion(index_wise_neg, itr_zero)
		end_loss = bce_criterion(seq_end_pos, end_ones) + bce_criterion(seq_end_neg, end_zero)

		# Additional losses for better stability
		itr_mse = mse_criterion(index_wise_pos, index_wise_neg)
		end_mse = mse_criterion(seq_end_pos, seq_end_neg)

		loss = 0.5*wd_loss + 0.7*itr_loss + 0.7*end_loss - 0.5*itr_mse - 0.5*end_mse
		for param in model.item_emb.parameters(): 
			loss += 0.0001 * torch.norm(param)

		loss.backward()
		adam_optimizer.step()

	print("Total loss after epoch {}: {}".format(epoch, loss.item()))
	if epoch % 20 == 0:
		model.eval()
		print('Making Dumps of the Parameters and Sequences', end='')
		torch.save(model.state_dict(), 'Hash/'+str(args.dataset)+'.pth')
		make_dumps(model, dataset, args)
		model.train()

sampler.close()

# Running the evaluator.
print('Running the Evaluator')
evaluate(args.dataset)