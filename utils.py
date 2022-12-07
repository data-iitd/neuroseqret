import sys
import pdb
import copy
import random
import pickle
from tqdm import tqdm
import numpy as np
from multiprocessing import Process, Queue

from dtaidistance import dtw
from scipy.spatial import distance
import itertools, pdb, sys
from rank_eval import ndcg

from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

def sample_function(Query_ev, Query_ti, Train_ev, Train_ti, num_queries, num_marks, num_pos, num_pos_neg, batch_size, result_queue, SEED):
	def sample(seq_id):
		positive = seq_id*num_pos_neg + np.random.randint(0, num_pos)
		negative = seq_id*num_pos_neg + np.random.randint(num_pos+1, num_pos_neg)

		query_ev = Query_ev[seq_id]
		query_ti = Query_ti[seq_id]

		pos_corpus_ev = Train_ev[positive]
		pos_corpus_ti = Train_ti[positive]

		neg_corpus_ev = Train_ev[negative]
		neg_corpus_ti = Train_ti[negative]

		return (seq_id, query_ev, query_ti, pos_corpus_ev, pos_corpus_ti, neg_corpus_ev, neg_corpus_ti)

	np.random.seed(SEED)
	while True:
		one_batch = []
		for i in range(batch_size):
			qid = np.random.randint(0, num_queries)
			for k in range(num_pos_neg):
				one_batch.append(sample(qid))

		result_queue.put(zip(*one_batch))

class ParallelSampler(object):
	def __init__(self, Query_ev, Query_ti, Train_ev, Train_ti, num_queries, num_marks, num_pos, num_pos_neg, batch_size=64, n_workers=1):
		self.result_queue = Queue(maxsize=n_workers * 10)
		self.processors = []
		for i in range(n_workers):
			self.processors.append(
				Process(target=sample_function, args=(Query_ev, Query_ti, Train_ev, Train_ti, num_queries, num_marks, num_pos, num_pos_neg, batch_size, self.result_queue, np.random.randint(2e9))))
			self.processors[-1].daemon = True
			self.processors[-1].start()

	def next_batch(self):
		return self.result_queue.get()

	def close(self):
		for p in self.processors:
			p.terminate()
			p.join()

def to_array(query_ev, query_ti, pos_corpus_ev, pos_corpus_ti, neg_corpus_ev, neg_corpus_ti):
	query_ev, query_ti, pos_corpus_ev, pos_corpus_ti, neg_corpus_ev, neg_corpus_ti = \
	np.asarray(query_ev), np.asarray(query_ti), np.asarray(pos_corpus_ev), np.asarray(pos_corpus_ti), \
	np.asarray(neg_corpus_ev), np.asarray(neg_corpus_ti)
	return query_ev, query_ti, pos_corpus_ev, pos_corpus_ti, neg_corpus_ev, neg_corpus_ti

def to_array_dump(query_ev, query_ti, pos_corpus_ev, pos_corpus_ti):
	query_ev, query_ti, pos_corpus_ev, pos_corpus_ti = np.asarray(query_ev).reshape((1, len(query_ev))), np.asarray(query_ti).reshape((1, len(query_ti))), np.asarray(pos_corpus_ev).reshape((1, len(pos_corpus_ev))), np.asarray(pos_corpus_ti).reshape((1, len(pos_corpus_ti)))
	return query_ev, query_ti, pos_corpus_ev, pos_corpus_ti

def data_partition(fname):
	dump = pickle.load(open("Data/Config.p", "rb"))
	num_pos_neg, num_marks, num_pos = dump[fname][0], dump[fname][1], dump[fname][2]
	location = 'Data/'+str(fname)+'/'
	Query_ev = np.loadtxt(location+"train_query_ev.txt", dtype=np.float32)
	Query_ti = np.loadtxt(location+"train_query_ti.txt", dtype=np.float32)
	Train_ev = np.loadtxt(location+"train_corpus_ev.txt", dtype=np.float32)
	Train_ti = np.loadtxt(location+"train_corpus_ti.txt", dtype=np.float32)
	num_queries = Query_ev.shape[0]
	print('Data Loaded')

	return [Query_ev, Query_ti, Train_ev, Train_ti, num_queries, num_marks, num_pos, num_pos_neg]

def make_dumps(model, dataset, args):
	location = 'Data/'+args.dataset+'/'
	[Query_ev, Query_ti, Train_ev, Train_ti, num_queries, num_marks, num_pos, num_pos_neg] = dataset
	Query_ev = np.loadtxt(location+"test_query_ev.txt", dtype=np.float32)
	Query_ti = np.loadtxt(location+"test_query_ti.txt", dtype=np.float32)
	query_embs = []
	corpus_embs = []

	# Dumping Queries 
	for i in tqdm(range(len(Query_ev)), desc='Dumping Queries: '):
		query_ev, query_ti, _, _ = to_array_dump(Query_ev[i], Query_ti[i], Query_ev[i], Query_ti[i])
		last_query_feats, _, _ = model.sequence_dumps(query_ev, query_ti, query_ev, query_ti)
		query_embs.append(last_query_feats.cpu().detach().numpy())

	# Dumping Corpus
	for i in tqdm(range(len(Train_ev)), desc='Dumping Corpus: '):
		corp_ev, corp_ti, _, _ = to_array_dump(Train_ev[i], Train_ti[i], Train_ev[i], Train_ti[i])
		last_corp_feats, _, _ = model.sequence_dumps(corp_ev, corp_ti, corp_ev, corp_ti)
		corpus_embs.append(last_corp_feats.cpu().detach().numpy())

	pickle.dump([query_embs, corpus_embs], open('Hash/'+args.dataset+"_Embs.p", "wb"))

def compute_precision_acc(labels, predictions, k):
	predictions = predictions[:k]
	labels = set([i[0] for i in labels])
	precision = 0.0
	j = 1
	for i in range(len(predictions)):
		if predictions[i][0] in labels:
			precision += j/(i+1)
			j += 1

	if precision == 0.0:
		return 0

	precision = precision /(j-1)
	return precision

def cal_dtw(seq_1, seq_2):
	return dtw.distance_fast(seq_1.astype(np.double), seq_2.astype(np.double), use_pruning=True)

def normalize(seq):
	seq.sort(key=lambda x: x[1])
	min_t = seq[0][1]
	max_t = seq[-1][1]
	seq = [[int(i[0]), 1 - (i[1] - min_t)/(max_t - min_t)] for i in seq]
	return seq

def mark(seq_1, seq_2):
	same = 0
	for i in range(len(seq_1)):
		if seq_2[i] == seq_1[i]:
			same += 1
	dist = max(len(seq_1), len(seq_2)) - same
	return dist

def wass(seq_1, seq_2):
	same = 0
	for i in range(len(seq_1)):
		same += np.abs(seq_2[i] - seq_1[i])
	return same

def read_data(query_ti, test_ti, query_ev, test_ev, num_pos, num_seq, sample_neg, query_embs, corpus_embs):
	# Calculating positives and negatives
	pred_list = []
	true_list = []
	query_ti = query_ti[:-5]
	for i in range(len(query_ti)):
		min_val = i*num_seq
		max_val = (i+1)*num_seq
		temp = []
		q_pos = []
		for j in range(min_val, max_val + sample_neg):
			# Adding the true values
			if j < min_val + num_pos:
				q_pos.append([j, 1])

			# Adding the corpus values
			independent = wass(query_ti[i], test_ti[j]) + mark(query_ev[i], test_ev[j])
			based =  distance.cosine(query_embs[i], corpus_embs[j])
			dtw_based = cal_dtw(query_ti[i], test_ti[j])

			dt_dist = 0.01*independent + based + dtw_based
			temp.append([j, dt_dist])
		
		true_list.append(q_pos)
		pred_list.append(normalize(temp))

	pred_list = np.asarray(pred_list)
	true_list = np.asarray(true_list)

	av_prec = 0
	for i in range(len(query_ti)):
		av_prec += compute_precision_acc(true_list[i], pred_list[i], 10)
	map_k = av_prec/len(query_ti)
	ndcg_k = ndcg(true_list, pred_list, 10)
	print(f"MAP: {map_k}, NDCG@10: {ndcg_k}")

def evaluate(data):
	query_embs, corpus_embs = pickle.load(open('Hash/'+data+"_Embs.p", "rb"))
	dump = pickle.load(open("Data/Config.p", "rb"))
	num_seq, num_marks, num_pos = dump[data][0], dump[data][1], dump[data][2]
	file = 'Data/'+data+'/'
	query_ti = np.loadtxt(file+"test_query_ti.txt", dtype=np.float32)
	test_ti = np.loadtxt(file+"train_corpus_ti.txt", dtype=np.float32)

	query_ev = np.loadtxt(file+"test_query_ev.txt", dtype=np.float32)
	test_ev = np.loadtxt(file+"train_corpus_ev.txt", dtype=np.float32)

	sample_neg = 200
	read_data(query_ti, test_ti, query_ev, test_ev, num_pos, num_seq, sample_neg, query_embs, corpus_embs)
	print("Done for "+data)