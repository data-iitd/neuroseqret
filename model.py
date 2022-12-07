import numpy as np
import torch
import sys
import pdb
from emb_dist import Emd_Dist
from UMNN import MonotonicNN

class PFFN(torch.nn.Module):
	def __init__(self, hidden_units, dropout_rate):
		super(PFFN, self).__init__()
		self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
		self.dropout1 = torch.nn.Dropout(p=dropout_rate)
		self.relu = torch.nn.ReLU()
		self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
		self.dropout2 = torch.nn.Dropout(p=dropout_rate)

	def forward(self, inputs):
		outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
		outputs = outputs.transpose(-1, -2)
		outputs += inputs
		return outputs

class NeuroSeqRet(torch.nn.Module):
	# Architecture of the proposed model
	def __init__(self, item_num, args):
		super(NeuroSeqRet, self).__init__()

		self.item_num = item_num
		self.dev = args.device
		self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
		self.time_embeddings = torch.nn.Linear(1, args.hidden_units)
		self.mean_nn = torch.nn.Linear(2*args.hidden_units, args.hidden_units)
		self.var_nn = torch.nn.Linear(2*args.hidden_units, args.hidden_units)
		self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
		self.item_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
		self.to_positive = torch.nn.ReLU()
		self.sigmoid = torch.nn.Sigmoid()

		# For index-wise and sequence-end comparisons
		self.index_wise = torch.nn.Linear(args.hidden_units, 1)
		self.seq_end = torch.nn.Linear(args.hidden_units, 1)

		self.attention_layernorms = torch.nn.ModuleList()
		self.attention_layers = torch.nn.ModuleList()
		self.forward_layernorms = torch.nn.ModuleList()
		self.forward_layers = torch.nn.ModuleList()

		self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

		# Stacking the attention blocks
		for _ in range(args.num_blocks):
			new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
			self.attention_layernorms.append(new_attn_layernorm)

			new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate)
			self.attention_layers.append(new_attn_layer)

			new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
			self.forward_layernorms.append(new_fwd_layernorm)

			new_fwd_layer = PFFN(args.hidden_units, args.dropout_rate)
			self.forward_layers.append(new_fwd_layer)

	def get_event_embeddings(self, seq_ev, seq_ti):
		event_embs = self.item_emb(torch.LongTensor(seq_ev).to(self.dev))
		time_query = torch.Tensor(seq_ti).to(self.dev)
		time_query = torch.reshape(time_query, (time_query.shape[0], time_query.shape[1], 1))
		time_embs_query = self.time_embeddings(time_query)
		event_embs += time_embs_query
		event_embs *= self.item_emb.embedding_dim ** 0.5
		return event_embs

	def seq2dumps(self, query_ev, query_ti, corpus_ev, corpus_ti):
		positions = np.tile(np.array(range(query_ev.shape[1])), [query_ev.shape[0], 1])
		positions = self.pos_emb(torch.LongTensor(positions).to(self.dev))

		query_seqs = self.get_event_embeddings(query_ev, query_ti)
		query_seqs += positions

		corp_seqs = self.get_event_embeddings(corpus_ev, corpus_ti)
		corp_seqs += positions

		tl = query_seqs.shape[1]
		attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

		# Embeddings for Query
		inp_seqs = query_seqs
		for i in range(len(self.attention_layers)):
			inp_seqs = torch.transpose(inp_seqs, 0, 1)
			Q = self.attention_layernorms[i](inp_seqs)
			mha_outputs, _ = self.attention_layers[i](Q, inp_seqs, inp_seqs, attn_mask=attention_mask)
			inp_seqs = Q + mha_outputs
			inp_seqs = torch.transpose(inp_seqs, 0, 1)

			inp_seqs = self.forward_layernorms[i](inp_seqs)
			inp_seqs = self.forward_layers[i](inp_seqs)

		query_log_feats = self.last_layernorm(inp_seqs)
		last_query_feats = query_log_feats[:, -1]

		# Embeddings for Corpus
		inp_seqs = corp_seqs
		for i in range(len(self.attention_layers)):
			inp_seqs = torch.transpose(inp_seqs, 0, 1)
			Q = self.attention_layernorms[i](inp_seqs)
			mha_outputs, _ = self.attention_layers[i](Q, inp_seqs, inp_seqs, attn_mask=attention_mask)
			inp_seqs = Q + mha_outputs
			inp_seqs = torch.transpose(inp_seqs, 0, 1)

			inp_seqs = self.forward_layernorms[i](inp_seqs)
			inp_seqs = self.forward_layers[i](inp_seqs)

		corp_log_feats = self.last_layernorm(inp_seqs)
		last_corp_feats = corp_log_feats[:, -1]

		# Embeddings for Corpus given Query
		for i in range(len(self.attention_layers)):
			corp_seqs = torch.transpose(corp_seqs, 0, 1)
			inp_qrs = torch.transpose(query_seqs, 0, 1)
			Q = self.attention_layernorms[i](corp_seqs)
			mha_outputs, _ = self.attention_layers[i](Q, inp_qrs, corp_seqs)
			corp_seqs = Q + mha_outputs
			corp_seqs = torch.transpose(corp_seqs, 0, 1)

			corp_seqs = self.forward_layernorms[i](corp_seqs)
			corp_seqs = self.forward_layers[i](corp_seqs)

		cq_log_feats = self.last_layernorm(corp_seqs)
		last_cq_feats = cq_log_feats[:, -1]

		return last_query_feats, last_corp_feats, last_cq_feats

	def seq2feats(self, query_ev, query_ti, pos_corpus_ev, pos_corpus_ti, neg_corpus_ev, neg_corpus_ti, dump_embs=False):
		# Position Vectors
		positions = np.tile(np.array(range(query_ev.shape[1])), [query_ev.shape[0], 1])
		positions = self.pos_emb(torch.LongTensor(positions).to(self.dev))

		# Query
		query_seqs = self.item_emb(torch.LongTensor(query_ev).to(self.dev))
		time_query = torch.Tensor(query_ti).to(self.dev)
		time_query = torch.reshape(time_query, (time_query.shape[0], time_query.shape[1], 1))
		time_embs_query = self.time_embeddings(time_query)
		query_seqs += time_embs_query
		query_seqs *= self.item_emb.embedding_dim ** 0.5
		query_seqs += positions
		query_seqs = self.item_emb_dropout(query_seqs)

		# Positive Corpus
		pos_seqs = self.item_emb(torch.LongTensor(pos_corpus_ev).to(self.dev))
		time_pos = torch.Tensor(pos_corpus_ti).to(self.dev)
		time_pos = torch.reshape(time_pos, (time_pos.shape[0], time_pos.shape[1], 1))
		time_embs_pos = self.time_embeddings(time_pos)
		pos_seqs += time_embs_pos
		pos_seqs = self.get_event_embeddings(pos_corpus_ev, pos_corpus_ti)
		pos_seqs *= self.item_emb.embedding_dim ** 0.5
		pos_seqs += positions
		pos_seqs = self.item_emb_dropout(pos_seqs)

		# Negative Corpus
		neg_seqs = self.item_emb(torch.LongTensor(neg_corpus_ev).to(self.dev))
		time_neg = torch.Tensor(neg_corpus_ti).to(self.dev)
		time_neg = torch.reshape(time_neg, (time_neg.shape[0], time_neg.shape[1], 1))
		time_embs_neg = self.time_embeddings(time_neg)
		neg_seqs += time_embs_neg
		neg_seqs *= self.item_emb.embedding_dim ** 0.5
		neg_seqs += positions
		neg_seqs = self.item_emb_dropout(neg_seqs)
		
		# For masking zeros in rows
		timeline_mask = torch.BoolTensor(query_ev == 0).to(self.dev)
		query_seqs *= ~timeline_mask.unsqueeze(-1) # Remove events before last N
		pos_seqs *= ~timeline_mask.unsqueeze(-1) # Remove events before last N
		neg_seqs *= ~timeline_mask.unsqueeze(-1) # Remove events before last N

		tl = query_seqs.shape[1]
		attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

		# Attention for Query Sequence
		inp_seqs = query_seqs
		for i in range(len(self.attention_layers)):
			inp_seqs = torch.transpose(inp_seqs, 0, 1)
			Q = self.attention_layernorms[i](inp_seqs)
			mha_outputs, _ = self.attention_layers[i](Q, inp_seqs, inp_seqs, attn_mask=attention_mask)
			inp_seqs = Q + mha_outputs
			inp_seqs = torch.transpose(inp_seqs, 0, 1)

			inp_seqs = self.forward_layernorms[i](inp_seqs)
			inp_seqs = self.forward_layers[i](inp_seqs)
			inp_seqs *=  ~timeline_mask.unsqueeze(-1)

		query_log_feats = self.last_layernorm(inp_seqs)
		last_query_feats = query_log_feats[:, -1]

		# Attention for Positive Corpus Sequence
		# attention_mask = torch.ones((tl, tl), dtype=torch.bool, device=self.dev)
		for i in range(len(self.attention_layers)):
			pos_seqs = torch.transpose(pos_seqs, 0, 1)
			inp_qrs = torch.transpose(query_seqs, 0, 1)
			Q = self.attention_layernorms[i](pos_seqs)
			mha_outputs, _ = self.attention_layers[i](Q, inp_qrs, pos_seqs)
			pos_seqs = Q + mha_outputs
			pos_seqs = torch.transpose(pos_seqs, 0, 1)

			pos_seqs = self.forward_layernorms[i](pos_seqs)
			pos_seqs = self.forward_layers[i](pos_seqs)
			pos_seqs *=  ~timeline_mask.unsqueeze(-1)

		pos_log_feats = self.last_layernorm(pos_seqs)
		last_pos_feats = pos_log_feats[:, -1]

		# Attention for Negative Corpus Sequence
		for i in range(len(self.attention_layers)):
			neg_seqs = torch.transpose(neg_seqs, 0, 1)
			inp_qrs = torch.transpose(query_seqs, 0, 1)
			Q = self.attention_layernorms[i](neg_seqs)
			mha_outputs, _ = self.attention_layers[i](Q, inp_qrs, neg_seqs)
			neg_seqs = Q + mha_outputs
			neg_seqs = torch.transpose(neg_seqs, 0, 1)

			neg_seqs = self.forward_layernorms[i](neg_seqs)
			neg_seqs = self.forward_layers[i](neg_seqs)
			neg_seqs *=  ~timeline_mask.unsqueeze(-1)

		neg_log_feats = self.last_layernorm(neg_seqs)
		last_neg_feats = neg_log_feats[:, -1]

		return query_log_feats, pos_log_feats, neg_log_feats, last_query_feats, last_pos_feats, last_neg_feats

	def calc_mean_var(self, log_feats, seq_ev, seq_ti):
		ev_embs = self.get_event_embeddings(seq_ev, seq_ti)
		log_feats = torch.cat((log_feats, ev_embs), 2)
		mean_vec = self.mean_nn(log_feats)
		var_vec = self.var_nn(log_feats)
		var_vec = torch.min(torch.exp(var_vec), torch.ones_like(var_vec)*100)
		return mean_vec, var_vec

	def kl_divergence(self, query_mean, query_var, corp_mean, corp_var):
		part_1 = torch.log(torch.div(corp_var, query_var))
		num_part_2 = torch.square(query_var) + torch.square(torch.sub(query_mean, corp_mean))
		den_part_2 = 2*torch.square(query_var)
		part_2 = torch.div(num_part_2, den_part_2)
		part_3 = torch.ones_like(part_2)*0.5
		div = part_1 + part_2 - part_3
		return div

	def forward(self, seq_id, query_ev, query_ti, pos_corpus_ev, pos_corpus_ti, neg_corpus_ev, neg_corpus_ti):
		query_log_feats, pos_log_feats, neg_log_feats, last_query_feats, last_pos_feats, last_neg_feats = \
		self.seq2feats(query_ev, query_ti, pos_corpus_ev, pos_corpus_ti, neg_corpus_ev, neg_corpus_ti)

		'''Uncomment these lines to run the KL divergence based ablation model'''
		# query_mean, query_var = self.calc_mean_var(query_log_feats, query_ev, query_ti)
		# pos_mean, pos_var = self.calc_mean_var(pos_log_feats, pos_corpus_ev, pos_corpus_ti)
		# neg_mean, neg_var = self.calc_mean_var(neg_log_feats, neg_corpus_ev, neg_corpus_ti)

		# dist_pos = self.kl_divergence(query_mean, query_var, pos_mean, pos_var)
		# dist_neg = self.kl_divergence(query_mean, query_var, neg_mean, neg_var)
		# diff_loss = self.to_positive(dist_pos.sum(dim=-1) - dist_neg.sum(dim=-1))

		# Maching the distance based simmilarity between query and corpus
		wasserstein = WassDistance(eps=1, max_iter=100)
		dist_pos, P, C = wasserstein(query_log_feats, pos_log_feats, self.dev)
		dist_neg, P, C = wasserstein(query_log_feats, neg_log_feats, self.dev)
		dist_pos = dist_pos.to(self.dev)
		dist_neg = dist_neg.to(self.dev)

		# Cross-Entropy between query and corpus index_wise and sequence_ends
		index_wise_pos = torch.sigmoid(self.index_wise(pos_log_feats))
		index_wise_neg = torch.sigmoid(self.index_wise(neg_log_feats))

		seq_end_pos = torch.sigmoid(self.seq_end(last_pos_feats))
		seq_end_neg = torch.sigmoid(self.seq_end(last_neg_feats))
		
		# For Next Mark Prediction -- Shifting Marks to the right
		query_ev = np.roll(query_ev, -1, axis=1)
		pos_corpus_ev = np.roll(pos_corpus_ev, -1, axis=1)
		neg_corpus_ev = np.roll(neg_corpus_ev, -1, axis=1)

		query_embs = self.item_emb(torch.LongTensor(query_ev).to(self.dev))
		pos_embs = self.item_emb(torch.LongTensor(pos_corpus_ev).to(self.dev))
		neg_embs = self.item_emb(torch.LongTensor(neg_corpus_ev).to(self.dev))

		query_logits = (query_log_feats * query_embs).sum(dim=-1)
		pos_logits = (pos_log_feats * pos_embs).sum(dim=-1)

		return dist_pos, dist_neg, query_logits, pos_logits, index_wise_pos, index_wise_neg, seq_end_pos, seq_end_neg

	def sequence_dumps(self, query_ev, query_ti, corpus_ev, corpus_ti):
		last_query_feats, last_corp_feats, last_cq_feats = self.seq2dumps(query_ev, query_ti, corpus_ev, corpus_ti)
		return last_query_feats, last_corp_feats, last_cq_feats