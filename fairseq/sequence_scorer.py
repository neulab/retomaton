# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sys
import numpy as np
import scipy.sparse as sp
import pickle

from fairseq import utils
from fairseq.data import Dictionary


class SequenceScorer(object):
    """Scores the target for a given source sentence."""

    def __init__(self, tgt_dict, softmax_batch=None, compute_alignment=False, args=None):
        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos()
        self.softmax_batch = softmax_batch or sys.maxsize
        assert self.softmax_batch > 0
        self.compute_alignment = compute_alignment
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if args.cluster is None:
            self.cluster = None
            print('No clustering is used.')
        else:
            self.cluster = np.load(args.cluster)
        
        if args.members is None:
            self.members = None
            print('No cluster-members file is used.')
        else:
            with open(args.members, 'rb') as file:
                self.members = pickle.load(file)
        
        if self.members is None or self.cluster is None:
            self.extend_pointers_using_clusters = lambda pointers: pointers
        self.max_knns = self.args.max_knns if self.args.max_knns is not None else self.args.dstore_size
        self.lookup_after_history = []        
        

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        net_input = sample['net_input']

        def batch_for_softmax(dec_out, target):
            # assumes decoder_out[0] is the only thing needed (may not be correct for future models!)
            first, rest = dec_out[0], dec_out[1:]
            bsz, tsz, dim = first.shape
            if bsz * tsz < self.softmax_batch:
                yield dec_out, target, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                flat_tgt = target.contiguous().view(flat.shape[:-1])
                s = 0
                while s < flat.size(1):
                    e = s + self.softmax_batch
                    yield (flat[:, s:e],) + rest, flat_tgt[:, s:e], False
                    s = e

        def gather_target_probs(probs, target):
            probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            return probs

        def combine_knn_and_vocab_probs(knn_p, vocab_p, coeff):
            combine_probs = torch.stack([vocab_p, knn_p], dim=0)
            coeffs = torch.ones_like(combine_probs)
            coeffs[0] = np.log(1 - coeff)
            coeffs[1] = np.log(coeff)
            curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)

            return curr_prob

        orig_target = sample['target']

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        for model in models:
            model.eval()
            decoder_out = model(**net_input)
            attn = decoder_out[1]
            if type(attn) is dict:
                attn = attn.get('attn', None)

            batched = batch_for_softmax(decoder_out, orig_target)
            probs, idx = None, 0
            for i, (bd, tgt, is_single) in enumerate(batched):
                sample['target'] = tgt
                curr_prob = model.get_normalized_probs(bd, log_probs=len(models) == 1, sample=sample).data

                if is_single:
                    probs = gather_target_probs(curr_prob, orig_target)
                else:
                    if probs is None:
                        probs = curr_prob.new(orig_target.numel())
                    step = curr_prob.size(0) * curr_prob.size(1)
                    end = step + idx
                    tgt_probs = gather_target_probs(curr_prob.view(tgt.shape + (curr_prob.size(-1),)), tgt)
                    probs[idx:end] = tgt_probs.view(-1)
                    idx = end
                sample['target'] = orig_target

            probs = probs.view(sample['target'].shape)

            if 'knn_dstore' in kwargs:
                dstore = kwargs['knn_dstore']
                # TxBxC
                queries = bd[1][self.args.knn_keytype]
                if len(models) != 1:
                    raise ValueError('Only knn *log* probs are supported.')

                yhat_knn_prob = dstore.get_knn_log_prob(
                        queries.permute(1, 0, 2).contiguous(),
                        orig_target,
                        pad_idx=self.pad)
                yhat_knn_prob = yhat_knn_prob.squeeze(-1)
                if self.args.fp16:
                    yhat_knn_prob = yhat_knn_prob.half()
                    probs = probs.half()

                probs = combine_knn_and_vocab_probs(
                            yhat_knn_prob, probs, self.args.lmbda)

            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None and torch.is_tensor(attn):
                attn = attn.data
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        if len(models) > 1:
            avg_probs.div_(len(models))
            avg_probs.log_()
            if avg_attn is not None:
                avg_attn.div_(len(models))

        bsz = avg_probs.size(0)
        hypos = []
        start_idxs = sample['start_indices'] if 'start_indices' in sample else [0] * bsz
        for i in range(bsz):
            # remove padding from ref
            ref = utils.strip_pad(sample['target'][i, start_idxs[i]:], self.pad) \
                if sample['target'] is not None else None
            tgt_len = ref.numel()
            avg_probs_i = avg_probs[i][start_idxs[i]:start_idxs[i] + tgt_len]
            score_i = avg_probs_i.sum() / tgt_len
            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                if self.compute_alignment:
                    alignment = utils.extract_hard_alignment(
                        avg_attn_i,
                        sample['net_input']['src_tokens'][i],
                        sample['target'][i],
                        self.pad,
                        self.eos,
                    )
                else:
                    alignment = None
            else:
                avg_attn_i = alignment = None
            hypos.append([{
                'tokens': ref,
                'score': score_i,
                'attention': avg_attn_i,
                'alignment': alignment,
                'positional_scores': avg_probs_i,
                'dstore_keys': decoder_out[1][self.args.knn_keytype][start_idxs[i]:,i,:] if self.args.save_knnlm_dstore else None,
            }])
        return hypos

    @torch.no_grad()
    def score_with_knnlm(self, hypos, dstore):
        # TxBxC
        for hypo in hypos:
            hypo = hypo[0]
            queries = hypo['queries'] # (time, dim)
            orig_target = hypo['tokens'] # (time, )
            lm_probs = hypo['positional_scores'] # (time, )
            if self.args.fp16:
                lm_probs = lm_probs.half()

            cur_knns = np.array([], dtype=np.int64)
            cur_dists = np.array([], dtype=np.float32)
            no_lookup_counter = 0

            probs_per_timestep = []
            for i in range(queries.size(0)):
                perform_search = False
                extended_pointers = None
                cur_knns = cur_knns[cur_dists.argsort()[::-1]]
                pointers = cur_knns + 1

                if self.args.no_pointer or cur_knns.size < self.args.min_knns:
                    perform_search = True
                    self.lookup_after_history.append(no_lookup_counter)
                    no_lookup_counter = 0
                else:
                    no_lookup_counter += 1

                extended_pointers = pointers
                if pointers.size >= self.max_knns:
                    extended_pointers = extended_pointers[:self.max_knns]
                elif pointers.size > 0 and not self.args.no_pointer:
                    extended_pointers = self.extend_pointers_using_clusters(pointers)
                
                cur_knn_log_prob, knns, correct_vals_mask, dists = dstore.get_knn_log_prob(
                    queries[i,:].unsqueeze(0),
                    orig_target[i].unsqueeze(0),
                    pointers=None if self.args.no_pointer else extended_pointers.reshape(1, -1),
                    perform_search=perform_search)

                if self.args.fp16:
                    cur_knn_log_prob = cur_knn_log_prob.half()
                
                if not self.args.no_pointer:
                    vals_are_correct_and_pointer_available = correct_vals_mask & (knns < self.args.dstore_size - 1)
                    cur_knns = knns[vals_are_correct_and_pointer_available]
                    cur_dists = dists[vals_are_correct_and_pointer_available]

                combined = self.combine_knn_and_vocab_probs(
                            cur_knn_log_prob, lm_probs[i].unsqueeze(0), self.args.lmbda, dstore)
                probs_per_timestep.append(combined[0])

            
            hypo['positional_scores'] = torch.as_tensor(probs_per_timestep)
        return hypos

    def extend_pointers_using_clusters(self, pointers):
        # Don't take the same cluster twice
        clusters, cluster_counts = np.unique(self.cluster[pointers], return_counts=True)
        # Take smaller clusters first
        clusters = clusters[np.argsort(-cluster_counts)]
        members = np.nonzero(self.members[clusters])[1]
        # Prefer datastore entries that were directly pointed to by the previous time step's
        # datastore entries, over other members of their cluster
        extended_pointers = np.concatenate([pointers, members])
        if len(extended_pointers) > self.max_knns:
            extended_pointers = extended_pointers[:self.max_knns]
        return extended_pointers