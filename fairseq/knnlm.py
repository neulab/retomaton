import torch
import faiss
import faiss.contrib.torch_utils
import math
import numpy as np
from fairseq import utils
import time
from fairseq.data import Dictionary

class KNN_Dstore(object):
    def __init__(self, args):
        self.half = args.fp16
        self.dimension = args.decoder_embed_dim
        self.k = args.k
        self.dstore_size = args.dstore_size
        self.metric_type = args.faiss_metric_type
        self.sim_func = args.knn_sim_func
        self.dstore_fp16 = args.dstore_fp16
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reconstruct_index, self.index = self.setup_faiss(args)
        self.args = args


    def setup_faiss(self, args):
        if not args.dstore_filename:
            raise ValueError('Cannot build a datastore without the data.')

        start = time.time()
        cpu_index = faiss.read_index(args.indexfile, faiss.IO_FLAG_ONDISK_SAME_DIR)
        print('Reading datastore took {} s'.format(time.time() - start))
        cpu_index.nprobe = args.probe

        if args.knnlm_gpu:
            start = time.time()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index, co)
            print('Moving index to GPU took {} s'.format(time.time() - start))
        else:
            gpu_index = cpu_index

        # make_direct_map() allows calling reconstruct(n), 
        # and reconstructing key vectors given their ids
        # currently, this is implemented only for CPU indexes:
        # https://github.com/facebookresearch/faiss/issues/2181
        cpu_index.make_direct_map()

        dstore_float_dtype = np.float32
        dstore_int_dtype = np.int32
        if args.dstore_fp16:
            print('Keys are fp16')
            dstore_float_dtype = np.float16

        if not args.no_load_keys:
            self.keys = np.memmap(args.dstore_filename+'_keys.npy', dtype=dstore_float_dtype, mode='r',
                                  shape=(self.dstore_size, self.dimension))
        self.vals = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int64, mode='r',
                              shape=(self.dstore_size, 1))
        # self.vals = torch.from_numpy(self.vals).to(self.device)

        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if args.move_dstore_to_mem:
            print('Loading to memory...')
            start = time.time()

            if not args.no_load_keys:
                del self.keys
                self.keys_from_memmap = np.memmap(args.dstore_filename+'_keys.npy', dtype=dstore_float_dtype, mode='r', shape=(self.dstore_size, self.dimension))
                self.keys = np.zeros((self.dstore_size, self.dimension), dtype=dstore_float_dtype)
                self.keys = self.keys_from_memmap[:]
                self.keys = self.keys.astype(dstore_float_dtype)

            del self.vals
            vals_from_memmap = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int, mode='r', shape=(self.dstore_size, 1))
            self.vals = np.zeros((self.dstore_size, 1), dtype=np.int64)
            self.vals = vals_from_memmap[:]
            self.vals = self.vals.astype(dstore_int_dtype)
            del vals_from_memmap
            print('Loading to memory took {} s'.format(time.time() - start))

        return cpu_index, gpu_index


    def get_knns(self, queries):
        start = time.time()
        dists, knns = self.index.search(queries, self.k)
        return dists, knns

    def reconstruct_ids(self, ids):
        reconstruct_func = np.vectorize(lambda x: self.reconstruct_index.reconstruct(int(x)), otypes=[object])
        vectors = reconstruct_func(ids[0])
        vectors = np.stack(vectors).reshape(ids.shape + (self.dimension, ))
        t_vectors = torch.from_numpy(vectors).to(self.device)
        if self.half:
            t_vectors = t_vectors.half()
        return t_vectors

    def get_knn_log_prob(self, queries, tgt, pointers=None, perform_search=True):
        def dist_func(d, k, q, function=None):
            if not function:
                # Default behavior for L2 metric is to recompute distances.
                # Default behavior for IP metric is to return faiss distances.
                qsize = q.shape
                if self.metric_type == 'l2':
                    start = time.time()
                    knns_vecs = torch.from_numpy(self.keys[k]).to(self.device)
                    if self.half:
                        knns_vecs = knns_vecs.half()
                    l2 = torch.sum((q.unsqueeze(1) - knns_vecs.detach())**2, dim=2)
                    return -1 * l2
                return d

            if function == 'dot':
                qsize = q.shape
                return (torch.from_numpy(self.keys[k]).to(self.device) * q.view(qsize[0], 1, qsize[1])).sum(dim=-1)

            if function == 'do_not_recomp_l2':
                return -1 * d

            raise ValueError("Invalid knn similarity function!")

        # queries  are BxTxC
        # reshape: (BxT)xC
        qshape = queries.shape
        queries = queries.reshape(-1, qshape[-1])
        tgt = tgt.contiguous().view(-1, 1)

        pointer_dists = torch.tensor([[]]).to(self.device)
        if pointers is not None and pointers.size > 0 and self.sim_func == 'do_not_recomp_l2':
            pointer_vectors = self.reconstruct_ids(pointers)
            pointer_dists = torch.sum((queries.unsqueeze(1) - pointer_vectors.detach()) ** 2, dim=2)
        
        if perform_search:
            # lookup KNNs
            # dists, knns = self.get_knns(queries[tgt != pad_idx])
            dists, knns = self.get_knns(queries)
            knns = knns.cpu().numpy()
            if pointers is not None and pointers.size > 0:
                knns = np.concatenate([knns, pointers], axis=-1)
                dists = torch.cat([dists, pointer_dists], axis=-1)
        else:
            knns = pointers
            dists = pointer_dists

        # Compute distance to KNNs
        # dists = dist_func(dists, knns, queries[tgt != pad_idx, :], function=self.sim_func)
        dists = dist_func(dists, knns, queries, function=self.sim_func)

        vals_at_knns = torch.from_numpy(self.vals[knns]).long().to(self.device).squeeze(-1)
        original_dists = dists.squeeze().cpu().numpy()
        vals_eq_tgt = torch.eq(vals_at_knns, tgt)
        original_vals_eq_tgt = vals_eq_tgt.squeeze().cpu().numpy()

        probs = utils.log_softmax(dists, dim=-1)

        # index_mask = torch.eq(torch.from_numpy(self.vals[knns]).long().cuda().squeeze(-1), tgt[tgt != pad_idx].unsqueeze(-1)).float()
        # index_mask[index_mask == 0] = -10000 # for stability
        # index_mask[index_mask == 1] = 0
        index_mask = torch.where(
            vals_eq_tgt, 0.0, -10000.0)

        # (T_reducedxB)
        yhat_knn_prob = torch.logsumexp(probs + index_mask, dim=-1).clone()
        # full_yhat_knn_prob = torch.full([qshape[0]*qshape[1]], -10000.0).cuda()
        # full_yhat_knn_prob[tgt != pad_idx] = yhat_knn_prob

        # TxBx1
        return yhat_knn_prob.view(-1,), knns.squeeze(), original_vals_eq_tgt, original_dists

