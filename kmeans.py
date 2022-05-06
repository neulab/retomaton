import faiss
from faiss import IndexFlatL2
import numpy as np
import argparse
from collections import defaultdict
from tqdm import tqdm
import scipy.sparse as sp
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dstore-size', type=int, default=103225485)
    parser.add_argument('--dstore', type=str, help='path to store')
    parser.add_argument('--num-clusters', type=int, help='typically, around 1/100 of the datastore size. See also Figures 8 and 9 in the paper: https://arxiv.org/pdf/2201.12431.pdf')
    parser.add_argument('--sample', type=int, help='The number of data samples to use as the clustering data. If possible -- use the entire datastore. If not, use as large sample as memory allows.')
    parser.add_argument('--batch-size', type=int, default=500000)
    parser.add_argument('--dim', type=int, default=1024)
    parser.add_argument('--save', type=str)
    args = parser.parse_args()

    keys = np.memmap(args.dstore + '_keys.npy',
                    dtype=np.float16, mode='r', shape=(args.dstore_size, args.dim))

    rs = np.random.RandomState(1)
    if args.sample > args.dstore_size:
        print('Taking all data for training')
        to_cluster = keys[:]
    else:
        to_cluster = np.zeros((args.sample, args.dim), dtype=np.float16)
        idx = rs.choice(np.arange(args.dstore_size), size=args.sample, replace=False)
        to_cluster[:] = keys[idx]

    to_cluster = to_cluster.astype(np.float32)

    print('start cluster')
    niter = 20
    verbose = True

    kmeans = faiss.Kmeans(args.dim, args.num_clusters, niter=niter, verbose=verbose, gpu=True, seed=1)
    kmeans.train(to_cluster)

    centroids_filename = f'{args.save}_s{args.sample}_k{args.num_clusters}_centroids.npy'
    np.save(centroids_filename, kmeans.centroids)
    print(f'Saved centroids to {centroids_filename}')

    # Finished training the k-means clustering,
    # Now we assign each data point to its closest centroid

    vals_from_memmap = np.memmap(args.dstore + '_vals.npy',
                                dtype=np.int64, mode='r', shape=(args.dstore_size, 1))
    vals = np.zeros((args.dstore_size, 1), dtype=np.int64)
    vals[:] = vals_from_memmap[:]
    vals = vals.squeeze()
    del vals_from_memmap

    print('to add:', args.dstore_size)

    print('Creating index and adding centroids')
    index = IndexFlatL2(args.dim)
    index.add(kmeans.centroids)
    print('Index created, moving index to GPU')
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index, co)
    print('Moved index to GPU')

    start = 0
    # dists = []
    centroid_ids = []

    print('Starting to add tokens')
    while start < args.dstore_size:
        end = min(args.dstore_size, start + args.batch_size)
        to_search = keys[start:end].copy()
        d, key_i = index.search(to_search.astype(np.float32), 1)
        # dists.append(d.squeeze())
        centroid_ids.append(key_i.squeeze())
        start += args.batch_size
        if (start % 1000000) == 0:
            print('Assigned %d tokens so far' % start)

    # print(np.concatenate(dists).shape)
    # np.save(args.centroids + '_dist.npy', np.concatenate(dists))
    centroid_ids = np.concatenate(centroid_ids)
    centroid_ids_filename = f'{args.centroids}_centroid_ids.npy'
    np.save(centroid_ids_filename, centroid_ids)

    print('Saved centroid assignments, processing the mapping of cluster->members')

    parent_cluster = centroid_ids

    cluster_to_members = defaultdict(set)
    for key_i, cluster in tqdm(enumerate(parent_cluster), total=args.dstore_size):
        cluster_to_members[cluster.item()].add(key_i)

    row_ind = [k for k, v in cluster_to_members.items() for _ in range(len(v))]
    col_ind = [i for ids in cluster_to_members.values() for i in ids]
    members_sp = sp.csr_matrix(([1]*len(row_ind), (row_ind, col_ind)))

    members_filename = args.centroids + '_members.pkl'
    with open(members_filename, 'wb') as f:
        pickle.dump(members_sp, f)

    print(f'Done, found {len(cluster_to_members)} clusters, written to {members_filename}')