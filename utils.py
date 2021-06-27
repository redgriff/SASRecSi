import sys
import copy
import torch
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from multiprocessing import Process, Queue

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def get_legnth_dict_items(dictionary):
    first_key = list(dictionary.keys())[0]
    return len(dictionary[first_key])


def sample_function(
    users_seqs,
    items_info,
    usernum,
    itemnum,
    batch_size,
    maxlen,
    result_queue,
    SEED,
):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(users_seqs[user]) <= 1:
            user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)

        seq_itm = np.zeros([maxlen, get_legnth_dict_items(items_info)])
        pos = np.zeros([maxlen], dtype=np.int32)
        pos_itm = np.zeros([maxlen, get_legnth_dict_items(items_info)])
        neg = np.zeros([maxlen], dtype=np.int32)
        neg_itm = np.zeros([maxlen, get_legnth_dict_items(items_info)])
        nxt = users_seqs[user][-1]
        idx = maxlen - 1

        ts = set(users_seqs[user])
        for i in reversed(users_seqs[user][:-1]):
            seq[idx] = i
            seq_itm[idx] = items_info[i]
            pos[idx] = nxt
            pos_itm[idx] = items_info[nxt]
            if nxt != 0:
                tmp = random_neq(1, itemnum + 1, ts)
                neg[idx] = tmp
                neg_itm[idx] = items_info[tmp]
            nxt = i
            neg_itm[idx] = i
            idx -= 1
            if idx == -1:
                break

        return (user, seq, seq_itm, pos, pos_itm, neg, neg_itm)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(
        self,
        users_seqs,
        items_info,
        usernum,
        itemnum,
        batch_size=64,
        maxlen=10,
        n_workers=1,
    ):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(
                    target=sample_function,
                    args=(
                        users_seqs,
                        items_info,
                        usernum,
                        itemnum,
                        batch_size,
                        maxlen,
                        self.result_queue,
                        np.random.randint(2e9),
                    ),
                )
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}

    # assume user/item index starting from 1
    with open(f"data/{fname}/reviews_Steam.txt", "r") as f:
        for line in f:
            u, i = line.rstrip().split(" ")
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)

        for user in User:
            nfeedback = len(User[user])
            if nfeedback < 3:
                user_train[user] = User[user]
                user_valid[user] = []
                user_test[user] = []
            else:
                user_train[user] = User[user][:-2]
                user_valid[user] = []
                user_valid[user].append(User[user][-2])
                user_test[user] = []
                user_test[user].append(User[user][-1])

    items_info = pd.read_csv(f"data/{fname}/items_info_pca.csv")

    items_info["id"] = items_info["id"].astype(int)
    items_info = items_info.set_index("id")
    items_info = {idx: row.values for idx, row in items_info.iterrows()}

    # items_info = items_info.to_dict('index')

    return [user_train, user_valid, user_test, items_info, usernum, itemnum]


# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, items_info, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        seq_itm = np.zeros([args.maxlen, get_legnth_dict_items(items_info)])
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        seq_itm[idx] = items_info[valid[u][0]]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            seq_itm[idx] = items_info[i]
            idx -= 1
            if idx == -1:
                break

        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        item_idx_itm = [items_info[test[u][0]]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            item_idx_itm.append(items_info[t])

        predictions = -model.predict(
            *[np.array(l) for l in [[u], [seq], [seq_itm], item_idx, item_idx_itm]]
        )
        predictions = predictions[0]  # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print(".", end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, items_info, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        seq_itm = np.zeros([args.maxlen, get_legnth_dict_items(items_info)])
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            seq_itm[idx] = items_info[i]
            idx -= 1
            if idx == -1:
                break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        item_idx_itm = [items_info[valid[u][0]]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            item_idx_itm.append(items_info[t])

        predictions = -model.predict(
            *[np.array(l) for l in [[u], [seq], [seq_itm], item_idx, item_idx_itm]]
        )
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print(".", end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

 