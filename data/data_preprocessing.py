import os
import pickle as pkl
import gzip
from collections import defaultdict
from datetime import datetime

ROOT_DIR = os.path.join(os.getcwd(), "data", "Steam")


def parse(g):
    for l in g:
        yield eval(l)


countU = defaultdict(lambda: 0)
countP = defaultdict(lambda: 0)
line = 0

dataset_name = "Steam"

json_file_path = os.path.join(ROOT_DIR, f"reviews_{dataset_name}.json.gz")
with gzip.open(json_file_path, "r") as g:
    line = 0
    for l in parse(g):
        line += 1
        if line % 100000 == 0:
            print(line)

        product_id = int(l["product_id"])
        username = l["username"]
        time = datetime.strptime(l["date"], "%Y-%m-%d")
        countU[username] += 1
        countP[product_id] += 1

usermap = dict()
usernum = 0
itemmap = dict()
itemnum = 0
users_seqs = dict()
with gzip.open(json_file_path, "r") as g:
    line = 0
    for l in parse(g):
        line += 1
        if line % 100000 == 0:
            print(line)

        product_id = int(l["product_id"])
        username = l["username"]
        time = datetime.strptime(l["date"], "%Y-%m-%d")
        if countU[username] < 5 or countP[product_id] < 5:
            continue

        if username in usermap:
            userid = usermap[username]
        else:
            usernum += 1
            userid = usernum
            usermap[username] = userid
            users_seqs[userid] = []

        if product_id in itemmap:
            itemid = itemmap[product_id]
        else:
            itemnum += 1
            itemid = itemnum
            itemmap[product_id] = itemid

        users_seqs[userid].append([time, itemid])
# sort reviews in User according to time

for userid in users_seqs.keys():
    users_seqs[userid].sort(key=lambda x: x[0])

print(usernum, itemnum)

with open(os.path.join(ROOT_DIR, "usermap.pkl"), "wb") as handle:
    pkl.dump(usermap, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(os.path.join(ROOT_DIR, "itemmap.pkl"), "wb") as handle:
    pkl.dump(itemmap, handle, protocol=pkl.HIGHEST_PROTOCOL)

txt_file_path = os.path.join(ROOT_DIR, f"reviews_{dataset_name}.txt")
f = open(txt_file_path, "w")
for user in users_seqs.keys():
    for i in users_seqs[user]:
        f.write(f"{user} {i[1]}\n")
f.close()
