import pickle, numpy as np
opt = pickle.load(open("./conc-optimal100.pkl", "br"))
# lines = open("./kld_1e-2_hidden_64_final.txt", "r").readlines()
lines = open("./kld_1e-4_hidden_64.txt", "r").readlines()
b = np.array(list(map(lambda line: np.float64(line.strip()[5:]), lines)))
opt = np.array(list(map(lambda x: x[0], opt[0])))
diff = (b / opt - 1) * 100
print(diff.mean())
