import os, json
with open("log", 'w') as f: #clear log file
    pass
batch_size_search_space = [4, 8, 16, 32, 64]
for batch_size in batch_size_search_space[::-1]:
    conf = json.loads(open('conf').read())
    conf['batch_size'] = batch_size
    with open('conf', 'w') as f:
        f.write(conf)
#plot running time curves
# plt.figure("Running time")
# plt.plot(range(epochs), running_time, label=name)
# plt.xlabel(str(epochs) + ' iterations')
# plt.ylabel("Running time (s)")
# plt.legend()
