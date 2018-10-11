import os, json
with open("log", 'w') as f: #clear log file
    pass
batch_size_search_space = [4, 8, 16, 32, 64]
for batch_size in batch_size_search_space[::-1]:
    file = open('conf', 'w')
    conf = file.read()
    if len(conf) != 0:
        conf = json.loads(conf)
    else:
        conf = {}
    # file.close()
    conf['batch_size'] = batch_size
    json.dump(conf, file, ensure_ascii=False)
    file.close()
#plot running time curves
# plt.figure("Running time")
# plt.plot(range(epochs), running_time, label=name)
# plt.xlabel(str(epochs) + ' iterations')
# plt.ylabel("Running time (s)")
# plt.legend()
