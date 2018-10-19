import os, json
from subprocess import call
import pylab as plt

EPOCHS = 1000
BATCH_SIZE = 16
NHIDDEN = 15

with open("log", 'w') as f: #clear log file
    pass
beta_search_space = [0, 1e-3, 1e-6, 1e-9, 1e-12]
for beta in beta_search_space[::-1]:
    file = open('conf', 'r')
    conf = file.read()
    if len(conf) != 0:
        conf = json.loads(conf)
    else:
        conf = {}
    file.close()
    conf['epochs'] = EPOCHS
    conf['batch_size'] = BATCH_SIZE
    conf['nhidden'] = NHIDDEN
    conf['beta'] = beta
    file = open('conf', 'w')
    json.dump(conf, file, ensure_ascii=False)
    file.close()
    call(["python", "landsat.py"])
with open("log", 'r') as file:
    log = json.loads(file.read())
    plt.figure("Running time")
    plt.xlabel(str(EPOCHS) + ' iterations')
    plt.ylabel("Running time (s)")
    for entry in log["logs"]:
        name = "Beta: {}".format(entry['beta'])
        running_time = entry["running_time"]
        #plot running time curves
        plt.plot(range(EPOCHS), running_time, label=name)
        plt.legend()
    #plt.show()
    plt.savefig("Figures/Running_time.png", bbox_inches='tight')
