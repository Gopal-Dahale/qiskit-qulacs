import glob
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

liblist = ["qiskit", "qulacs"]


def load(folder_name):
    filepaths = []

    for libname in liblist:
        path = f"./{folder_name}/{libname}/.benchmarks/*/*.json"
        flist = glob.glob(path)
        flist = [fname.replace("\\", "/") for fname in flist]
        # pick latest one
        flist.sort(key=lambda x: int(x.split("/")[-1].split("_")[0]), reverse=True)
        if len(flist) > 0:
            filepaths.append((libname, flist[0]))

    dat = defaultdict(dict)
    for filepath in filepaths:
        data = json.load(open(filepath[1]))

        def fetch_normal(libname, dat, data):
            items = data["benchmarks"]
            for item in items:
                name = item["group"]
                nqubits = int(item["param"])
                stats = item["stats"]
                key = name
                dat[key][nqubits] = float(stats["min"])

        fetch_normal(filepath[0], dat, data)

    return dat


def plot(dat, folder):
    tests = sorted(list(set([test.split("_")[-1] for test in list(dat.keys())])))
    cmap = plt.get_cmap("tab10")
    cnt = 0

    fig, axs = plt.subplots(1, len(tests), figsize=(5 * len(tests), 5))
    if len(tests) == 1:
        axs = [axs]

    for name in liblist:
        cid = liblist.index(name)
        for i, test in enumerate(tests):
            stats = dat[f"{name}_{test}"]
            fil = np.array(list(stats.items())).T
            legend = "pure qiskit" if name == "qiskit" else "qiskit-qulacs"
            axs[i].plot(fil[0], fil[1], ".-", label=legend, c=cmap(cid))

    axs[0].set_ylabel("Time [sec]", fontsize=15)
    for ax, test in zip(axs, tests):
        ax.legend()
        ax.set_xlabel("# of qubits", fontsize=15)
        ax.set_xticks([5, 10, 15, 20])
        ax.set_yscale("log")
        ax.set_title(test)
        ax.grid(which="major", color="black", linestyle="-", alpha=0.3)
        ax.grid(which="minor", color="black", linestyle="-", alpha=0.1)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.legend(fontsize=10)
    plt.savefig(f"./plots/fig_compare_{folder}.png")
    plt.clf()


if __name__ == "__main__":
    for folder in ["cpu", "gpu"]:
        dat = load(folder)
        plot(dat, folder)
