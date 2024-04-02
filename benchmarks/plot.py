import glob
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

liblist = ["qiskit", "qulacs", "qiskit_qulacs"]


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


def filter_fun(x, name):
	if name == "qiskit":
		return (name in x) and ("qulacs" not in x)
	elif name == "qulacs":
		return (name in x) and ("qiskit" not in x)

	return name in x


def plot(dat, folder):
	test_classes = ["statevector", "estimator", "sampler", "gradient"]
	cmap = plt.get_cmap("tab10")
	cnt = 0

	fig, axs = plt.subplots(2, 2, figsize=(10, 10))
	if len(test_classes) == 1:
		axs = [axs]

	expts = list(dat.keys())

	for name in liblist:
		cid = liblist.index(name)
		tests = list(filter(lambda x: filter_fun(x, name), expts))
		for i, test in enumerate(test_classes):
			sub_tests = list(filter(lambda x: test in x, tests))
			for sub_test in sub_tests:
				stats = dat[sub_test]
				fil = np.array(list(stats.items())).T
				legend = (
					"qulacs_exp_val" if sub_test == "qulacs_estimator" else sub_test
				)
				# axs[i].plot(fil[0], fil[1], ".-", label=legend, c=cmap(cid))
				axs[i // 2][i % 2].plot(fil[0], fil[1], ".-", label=legend)

	for i in range(2):
		axs[i][0].set_ylabel("Time [sec]", fontsize=15)
		axs[-1][i].set_xlabel("# of qubits", fontsize=15)

	for ax, test in zip(axs.flatten(), test_classes):
		ax.legend()
		ax.set_xticks([5, 10, 15, 20])
		ax.set_yscale("log")
		ax.set_title(test)
		ax.tick_params(axis="x", labelsize=12)
		ax.tick_params(axis="y", labelsize=12)

		ax.grid(which="major", color="black", linestyle="-", alpha=0.3)
		ax.grid(which="minor", color="black", linestyle="-", alpha=0.1)

	# plt.xticks(fontsize=15)
	# plt.yticks(fontsize=15)
	plt.tight_layout()
	plt.legend(fontsize=10)
	plt.savefig(f"./plots/fig_compare_{folder}.png")
	plt.clf()


if __name__ == "__main__":
	# for folder in ["cpu", "gpu"]:
	for folder in ["cpu"]:
		dat = load(folder)
		plot(dat, folder)
