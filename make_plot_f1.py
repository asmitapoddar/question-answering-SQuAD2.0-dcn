import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def make_plot_f1_against_prediction_length(pred_len_f1, outpath):
	xvals = list(map(lambda x:x[0], pred_len_f1))
	yvals = list(map(lambda x:x[1], pred_len_f1))
	
	down_errs = list(map(lambda x:x[2], pred_len_f1))
	up_errs = list(map(lambda x:x[3], pred_len_f1))
	errs = [down_errs, up_errs]

	yticks = np.arange(0, 1.4, 0.2)
	
	xdelta = 50
	xticks = range(0, max(xvals) + xdelta, xdelta)

	fig = plt.figure()
	ax = fig.add_subplot()

	ax.grid(color='white')
	ax.set_facecolor('gainsboro')

	ax.set(ylabel="F1")
	ax.set(xlabel="# Tokens in Predicted Answer")

	ax.set(xticks=xticks)
	ax.set(yticks=yticks)

	ax.set(ylim=(min(yticks), max(yticks)))
	ax.errorbar(xvals, yvals, yerr=errs, fmt='o', markeredgecolor='k', ecolor='lightskyblue', capsize=4.0)
	plt.tight_layout()
	plt.savefig(outpath, dpi=300)

def make_plot_questions(que_f1, outpath):
	xvals_ques = list(map(lambda x:x[0], que_f1))
	yvals_ques = list(map(lambda x:x[1], que_f1))
	
	down_errs_ques = list(map(lambda x:x[2], que_f1))
	up_errs_ques = list(map(lambda x:x[3], que_f1))
	errs_ques = [down_errs_ques, up_errs_ques]

	yticks = np.arange(0, 1.4, 0.2)
	xdelta = 5
	xticks = range(0, max(xvals) + xdelta, xdelta)

	fig = plt.figure()
	ax = fig.add_subplot()

	ax.grid(color='white')
	ax.set_facecolor('gainsboro')

	ax.set(ylabel="F1")
	ax.set(xlabel="# Tokens in Question")

	ax.set(xticks=xticks)
	ax.set(yticks=yticks)

	ax.set(ylim=(min(yticks), max(yticks)))
	ax.errorbar(xvals_ques, yvals_ques, yerr=errs_ques, fmt='o', markeredgecolor='k', ecolor='lightskyblue', capsize=4.0)
	plt.tight_layout()
	plt.savefig(outpath, dpi=300)

def make_plot_f1(ans_f1, que_f1, doc_f1, out_path):
	
	xvals_ans = list(map(lambda x:x[0], ans_f1))
	yvals_ans = list(map(lambda x:x[1], ans_f1))

	down_errs_ans = list(map(lambda x:x[2], ans_f1))
	up_errs_ans = list(map(lambda x:x[3], ans_f1))
	errs_ans = [down_errs_ans, up_errs_ans]

	xvals_ques = list(map(lambda x:x[0], que_f1))
	yvals_ques = list(map(lambda x:x[1], que_f1))
	
	down_errs_ques = list(map(lambda x:x[2], que_f1))
	up_errs_ques = list(map(lambda x:x[3], que_f1))
	errs_ques = [down_errs_ques, up_errs_ques]

	xvals_docs = list(map(lambda x:x[0], doc_f1))
	yvals_docs = list(map(lambda x:x[1], doc_f1))
	
	down_errs_docs = list(map(lambda x:x[2], doc_f1))
	up_errs_docs = list(map(lambda x:x[3], doc_f1))
	errs_docs = [down_errs_docs, up_errs_docs]


	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
	
	yticks = np.arange(0, 1.4, 0.2)
	
	ax1.grid(color='white')
	ax1.set_facecolor('gainsboro')
	
	ax2.grid(color='white')
	ax2.set_facecolor('gainsboro')

	ax3.grid(color='white')
	ax3.set_facecolor('gainsboro')

	#Documents
	ax1.set(ylabel="F1")
	ax1.set(xlabel="# Tokens in Document")
	xdelta_doc = 100
	ax1.set(xticks = range(0, max(xvals_docs)+xdelta_doc, xdelta_doc))
	ax1.set(yticks = yticks)
	ax1.set(ylim = (min(yticks), max(yticks)))
	ax1.errorbar(xvals_docs, yvals_docs, yerr=errs_docs, fmt='o', markeredgecolor='k', ecolor='lightskyblue', capsize=4.0)

	#Questions
	xdelta = 5
	ax2.set(xticks = range(0, max(xvals_ques)+xdelta, xdelta))
	#ax2.set(yticks = yticks)
	ax2.set(ylim = (min(yticks), max(yticks)))
	ax2.errorbar(xvals_ques, yvals_ques, yerr=errs_ques, fmt='o', markeredgecolor='k', ecolor='lightskyblue', capsize=4.0)
	ax2.set(xlabel="# Tokens in Question")

	# Answers
	ax3.set(xticks = range(0, max(xvals_ans)+xdelta, xdelta))
	ax3.set(yticks = yticks)
	#ax3.set(ylim = (min(yticks), max(yticks)))
	ax3.errorbar(xvals_ans, yvals_ans, yerr=errs_ans, fmt='o', markeredgecolor='k', ecolor='lightskyblue', capsize=4.0)
	ax3.set(xlabel="Average # Tokens in Answer")

	plt.tight_layout()
	plt.savefig(out_path, dpi=300)


def plot_f1_histogram(all_f1s, out_path):
	plt.figure()
	plt.hist(all_f1s, weights=np.ones(len(all_f1s)) / len(all_f1s))
	plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
	plt.xlabel("F1 score")
	plt.ylabel("% questions")
	plt.savefig(out_path, dpi=300)


def f1_distribution_summary(all_f1s, out_path):
	num_f1s = len(all_f1s)
	zero_f1s = np.count_nonzero(np.array(all_f1s) < 1e-6)
	percent_zero_f1s = 100 * zero_f1s / num_f1s

	one_f1s = np.count_nonzero(np.array(all_f1s) > 1.0 - 1e-6)
	percent_one_f1s = 100 * one_f1s / num_f1s

	partial_f1s = num_f1s - (zero_f1s + one_f1s)
	percent_partial_f1s = 100 * partial_f1s / num_f1s

	with open(out_path, "w") as f:
		f.write("zero_f1=%s, one_f1=%s, partial_f1=%s\n" % (percent_zero_f1s, percent_one_f1s, percent_partial_f1s))

def test():
	# last two components of each tuple are values for error bars (upper limit, lower limit)
	ans_f1 = [(1, 0.5, 0.05, 0.05), (2, 0.2, 0.09, 0.09), (3, 0.1,0.05, 0.09), (23, 0.5, 0.03, 0.09), (24, 0.9, 0.03, 0.09), (103, 0.2, 0.9, 0.09)]
	que_f1 = [(3, 0.9, 0.03, 0.09), (2, 0.3, 0.03, 0.09), (1, 0.1, 0.03, 0.09), (34, 0.4, 0.02, 0.09)]
	doc_f1 = [(124, 0.9, 0.01, 0.09), (923, 0.2, 0.01, 0.09), (422, 0.5, 0.01, 0.09)]
	make_plot_f1(ans_f1, que_f1, doc_f1, "test_make_plot_f1_1.png")

	f1s = np.array([min(1, max(0, np.random.normal(0.5, 0.2))) for _ in range(1000)])
	plot_f1_histogram(f1s, "test_make_plot_f1_2.png")

	make_plot_f1_against_prediction_length(ans_f1, "test_make_plot_f1_3.png")


if __name__ == '__main__':
	print("Running make_plot_f1.test()")
	test()