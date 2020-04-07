import numpy as np
import matplotlib.pyplot as plt 

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

def make_plot_f1(ans_f1, que_f1, doc_f1, out_path):
	
	xvals_ans = list(map(lambda x:x[0], ans_f1))
	yvals_ans = list(map(lambda x:x[1], ans_f1))
	errs_ans = list(map(lambda x:x[2], ans_f1))

	xvals_ques = list(map(lambda x:x[0], que_f1))
	yvals_ques = list(map(lambda x:x[1], que_f1))
	errs_ques = list(map(lambda x:x[2], que_f1)) 

	xvals_docs = list(map(lambda x:x[0], doc_f1))
	yvals_docs = list(map(lambda x:x[1], doc_f1))
	errs_docs = list(map(lambda x:x[2], doc_f1))

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
	plt.savefig(out_path)

def test():
	ans_f1 = [(1, 0.5, 0.05), (2, 0.2, 0.09), (3, 0.1,0.05), (23, 0.5, 0.03), (24, 0.9, 0.03)]
	que_f1 = [(3, 0.9, 0.03), (2, 0.3, 0.03), (1, 0.1, 0.03), (34, 0.4, 0.02)]
	doc_f1 = [(124, 0.9, 0.01), (923, 0.2, 0.01), (422, 0.5, 0.01)]
	make_plot_f1(ans_f1, que_f1, doc_f1, "test.png")