import sys 

def num_distinct_node_ids(dot_path, debug_print=False):
	nodes = []
	with open(dot_path) as f:
		for line in list(f):
			splitline = [x.strip() for x in line.split(" ")]
			ints = [k for k in splitline if k.isdigit()]
			for k in ints:
				nodes.append(k)
	distinct_nodes = len(set(nodes))
	if debug_print:
		print("Number of distinct nodes for " + dot_path, distinct_nodes)
	sorted_nodes = sorted(list(set(nodes)))
	return distinct_nodes, sorted_nodes

def list_diff(list1, list2): 
	return sorted((list(set(list1) - set(list2))))

def new_nodes(old_nodes, new_nodes):
	new_stuff = [k for k in new_nodes if k not in old_nodes]
	return sorted(list(set(new_stuff)))

def freed_nodes(old_nodes, new_nodes):
	freed_stuff = [k for k in old_nodes if k not in new_nodes]
	return sorted(list(set(freed_stuff)))	

def persisting_nodes(old_nodes, new_nodes):
	persisting_stuff = [k for k in old_nodes if k in new_nodes]
	return sorted(list(set(persisting_stuff)))
