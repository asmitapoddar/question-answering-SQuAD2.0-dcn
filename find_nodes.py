import os

dotPath = "dot_output2.dot"

with open("persisting_nodes.txt") as persisting, open(dotPath) as dotOut:
	for line in list(persisting):
		line = line.strip()
		cmd = "less " + dotPath + " | grep -A 1 --color " + "\"" + str(line) + " \[\""
		os.system(cmd)

