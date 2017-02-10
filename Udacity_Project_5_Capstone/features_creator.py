import pandas as pd
import numpy as np
import sklearn
from IPython.display import display
################################################Set up questions asked repeatedly through event
################################################
list_of_question = ['what_you_want', 'you_think_opposite_wants', 'how_you_measure', 
'you_think_most_want', 'you_think_they_rate_you']
################################################features used through the event
################################################
features_of_attraction = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']
################################################
################################################
mod_list_of_question = [str(i) + '_' + str(j) for i in list_of_question for j in features_of_attraction]
################################################set up features
################################################
lister = [str(k) + str(i) + '_' + str(j) 
for j in range(1,4) for i in range(1,6) for k in features_of_attraction ]
################################################make a list of features
################################################with indices
master_list = {}#this is a dictionary that places the same features for day of event, day after, and three weeks after
count = 0
j = 0
k = 0
for i in mod_list_of_question:
	#print i
	master_list[i] = []
	while j >= 0 and j <= 89:
		if count >= 0 and count <= 3:
			master_list[i].append(lister[j])
			j += 30
			count += 1
			#print i, j, k
	k += 1
	j = k
	count = 0
################################################drop redundant features that were
################################################made by automation above
drop_stuff = ['how_you_measure_shar', 'you_think_they_rate_you_shar']
for i in drop_stuff:
	del master_list[i]
################################################
################################################
rounds = ['first_round', 'second_round', 'third_round']
data_cleaner = {}
counter = 0
j = 0
for i in range(len(rounds)):
	data_cleaner[rounds[i]] = []
	while counter >= 0 and counter < 30:
		data_cleaner[rounds[i]].append(lister[j])
		counter += 1
		j += 1
		#print counter
	if counter == 30:
		#print data_cleaner
		counter = 0
		i += 1
		continue
################################################
################################################
more_stuff_to_drop = ['shar3_1', 'shar3_2', 'shar3_3']
additional_stuff_to_drop = ['shar5_1', 'shar5_2', 'shar5_3']
for i, j, k in zip(rounds, more_stuff_to_drop, additional_stuff_to_drop):
		data_cleaner[i].remove(j)
		data_cleaner[i].remove(k)
################################################
################################################
if __name__ == '__main__':
	"""print list_of_question
	print features_of_attraction
	print mod_list_of_question
	print rounds
	print lister"""	
	for i, j in data_cleaner.iteritems():
		print i, j
	for i, j in master_list.items():
		print i, j
################################################
################################################