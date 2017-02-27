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
non_matches_decision_investigation = [i for i in features_of_attraction]
non_matches_decision_investigation.append('dec')
non_matches_decision_investigation.append('dec_o')
rating_by_partner_features = [i + '_o' for i in features_of_attraction]
for i in rating_by_partner_features:
    non_matches_decision_investigation.append(i)
non_matches_decision_investigation.append('match')
################################################
################################################
clean_up_1 = ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1', 
'attr1_2', 'sinc1_2', 'intel1_2', 'fun1_2', 'amb1_2', 'shar1_2',
'attr1_3', 'sinc1_3', 'intel1_3', 'fun1_3', 'amb1_3', 'shar1_3',
'attr2_1', 'sinc2_1', 'intel2_1', 'fun2_1', 'amb2_1', 'shar2_1', 
]
clean_up_2 = ['attr3_1', 'sinc3_1', 'intel3_1', 'fun3_1', 'amb3_1',
'attr3_2', 'sinc3_2', 'intel3_2', 'fun3_2', 'amb3_2',
'attr3_3', 'sinc3_3', 'intel3_3', 'fun3_3', 'amb3_3',
]
clean_up_3 = ['attr5_1', 'sinc5_1', 'intel5_1', 'fun5_1', 'amb5_1',
'attr5_2', 'sinc5_2', 'intel5_2', 'fun5_2', 'amb5_2',
'attr5_3', 'sinc5_3', 'intel5_3', 'fun5_3', 'amb5_3'
]
clean_up_4 = ['attr4_1', 'sinc4_1', 'intel4_1', 'fun4_1', 'amb4_1', 'shar4_1',
'attr4_2', 'sinc4_2', 'intel4_2', 'fun4_2', 'amb4_2', 'shar4_2',
'attr4_3', 'sinc4_3', 'intel4_3', 'fun4_3', 'amb4_3', 'shar4_3',
'attr2_3', 'sinc2_3', 'intel2_3', 'fun2_3', 'amb2_3', 'shar2_3'
]
clean_up_5 = ['attr2_2', 'sinc2_2', 'intel2_2', 'fun2_2', 'amb2_2', 'shar2_2']
################################################
################################################
actual_decisions = []
for i in range(1,3):
	for j in features_of_attraction:
		actual_decisions.append(j + '7_' + str(i))
################################################
################################################
if __name__ == '__main__':
	#print list_of_question
	#print features_of_attraction
	#print mod_list_of_question
	#print rounds
	#print lister	
	for i, j in data_cleaner.iteritems():
		print i, j, '\n'
	for i, j in master_list.items():
		print i, j, '\n'
	print 'clean_up_1', '\n', clean_up_1, '\n'
	print 'clean_up_2', '\n', clean_up_2, '\n'
	print 'clean_up_3', '\n', clean_up_3, '\n'
	print 'clean_up_4', '\n', clean_up_4, '\n'
	print 'clean_up_5', '\n', clean_up_5, '\n'
	print 'actual_decisions', '\n', actual_decisions, '\n'
	print 'non_matches_decision_investigation', '\n', non_matches_decision_investigation, '\n'
################################################
################################################