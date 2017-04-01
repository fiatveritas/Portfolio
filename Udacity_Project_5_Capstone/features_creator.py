import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
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
preferences_of_attraction = ['pf_o_' + i[:3] for i in features_of_attraction]
################################################
################################################
halfway_questions = [i + '1_s' for i in features_of_attraction]
for i in features_of_attraction[:-1]:
	halfway_questions.append(i + '3_s')
################################################
################################################
interests = ['sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga']
################################################
################################################
mod_list_of_question = [str(i) + '_' + str(j) for i in list_of_question for j in features_of_attraction]
################################################
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
rating_by_partner_features = [i + '_o' for i in features_of_attraction]
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

clean_up_6 = [j + str(i) + '_s' for i in range(1,4) for j in features_of_attraction]
clean_up_6 = clean_up_6[:6] + clean_up_6[12:]
del clean_up_6[-1]
################################################
################################################
to_do_list = [['attr3_1', 'sinc3_1', 'intel3_1', 'fun3_1', 'amb3_1'], 
['attr3_2', 'sinc3_2', 'intel3_2', 'fun3_2', 'amb3_2'], 
['attr3_3', 'sinc3_3', 'intel3_3', 'fun3_3', 'amb3_3']]
to_do_list_2 = [['attr4_1', 'sinc4_1', 'intel4_1', 'fun4_1', 'amb4_1', 'shar4_1'],
['attr4_2', 'sinc4_2', 'intel4_2', 'fun4_2', 'amb4_2', 'shar4_2'],
['attr4_3', 'sinc4_3', 'intel4_3', 'fun4_3', 'amb4_3', 'shar4_3'],
['attr2_3', 'sinc2_3', 'intel2_3', 'fun2_3', 'amb2_3', 'shar2_3']]
to_do_list_3 = [['attr5_1', 'sinc5_1', 'intel5_1', 'fun5_1', 'amb5_1'],
['attr5_2', 'sinc5_2', 'intel5_2', 'fun5_2', 'amb5_2'],
['attr5_3', 'sinc5_3', 'intel5_3', 'fun5_3', 'amb5_3']]
to_do_list_4 =  ['attr3_s', 'sinc3_s', 'intel3_s', 'fun3_s', 'amb3_s']
to_do_list_5 = ['attr1_s', 'sinc1_s', 'intel1_s', 'fun1_s', 'amb1_s', 'shar1_s']
to_do_list_6 = [['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1'], 
['attr1_2', 'sinc1_2', 'intel1_2', 'fun1_2', 'amb1_2', 'shar1_2'],
['attr1_3', 'sinc1_3', 'intel1_3', 'fun1_3', 'amb1_3', 'shar1_3'],
['attr2_1', 'sinc2_1', 'intel2_1', 'fun2_1', 'amb2_1', 'shar2_1']]
to_do_list_7 = [['attr7_2', 'sinc7_2', 'intel7_2', 'fun7_2', 'amb7_2', 'shar7_2'], 
['attr7_3', 'sinc7_3', 'intel7_3', 'fun7_3', 'amb7_3', 'shar7_3']]
################################################
################################################
actual_decisions = []
for i in range(2, 4):
	for j in features_of_attraction:
		actual_decisions.append(j + '7_' + str(i))
################################################
################################################
feature_space = ['iid', 'gender', 'race', 'field_cd', 'career_c', 'zipcode', 'goal', 'met', 'go_out', 'date', 'age', 'imprace', 'imprelig', 'like', 'prob', 'exphappy'] + clean_up_1[0:6] + clean_up_1[18:] + clean_up_2[0:5] + features_of_attraction + interests + preferences_of_attraction + rating_by_partner_features + ['age_o', 'like_o', 'prob_o', 'int_corr', 'race_o', 'samerace', 'pid', 'order', 'met_o']
all_space = feature_space + ['dec', 'dec_o', 'match']
################################################
################################################
list_of_lists = clean_up_1 + clean_up_2 + clean_up_3 + clean_up_4 + clean_up_5 + clean_up_6 + halfway_questions + actual_decisions + features_of_attraction + rating_by_partner_features + preferences_of_attraction + ['like_o', 'prob_o', 'imprace', 'imprelig', 'like', 'prob', 'exphappy', 'expnum', 'match_es', 'satis_2', 'you_call', 'them_cal', 'numdat_3', 'num_in_3'] + interests
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
	print 'clean_up_6', '\n', clean_up_6, '\n'
	print 'features_of_attraction', '\n', features_of_attraction, '\n'
	print 'preferences_of_attraction', '\n', preferences_of_attraction, '\n'
	print 'actual_decisions', '\n', actual_decisions, '\n'
	print 'rating_by_partner_features', '\n', rating_by_partner_features, '\n'
	print 'halfway_questions', '\n', halfway_questions, '\n'
	print 'interests', '\n', interests, '\n'
	print 'to_do_list', '\n', to_do_list, '\n'
	print 'feature_space', '\n', feature_space, '\n'
	print 'all_space', '\n', all_space, '\n'
	print 'list_of_lists', '\n', list_of_lists, '\n'
	for i in to_do_list:
		print i
################################################
################################################
def convert_income_to_float(data):
	income_lister = np.array([float(''.join(str(i).split(','))) for i in data['income']])
	income_lister = pd.DataFrame(data = income_lister, columns = ['income'])
	return data.update(income_lister)
################################################
################################################
def convert_tuition_to_float(data):
	tuition_lister = np.array([float(''.join(str(i).split(','))) for i in data['tuition']])
	tuition_lister = pd.DataFrame(data = tuition_lister, columns = ['tuition'])
	return data.update(tuition_lister)
################################################
################################################
def sat_to_float(data):
	sat_lister = np.array([float(''.join(str(i).split(','))) for i in data['mn_sat']])
	sat_lister = pd.DataFrame(data = sat_lister, columns = ['mn_sat'])
	return data.update(sat_lister)
################################################
################################################
def zipcode_to_float(data):
	zipcode_lister = np.array([float(''.join(str(i).split(','))) for i in data['zipcode']])
	zipcode_lister = pd.DataFrame(data = zipcode_lister, columns = ['zipcode'])
	return data.update(zipcode_lister)
################################################
################################################
def likert_scale_question_3(data):
	for i in clean_up_2:
		data[i].replace(to_replace = 12.0, value = 10.0, inplace = True)
################################################
################################################
def scale_question_3(data):
	for i in to_do_list:
		summation = data[i].sum(axis = 1).copy()
		for j in i:
			data[j] /= summation
			data[j] *= 100
################################################
################################################
def scale_question_4(data):
	for i in to_do_list_2:
		summation = data[i].sum(axis = 1).copy()
		for j in i:
			data[j] /= summation
			data[j] *= 100
################################################
################################################
def scale_question_5(data):
	for i in to_do_list_3:
		summation = data[i].sum(axis = 1).copy()
		for j in i:
			data[j] /= summation
			data[j] *= 100
################################################
################################################
def scale_question_1(data):
	for i in to_do_list_6:
		summation = data[i].sum(axis = 1).copy()
		for j in i:
			data[j] /= summation
			data[j] *= 100
################################################
################################################
def scale_question_2(data):
	summation = data[clean_up_5].sum(axis = 1).copy()
	for j in clean_up_5:
		data[j] /= summation
		data[j] *= 100
################################################
################################################
def scale_question_7(data):
	for i in to_do_list_7:
		summation = data[i].sum(axis = 1).copy()
		for j in i:
			data[j] /= summation
			data[j] *= 100
################################################
################################################
def scale_rating_received(data):
	summation = data[rating_by_partner_features].sum(axis = 1).copy()
	for j in rating_by_partner_features:
		data[j] /= summation
		data[j] *= 100

################################################
################################################
def scale_rating_given(data):
	summation = data[features_of_attraction].sum(axis = 1).copy()
	for j in features_of_attraction:
		data[j] /= summation
		data[j] *= 100
################################################
################################################
def scale_half_way(data):
	summation = data[to_do_list_4].sum(axis = 1).copy()
	for j in to_do_list_4:
		data[j] /= summation
		data[j] *= 100
################################################
################################################
def scale_half_way_2(data):
	summation = data[to_do_list_5].sum(axis = 1).copy()
	for j in to_do_list_5:
		data[j] /= summation
		data[j] *= 100
################################################
################################################
def scale_age(data):
	data['age'] = (data['age'] - data['age'].min()) / (data['age'].max() - data['age'].min())
################################################
################################################
def scale_majority_of_features(data):
	for i in list_of_lists:
		data[i] = (data[i] - data[i].min()) / (data[i].max() - data[i].min())
################################################
################################################
def scale_exphappy(data):
	data['exphappy'] = (data['exphappy'] - data['exphappy'].min()) / (data['exphappy'].max() - data['exphappy'].min())
################################################
################################################
def dating_attributes_vs_time_describe(data, gender):
	for i, j in data_cleaner.iteritems():
		stuff = pd.DataFrame(data = data.drop_duplicates(subset = 'iid', keep = 'first'), columns = ['iid', 'wave', 'gender'] + j)
		new_frame = stuff[stuff['gender'] == gender].copy()
		new_frame.drop(labels = ['iid', 'gender', 'wave'], axis = 1, inplace = True)
		display(new_frame.describe())
################################################
################################################
def dating_attributes_vs_time_hist(data, gender):
	for i, j in master_list.iteritems():
		stuff = pd.DataFrame(data = data.drop_duplicates(subset = 'iid', keep = 'first'), columns = ['iid', 'wave', 'gender'] + j)
		new_frame = stuff[stuff['gender'] == gender].copy()
		new_frame.drop(labels = ['iid', 'gender', 'wave'], axis = 1, inplace = True)
		new_frame.plot(kind = 'hist', stacked = True, bins = 10)
################################################
################################################
def count_samples_in_features(data):
	local_list = [(j, i) for i, j in zip(data.keys(), data.count())]
	local_list.sort()
	local_list.reverse()
	for i in local_list:
		print i[1], i[0], '|||',
################################################
################################################
def make_corr(data):
	sns.set(style="white")
	corr = data.corr()
	mask = np.zeros_like(corr, dtype = np.bool)
	mask[np.triu_indices_from(mask)] = True
	f, ax = plt.subplots(figsize = (22, 18))
	cmap = sns.diverging_palette(255, 140, as_cmap = True)
	sns.heatmap(corr, mask = mask, cmap = cmap, vmax = .3, 
		square = True, xticklabels = 5, yticklabels = 5, linewidths = 1, cbar_kws = {"shrink": .5}, ax = ax)
################################################
################################################
def outlier_detection(data):
	index_count = {}
	for feature in data[feature_space[10:72]].keys():
		Q1, Q3 = data[feature].quantile(q = [.25, .75])
		step = 1.5 * (Q3 - Q1)
		for counter in data[~((data[feature] >= Q1 - step) & (data[feature] <= Q3 + step))].index.values:
			if counter not in index_count:
				index_count[counter] = 1
			else:
				index_count[counter] += 1
	repeated_values = {key:value for key, value in index_count.items() if value >= 2}
	list_of_tuple = [(j, i) for i, j in repeated_values.iteritems()]
	list_of_tuple.sort()
	list_of_tuple.reverse()
	holder = 0
	to_be_removed = []
	for i, j in repeated_values.iteritems():
		if j > 15:
			holder += 1
			to_be_removed.append(i)
			#print (i, j),
	#print str(holder), str(holder / 4771.0 *100), '%'
	return to_be_removed
	#print list_of_tuple, 'length of list is {}'.format(len(list_of_tuple))
	#return data.drop(data.index[outliers])
################################################
################################################
def forests(input_df, target_df):
	from sklearn.ensemble import ExtraTreesClassifier
	from sklearn.feature_selection import SelectFromModel
	clf = ExtraTreesClassifier(random_state = 0)
	clf = clf.fit(input_df, target_df)
	model = SelectFromModel(clf, prefit=True)
	input_df_new = model.transform(input_df)
	original_space = input_df.shape
	new_space_ETC = input_df_new.shape
	tuple_holder = [(j, i) for i, j in zip(feature_space, clf.feature_importances_)]
	tuple_holder.sort()
	tuple_holder.reverse()
	################################################
	################################################
	from sklearn.ensemble import RandomForestClassifier
	clf = RandomForestClassifier(random_state = 0)
	clf = clf.fit(input_df, target_df)
	model = SelectFromModel(clf, prefit=True)
	input_df_new = model.transform(input_df)
	new_space_RFC = input_df_new.shape
	tuple_holder_2 = [(j, i) for i, j in zip(feature_space, clf.feature_importances_)]
	tuple_holder_2.sort()
	tuple_holder_2.reverse()
	################################################
	################################################
	rank_number = 0
	print 'ExtraTreesClassifier', '\t'*4, 'RandomForestClassifier'
	print 'Old Space: ', original_space, '\t'*4, 'Old Space:', original_space
	print 'New Space: ', new_space_ETC, '\t'*4, 'New Space:', new_space_RFC
	for i, j in zip(tuple_holder, tuple_holder_2):
		rank_number += 1
		print rank_number, '|', i, '\t'*3, rank_number, '|', j
################################################
################################################
def f_classifier_selection(input_df, target_df):
	from sklearn.feature_selection import SelectKBest
	from sklearn.feature_selection import f_classif
	kBest = SelectKBest(f_classif, k = 'all')
	kBest.fit_transform(input_df, target_df)
	k_Best_features = [(j, i, k) for i, j, k in zip(input_df.keys(), kBest.scores_, kBest.pvalues_)]
	k_Best_features.sort()
	k_Best_features.reverse()
	counter = 0
	print 'SelectKBest: f_classif'
	for i in k_Best_features:
		counter += 1
		print counter, i
################################################
################################################
def chi_feature_selection(new_input_df, target_df):
	from sklearn.feature_selection import chi2
	from sklearn.feature_selection import SelectKBest
	kBest = SelectKBest(chi2, k = 'all')
	kBest.fit_transform(new_input_df, target_df)
	k_Best_features = [(j, i, k) for i, j, k in zip(new_input_df, kBest.scores_, kBest.pvalues_)]
	k_Best_features.sort()
	k_Best_features.reverse()
	counter = 0
	print 'SelectKBest: chi2'
	for i in k_Best_features:
		counter += 1
		print counter, i
################################################
################################################
def explained_ratio_pca(new_input_df):
	from sklearn.decomposition import PCA
	total_ratio = []
	for i in range(1, 9):
		pca = PCA(n_components = i)
		pca.fit(new_input_df)
		#print 'n = ', str(i), ', ', 'Total Ratio: ', pca.explained_variance_ratio_.sum()
		#print 'Components:', pca.components_
		#print 'Explained Variance:', pca.explained_variance_ 
		#print 'Explained Ratio: ', pca.explained_variance_ratio_
		total_ratio.append(pca.explained_variance_ratio_.sum())

	plt.figure(figsize = (20, 15))
	plt.subplot(221)
	#plt.bar(range(1,9), total_ratio, align='center', alpha = 1)
	plt.plot(range(1,9), total_ratio, alpha = 1)
	plt.xlabel('Projections')
	plt.ylabel('Total Explained Ratio')
	plt.title('Total Explained Ratio vs. Projections')

	plt.show()
	counter = 1
	for i in total_ratio:
		print 'n = ', counter, ', ', i
		counter += 1		
################################################
################################################
def pca_plotter(transformed_pca, target_df):
	new_target_df = pd.DataFrame(target_df, columns = ['dec'])
	new_target_df.reset_index(inplace = True)
	new_target_df.drop('index', axis = 1, inplace = True)

	no_decision = new_target_df[new_target_df['dec'] == 0].index
	yes_decision = new_target_df[new_target_df['dec'] == 1].index

	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	ax.scatter(transformed_pca.loc[no_decision]['x_s'], transformed_pca.loc[no_decision]['y_s'], transformed_pca.loc[no_decision]['z_s'], c = 'blue')
	ax.scatter(transformed_pca.loc[yes_decision]['x_s'], transformed_pca.loc[yes_decision]['y_s'], transformed_pca.loc[yes_decision]['z_s'], c = 'red')
	ax.set_xlabel('pca_1')
	ax.set_ylabel('pca_2')
	ax.set_zlabel('pca_3')
	#for ii in xrange(0,360,5):
	#	ax.view_init(elev=ii, azim=10)
	#	plt.savefig("movie_2%d.png" % ii)
	#	plt.draw()
	plt.show()

	plt.scatter(transformed_pca.loc[no_decision]['x_s'], transformed_pca.loc[no_decision]['y_s'], color = 'blue')
	plt.scatter(transformed_pca.loc[yes_decision]['x_s'], transformed_pca.loc[yes_decision]['y_s'], color = 'red')
	ax.set_xlabel('pca_1')
	ax.set_ylabel('pca_2')
	plt.show()

	plt.scatter(transformed_pca.loc[no_decision]['x_s'], transformed_pca.loc[no_decision]['z_s'], color = 'blue')
	plt.scatter(transformed_pca.loc[yes_decision]['x_s'], transformed_pca.loc[yes_decision]['z_s'], color = 'red')
	ax.set_xlabel('pca_1')
	ax.set_ylabel('pca_3')
	plt.show()

	plt.scatter(transformed_pca.loc[no_decision]['y_s'], transformed_pca.loc[no_decision]['z_s'], color = 'blue')
	plt.scatter(transformed_pca.loc[yes_decision]['y_s'], transformed_pca.loc[yes_decision]['z_s'], color = 'red')
	ax.set_xlabel('pca_2')
	ax.set_ylabel('pca_3')

	plt.show()
################################################
################################################
def apply_nearest_neighbor(transformed_pca, target_df):
	from sklearn.decomposition import PCA
	from sklearn.grid_search import GridSearchCV
	from sklearn.metrics import make_scorer
	from sklearn import cross_validation
	from sklearn.metrics import f1_score
	from sklearn import neighbors
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.metrics import accuracy_score, classification_report

	num_train = int(.75 * transformed_pca.shape[0])
	num_test = int(transformed_pca.shape[0] - num_train)

	print 'Training set size: ', num_train, 'Test set size: ', num_test

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(transformed_pca, target_df, test_size = num_test, random_state = 0)
	f1_scorer = make_scorer(f1_score, pos_label = 1)
	parameters = [{'n_neighbors' : range(1,101)}]
	clf = neighbors.KNeighborsClassifier()
	clf = GridSearchCV(estimator = clf, param_grid = parameters, scoring = f1_scorer, cv = 10)
	clf.fit(X_train, y_train)

	print 'Best f1_score: ', clf.best_score_ 
	print 'With neighboors at: ', clf.best_params_
	print 'Best Estimator: ', clf.best_estimator_

	y_pred = clf.predict(X_test)
	print 'Test Size: ', X_test.shape[0]
	print 'Accuracy:' , accuracy_score(y_test, y_pred)
	print classification_report(y_test, y_pred)
################################################
################################################
def linear_classifier(input_df, target_df):
	import numpy as np
	from sklearn import svm, cross_validation
	from sklearn.grid_search import GridSearchCV
	from sklearn.metrics import accuracy_score, classification_report, f1_score, make_scorer

	#display(input_df)
	num_train = int(.75 * input_df.shape[0])
	num_test = int(input_df.shape[0] - num_train)
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(input_df, target_df, test_size = num_test, random_state = 0)

	svc = svm.SVC(kernel = 'linear', random_state = 0)
	f1_scorer = make_scorer(f1_score, pos_label = 1)
	parameters = [{'C' : np.logspace(-2, 2, num = 10)}]
	print parameters
	svc = GridSearchCV(estimator = svc, param_grid = parameters, scoring = f1_scorer, cv = 10)
	svc.fit(X_train, y_train)

	print 'Best f1_score: ', svc.best_score_ 
	print 'Best C: ', svc.best_params_
	print 'Best Estimator: ', svc.best_estimator_

	C = svc.best_params_['C']

	print 'Retrain on Best C: ', str(C)
	svc = svm.SVC(kernel = 'linear', C = C, random_state = 0)
	svc.fit(X_train, y_train)

	print 'Coefficients: ', svc.coef_
	print 'Intercept: ', svc.intercept_

	y_pred = svc.predict(X_test)
	print 'Test Size: ', X_test.shape[0]
	print 'Accuracy:' , accuracy_score(y_test, y_pred)
	print classification_report(y_test, y_pred)