import pandas as pd

data = pd.read_csv('Expenses_2016.csv')

number_of_purchases_in_day = {}

for i in data['Date']:
	if i not in number_of_purchases_in_day:
		number_of_purchases_in_day[i] = 1
	else:
		number_of_purchases_in_day[i] += 1
for key in number_of_purchases_in_day:
	print key, number_of_purchases_in_day[key]