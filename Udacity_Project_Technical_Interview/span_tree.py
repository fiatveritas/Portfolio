#!/usr/bin/python2.7.11

def question4(T, r, n1, n2):
	"""This script finds the least common ancestor between\
    two nodes on a binary search tree."""
	if len(T) == 0:
		return T
	elif len(T) == 1:
		return r
	index_holder = None
	parent1 = []
	while index_holder != r:
		for a in range(len(T)):
			if T[a][n1] == 1:
				index_holder = a
				n1 = index_holder
				parent1.append(index_holder)
	other_index = None
	parent2 = []
	while other_index != r:
		for a in range(len(T)):
			if T[a][n2] == 1:
				other_index = a
				n2 = other_index
				parent2.append(other_index)
	for a1 in parent1:
		for a2 in parent2:
			if a1 == a2:
				return a1


print 'Test Cases for Question 4:'

print question4([],
                None,
                None,
                None)
# Should print []
print question4([[0]],
                0,
                0,
                0)
# Should print 0
print question4([[0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [1, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0]],
                3,
                1,
                4)
# Should print 3
print question4([[0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 1, 0, 0],
                 [1, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0]],
                1,
                0,
                6)
# Should print 2