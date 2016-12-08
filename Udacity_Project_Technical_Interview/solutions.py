#!/usr/bin/python2.7

def get_dict(string):
    primes_list = [5, 71, 79, 19, 2, 83, 31, 43, 11, 53, 37, 23, 41,
                3, 13, 73, 101, 17, 29, 7, 59, 47, 61, 97, 89, 67]#generate prime numbers for coding letters
    alphabet = [chr(i) for i in range(ord('a'), ord('z')+1)]
    dict_holder = {}
    for i, letter in enumerate(alphabet):
        dict_holder[letter] = primes_list[i]
    dict_length = 1
    for letter in list(string):
        dict_length *= dict_holder[letter]
    return dict_length


def question1(s, t):
    "this script takes in string, a, and checks if a substring, t, is an anagram\
    of the given string"
    if s == t:
        return True
    elif s == '' or t == '':
        return False
    else:
        t_dict_length = get_dict(t)
        for i in range(len(s) - len(t)):
            s_dict_length = get_dict(s[i:i+len(t)])
            if s_dict_length == t_dict_length:
                return True
        i += 1
        s_dict_length = get_dict(s[i:])
        if s_dict_length == t_dict_length:
            return True

        return False

print '################################'

print 'Test Cases for Question 1:'

print question1('', '')
# Prints True

print question1('', 'a')
# Prints False

print question1('a', '')
# Prints False

print question1('udacity', 'udacity')
# Prints True

print question1('udacity', 'ad')
# Prints True

print question1('abcdefghijklmnopqrstuvwxyz', 'udacity')
# Prints False

print question1('udacity', 'icty')
# Prints True

print '################################'

def question2(a):
    """This script finds the longest palindromic substring contained in a"""
    if len(a) <= 1:
        return 'no palindrome exists'

    max_len = ""
    for i in range(len(a)):
        for j in range(0, i):
            stringer = a[j:i+1]
            if stringer == stringer[::-1]:
                if len(stringer) > len(max_len):
                    max_len = stringer
    if len(max_len) == 0:
        return 'no palindrome exists'
    return max_len

print '################################'

print 'Test Cases for Question 2:'

print question2('')
#Prints 'no palindrome'
print question2('a')
#Print 'no palindrome'

print question2('jesse')
#Prints esse

print question2('112311')
#Print 11

print '################################'

def question3(G):
    """This script find the minimum \
    spanning tree that all vertices in a graph with the \
    smallest possible weight of edges."""

    if len(G) < 1:
        return G
    nodes = set(G.keys())
    min_span_tree = {}
    origin = G.keys()[0]
    min_span_tree[origin] = []

    while len(min_span_tree.keys()) < len(nodes):
        low_weight = float('inf')
        low_edge = None
        for node in min_span_tree.keys():
            edges = [(weight, vertex) for (vertex, weight) in G[node] if vertex not in min_span_tree.keys()]
            if len(edges) > 0:
                w, v = min(edges)
                if w < low_weight:
                    low_weight = w
                    low_edge = (node, v)
        min_span_tree[low_edge[0]].append((low_edge[1], low_weight))
        min_span_tree[low_edge[1]] = [(low_edge[0], low_weight)]
    return min_span_tree

print '################################'

print 'Test Cases for Question 3:'

print question3({})
# Should print {}

print question3({'A': []})
# Should print {'A': []}

print question3({'A': [('B', 2)], 'B': [('A', 2), ('C', 5)], 'C': [('B', 5)]})
#Shoud print {'A': [('B', 2)],
# 'B': [('A', 2), ('C', 5)], 
# 'C': [('B', 5)]}

print question3({'A': [('B', 3), ('E', 1)],
                 'B': [('A', 3), ('C', 9), ('D', 2), ('E', 2)],
                 'C': [('B', 9), ('D', 3), ('E', 7)],
                 'D': [('B', 2), ('C', 3)],
                 'E': [('A', 1), ('B', 2), ('C', 7)]})
# Should print
# {'A': [('E', 1)],
#  'C': [('D', 3)],
#  'B': [('E', 2), ('D', 2)],
#  'E': [('A', 1), ('B', 2)],
#  'D': [('B', 2), ('C', 3)]}

print '################################'

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

print '################################'

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

print '################################'

class Node(object):
    def __init__(self, data):
        self.data = data
        self.next = None


def question5(ll, m):

    """This script returns an element in a singly linked list that's 'm' elements from the end of a string 'a'."""

    if ll == None or m < 1: #check whether the linked list is empty or mth value less than one. If so, script terminates.
        return None

    last = ll # set last as the counter that will go to the mth location in following while loop
              # the ll and last nodes will be m positions away from each other

    while last != None and m > 0: # checks if last arrived at the end of linked or if m is out of bound
        last = last.next
        m -= 1

    if m != 0:
        return None

    while last != None: #this moves the pair of pointers until last arrives at last node
        last = last.next # moreover, ll get positioned where last was set
        ll = ll.next

    return ll.data

print '################################'


print 'Test Cases for Question 5:'

junk_0 = Node(None)
print question5(junk_0, 1)
# Should print None

junk_0 = Node(0)
print question5(junk_0, 4)
# Should print None

junk_0 = Node(0)
junk_1 = Node(1)
junk_2 = Node(2)
junk_3 = Node(3)
junk_4 = Node(4)

junk_0.next = junk_1
junk_1.next = junk_2
junk_2.next = junk_3
junk_3.next = junk_4

print question5(junk_0, 3)
#Should print 2

print '################################'
print '################################'