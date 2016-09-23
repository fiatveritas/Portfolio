#!/usr/bin/python2.7

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

    return ll