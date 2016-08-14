#!/usr/bin/python2.7

import re

def question2(a):
    """This script finds the longest palindromic substring contained in a"""
    word_holder = str(a)
    letter_lister = [ letter for letter in word_holder ]
    first_index = 0
    last_index = len(word_holder) - 1
    
    if len(word_holder) % 2 == 0:
    	even_bool = True
    elif len(word_holder) != 0:
    	even_bool = False
    return None

print question2('jesse')