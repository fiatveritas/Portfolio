#!/usr/bin/python2.7

import re

def question2(a):
    """This script finds the longest palindromic substring contained in a"""

    if check_backwards(a):
        return "True, \"{}\" \nis a palindrome".format(a)
    
    else:
        word = ''
        reversed_word = ''
        palindrome = ''
        list_of_palindrome = []

        holder = re.split('W+', a.lower())
        word = word.join(holder)
        #print word

        reversed_holder = [ i for i in reversed(word)]
        reversed_word = reversed_word.join(reversed_holder)
        #print reversed_word
        
        start = 0
        end = len(word) - 1
        while start <= len(word) - 1 and end >= 0:
            for letter in reversed_word:
                if word[start] == letter:
                    palindrome += letter
                    start += 1
                else:
                    list_of_palindrome.append(palindrome)
                    palindrome = ''
                

    
def check_backwards(a):
    """This function checks if the word\
    is already a palindrome."""
    #print a

    word = ''
    reversed_word = ''

    holder = re.split('\W+', a.lower())
    #print holder

    word = word.join(holder)
    #print word

    reversed_holder = [ i for i in reversed(word) ]
    #print reversed_holder

    reversed_word = reversed_word.join(reversed(word))
    #print reversed_word

    if reversed_word == word:
        return True
    else:
        return False

####Test Cases
print question2('Jesse')
#print question2('Anna')
#print question2('Are we not drawn onward, we few, drawn onward to new era?')
#print question2('No demerits tire me, Don.')
#print question2('112311')
#print check_backwards('Anna')
#print check_backwards('Kayak')
#print check_backwards('Are we not drawn onward, we few, drawn onward to new era?')
#print check_backwards("A but tuba.")
#print check_backwards('Jesse')
