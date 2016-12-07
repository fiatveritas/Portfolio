#!/usr/bin/python2.7

def question2(a):
    """This script finds the longest palindromic substring contained in a"""
    if len(a) <= 1:
        return a

    longest = ""
    for i in range(len(a)):
        for j in range(0, i):
            substring = a[j:i+1]
            if substring == substring[::-1]:
                if len(substring) > len(longest):
                    longest = substring
    if len(longest) == 0:
        return a[0]
    return longest

####Test Cases
#print question2('jesse')
#print question2('')
#print question2('a')
#print question2('112311')