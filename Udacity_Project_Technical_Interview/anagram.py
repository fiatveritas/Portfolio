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