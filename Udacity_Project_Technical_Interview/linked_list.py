class Node(object):
    def __init__(self, data):
        self.data = data
        self.next = None


def question5(ll, m):
    """This script returns an element in a singly linked list that's 'm' elements from the end of a string 'a'."""
    if ll:
        counter = 1
        node = ll
        while node.next:
            node = node.next
            counter += 1
        if m < counter:
            kounter = counter - m
            i = 0
            node = ll
            while i < kounter:
                node = node.next
                i += 1
        else:
            return None
    else:
        node = ll
    return node.data

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
# Should print 2