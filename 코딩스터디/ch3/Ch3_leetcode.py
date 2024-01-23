#20�� . ��ȣ �˻� �˰��� 

from collections import deque


class Solution(object):
    def isValid(self, s):
        stack = []
        bracket_map = {')': '(', '}': '{', ']': '['}

        for char in s:
            if char in bracket_map:  # �ݴ� ��ȣ�̸� 
                top_element = stack.pop() if stack else '#'  #top_element�� ���� �� ������. 
                if bracket_map[char] != top_element:  
                    return False
            else:  
                stack.append(char)  #���� ��ȣ�̸� �ش� ���ڸ� ���ÿ� �߰�

        return not stack

#225�� ť2���� ����....���� ���� ����.....
    
class MyStack(object):

    def __init__(self):
        self.q = deque()

    def push(self, x):
        self.q.append(x)
        

    def pop(self):
        for i in range(len(self.q) - 1):
            self.push(self.q.popleft())
        return self.q.popleft()
        

    def top(self):
        return self.q[-1]
        

    def empty(self):
        return len(self.q) == 0
        