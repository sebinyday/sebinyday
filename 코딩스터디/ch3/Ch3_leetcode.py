#20번 . 괄호 검사 알고리즘 

from collections import deque


class Solution(object):
    def isValid(self, s):
        stack = []
        bracket_map = {')': '(', '}': '{', ']': '['}

        for char in s:
            if char in bracket_map:  # 닫는 괄호이면 
                top_element = stack.pop() if stack else '#'  #top_element는 스택 젤 위에거. 
                if bracket_map[char] != top_element:  
                    return False
            else:  
                stack.append(char)  #여는 괄호이면 해당 문자를 스택에 추가

        return not stack

#225번 큐2개로 스택....아직 이해 못함.....
    
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
        