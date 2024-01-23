class ThreeStacks:
    def __init__(self, stack_size):
        self.array = [None] * (stack_size * 3)  # None 값으로 채워진 길이가 stack_size * 3인 리스트를 생성
        self.sizes = [0] * 3  # 각 스택의 현재 크기
        self.stack_size = stack_size    # 각 스택의 최대 크기 

    def push(self, stack_num, value):   # stack_num은 스택의 번호 0,1,2중 하나
        if self.sizes[stack_num] < self.stack_size:     #스택이 꽉차지 않았으면
            self.array[self.index_of_top(stack_num)] = value    #선택한 스택의 top 위치에 값 추가
            self.sizes[stack_num] += 1  # 스택의 크기 1 추가 
        else:
            raise Exception("Stack is full")    #스택이 꽉 찼음. 

    def pop(self, stack_num):
        if self.sizes[stack_num] == 0:      #스택이 비어있으면
            raise Exception("Stack is empty")   #예외 발생시킴
        value = self.array[self.index_of_top(stack_num) - 1]    #스택의 top에서 값을 제거하고 그걸 value에 저장
        self.array[self.index_of_top(stack_num) - 1] = None     #제거된 위치를 None 설정
        self.sizes[stack_num] -= 1      #스택의 크기 1 감소 
        return value    #제거된 값 반환 

    def index_of_top(self, stack_num):
        offset = stack_num * self.stack_size
        return offset + self.sizes[stack_num]

# 사용 예시
stacks = ThreeStacks(5)
stacks.push(0, 1)  # 스택 1에 1 추가
stacks.push(1, 3)  # 스택 2에 3 추가
stacks.push(2, 5)  # 스택 3에 5 추가
print(stacks.pop(0))  # 스택 1에서 pop, 1 반환
