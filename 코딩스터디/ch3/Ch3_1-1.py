class ThreeStacks:
    def __init__(self, stack_size):
        self.array = [None] * (stack_size * 3)  # None ������ ä���� ���̰� stack_size * 3�� ����Ʈ�� ����
        self.sizes = [0] * 3  # �� ������ ���� ũ��
        self.stack_size = stack_size    # �� ������ �ִ� ũ�� 

    def push(self, stack_num, value):   # stack_num�� ������ ��ȣ 0,1,2�� �ϳ�
        if self.sizes[stack_num] < self.stack_size:     #������ ������ �ʾ�����
            self.array[self.index_of_top(stack_num)] = value    #������ ������ top ��ġ�� �� �߰�
            self.sizes[stack_num] += 1  # ������ ũ�� 1 �߰� 
        else:
            raise Exception("Stack is full")    #������ �� á��. 

    def pop(self, stack_num):
        if self.sizes[stack_num] == 0:      #������ ���������
            raise Exception("Stack is empty")   #���� �߻���Ŵ
        value = self.array[self.index_of_top(stack_num) - 1]    #������ top���� ���� �����ϰ� �װ� value�� ����
        self.array[self.index_of_top(stack_num) - 1] = None     #���ŵ� ��ġ�� None ����
        self.sizes[stack_num] -= 1      #������ ũ�� 1 ���� 
        return value    #���ŵ� �� ��ȯ 

    def index_of_top(self, stack_num):
        offset = stack_num * self.stack_size
        return offset + self.sizes[stack_num]

# ��� ����
stacks = ThreeStacks(5)
stacks.push(0, 1)  # ���� 1�� 1 �߰�
stacks.push(1, 3)  # ���� 2�� 3 �߰�
stacks.push(2, 5)  # ���� 3�� 5 �߰�
print(stacks.pop(0))  # ���� 1���� pop, 1 ��ȯ
