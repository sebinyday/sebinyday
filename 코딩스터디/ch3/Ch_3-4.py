class MyQueue:
    def __init__(self):
        self.input_stack = []
        self.output_stack = []

    def enqueue(self, x):
        self.input_stack.append(x)

    def dequeue(self):
        self.move_input_to_output()
        return self.output_stack.pop()

    def peek(self):
        self.move_input_to_output()
        return self.output_stack[-1]

    def empty(self):
        return not self.input_stack and not self.output_stack

    def move_input_to_output(self):
        if not self.output_stack:
            while self.input_stack:
                self.output_stack.append(self.input_stack.pop())
