''' Demonstration of super().__init__() functionality'''

'''
When you call super().__init__() in the child class, it executes the parent class's __init__() method,
which allows the child class to inherit and use the attributes of parent class.

COMMENT IN & OUT THE super().__init__() to see what happens!
'''

class Parent:
    def __init__(self):
        self.arg1 = 'arg1'
        self.arg2 = 'arg2'
        self.arg2 = 'arg2'

    def combine(self, first, second):
        return first + second


class Child(Parent):
    def __init__(self):
        # super().__init__()
        self.child_arg1 = 'child_arg1'
        self.child_arg2 = 'child_arg2'

    def child_combine(self):
        argu1 = self.arg1
        return argu1
    
    def use_combine_parent(self):
        return self.combine(self.child_arg1, self.child_arg2)
    
child = Child()

print(child.child_combine()) # Works only with super().__init__() !!!!!
print(child.use_combine_parent()) # Works with and without super().__init__()

