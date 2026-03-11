# Lambda functions are simple anonymous functions with any number of arguments but
#should have only one expression
#Syntax ==== lambda args: expression

def add(a,b):
    return a+b

# same can be written as lanbda fn
sum = lambda a,b:a+b
print(sum(5,6))

#ex2 even fun
even=lambda a: a%2==0
print(even(24))

