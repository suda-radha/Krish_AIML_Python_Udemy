# operators can be overloaded in python just like magic methods __str__
# if you run dir(class) you see list of all magic methods, it that you see operator methods as well
# operator methods like: __eq__, __ge__, __gt__, __add__, __Sub__, __mul__

class Vector:
    def __init__(self,x, y):
        self.x=x
        self.y=y

    def __add__(self, other): ## we are trying to add vectors
        return Vector(self.x+other.x, self.y+other.y)

    def __sub__(self, other):  ## we are trying to sub vectors
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, other):  ## we are trying to multiply vectors
        return Vector(self.x * other.x, self.y * other.y)

    def __eq__(self, other): ## checking if 2 vectors are equal
        return self.x == other.x and self.y == other.y

    def __repr__(self): ## overriding how you want teh display of Vector call if someone tries to print
        return f"Vector({self.x}, {self.y})"

## create objects of vector class
v1=Vector(2,3)
v2=Vector(3,4)
v3=Vector(2,3)
print(v1+v2)
print(v1-v2)
print(v1*v2)
print(v1==v2)
print(v1==v3)

## output of above program is
"""
Vector(5, 7)
Vector(-1, -1)
Vector(6, 12)
False
True
"""


