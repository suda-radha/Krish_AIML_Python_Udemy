## Magic methods are dunder methods(double underscore methods)
#They show up when u run print(dir(classname))
#Ex: __init__ , __str__ etc

#Basic magic methods
class Person:
    def __init__(self,name,age):
        self.name=name
        self.age=age
person= Person("KrisH",22)
print(person) # output = <__main__.Person object at 0x000001E79B467230>
# let us override this method's default display above which is from __str__ method

#Basic magic methods with overriding __str__
class Person1:
    def __init__(self,name,age):
        self.name=name
        self.age=age

    def __str__(self):
        return f"{self.name}, {self.age} years old"
person1= Person1("KrisH",22)
print(person1) # now the o/p is KrisH, 22 years old

# likewise you can override all default magic methods