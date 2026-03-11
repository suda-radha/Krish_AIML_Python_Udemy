# classes and objects
class car:
    pass

audi=car()
bmw = car()

audi.windows=4

print(type(audi)) # <class '__main__.car'>

#dir of any class --> very imp==> shows all methods /variables in that class
# its shows inbuilt ones as well as user defined ones

dir(audi) # not showing for me , it also shows __init__ method

#Constructor, instance variables and instance methods in a class

# __init__ method in classes (inbuilt method)
# this __init__ is constructor method of that class

class Dog:
    #constructor
    def __init__(self,name,age):
        self.name=name
        self.age=age

    #instance methods
    def bark(self):
        print(f"{self.name} says woof woof" )
#create object of dog
Dog1=Dog('buddy',3)
print(Dog1.name)
print(Dog1.age)

Dog1.bark()
