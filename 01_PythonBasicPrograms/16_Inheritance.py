# note python supports single and multiple inheritance
#parent class Car
class Car:
    def __init__(self, windows, doors, enginetype):
        self.windows=windows
        self.doors=doors
        self.enginetype=enginetype

    def drive(self):
        print(f"The person will drive the {self.enginetype} car ")

#object
car1=Car(4,5,"petrol")
car1.drive()


#child class Tesla inherits Car
class Tesla(Car):# Tesla(Car) ==> Tesla inherits from parent Car
    def __init__(self,windows, doors, engineType,is_selfdriving):
        super().__init__(windows,doors,engineType)
        self.is_selfdriving=is_selfdriving

    def selfdriving(self):
        print(f"Tesla supports self driving: {self.is_selfdriving}")

#object
tesla1=Tesla(4,5,"electric",True)
tesla1.selfdriving() # calling child class method
tesla1.drive() # calling parent class method

#Multiple inheritance: 1 child inherits from multiple parents classes
#Base class1
class Animal:
    def __init__(self,name):
        self.name=name
    def speak(self):
        print("Subclass must inherit this speak() method")

#Base Class2
class Pet:
    def __init__(self, owner):
        self.owner=owner

#Derived class or child class
class Dog(Animal, Pet):#Dog inherist from 2 parents Animal and Dog
    def __init__(self, name,owner):
        Animal.__init__(self,name)
        Pet.__init__(self,owner)
    def speak(self):
        return f"{self.name} says woof woof! "

#Object
dog=Dog("Buddy", "Krish")
print(dog.speak())
print(dog.owner)