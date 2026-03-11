# polymorphism - same method - implemented in different classes (like child classes and ABC concept as well - see below ) parent class is ABC with pass
#Child methods overrides parent methods (with same method names) is also polymorphism  - this is polymorphism + inheritance
# multiple methods with same name and different parameters - wait and see if trainer talks about this - this is not there in PYTHON - SEE NOTES at the END
# polymorphism achieved through method overriding and interfaces


#Base class
class Animal:
    def speak(self):
        return "sound of the animal"
#global class method
def animal_speak(animal):
    print(animal.speak())

#Derived class 1
class Dog(Animal):
    def speak(self):
        return "woof!"
#Derived class 2
class Cat(Animal):
    def speak(self):
        return "meow!"

dog= Dog()
cat=Cat()
print(dog.speak())
print(cat.speak())
animal_speak(dog)
animal_speak(cat)

#polymorphism with Functions and methods

## Base class
class Shape:
    def area(self):
        return "The area of teh figure"
#Derived class 1
class Rectangle(Shape):
    def __init__(self,width, height):
        self.width=width
        self.height=height
    def area(self):
        return self.width*self.height

#Derived class 2
class Circle(Shape):
    def __init__(self, radius):
        self.radius=radius

    def area(self):
        return 3.14*self.radius*self.radius

#Function that demonstrates polymorphism

def print_area(shape):
    print(f"the area is {shape.area()}")

rectangle=Rectangle(4,5)
circle=Circle(3)

print_area(rectangle)
print_area(circle)

#polymorphism with ABC Abstract base classes(Interfaces)
from abc import ABC, abstractmethod
## Define an abstract class
class Vehicle(ABC):
    @abstractmethod
    def start_engine(self):
        pass

##Derived class 1
class Car(Vehicle):
    def start_engine(self):
        return "Car engine started"
#Derived class 2
class Motorcycle(Vehicle):
    def start_engine(self):
        return "Motorcycle engine started"

#Function to demo polymorphism
def start_vehicle(vehicle):
    print(vehicle.start_engine())
#Create object of car and motorcycle
car=Car()
motorcycle=Motorcycle()
start_vehicle(car)
start_vehicle(motorcycle)



