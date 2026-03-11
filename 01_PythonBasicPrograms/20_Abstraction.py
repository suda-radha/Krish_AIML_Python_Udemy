# Abstraction - is hiding complex implementation details and
# showing only needed features of an object

# Example Washing machine- u see only buttons visible - start, stop, rinse, soak
# other Ex: mobile phone, ATM, kiosks

# achieved using ABC package and abstractmethod
# you inherit ABC class

from abc import ABC, abstractmethod

#Abstract Base class
class Vehicle(ABC):
    def drive(self):
        print("This vehicle is used for driving")

    @abstractmethod
    def start_engine(self):
        pass #abstract method will be empty only

# create a derived class and write your own implementation
class Car(Vehicle):
    def start_engine(self):
        print("Car engine started")

# define a method to start the engine outside this class
def operate_vehicle(vehicle):
    vehicle.start_engine() #implemented child method
    vehicle.drive() #parent method

#Calling class
car = Car()
operate_vehicle(car)
