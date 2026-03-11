#Encapsulation with getters and setters
class Person:
    def __init__(self, name, age):
        self.__name=name ## Private var
        self.__age=age ## Private var

    ## getters can access private vars using self kw
    def get_name(self):
        return self.__name

    def get_age(self):
        return self.__age

    ## setter methods to change values of private vars
    def set_name(self, name):
        self.__name=name

    def set_age(self, age):
        if age> 0:
            self.__age=age
        else:
            print("Age cannot be negative")

#Access these private vars from same class using getters and setters
person =Person("Krish", 34)
print(person.get_name())
print(person.get_age())

person.set_name("Ram")
person.set_age(33)
print(person.get_name())
print(person.get_age())

person.set_age(-3)






