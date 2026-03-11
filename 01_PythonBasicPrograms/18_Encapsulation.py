# Encapsulation = encapsulate data variables and methods in a single unit,
# Encapsulation using getters and setters

# Access modifiers: public, protected, private variables

class Person:
    def __init__(self,name, age):
        self.name=name #public var
        self.age=age #public var

person=Person("krish",25)
print(person.name)
print(person.age)

print(dir(person)) ## u see last 2 items as name and age as they are public vars
for item in dir(person):
    print(item)

##PRivate variables - using double underscore
class Person1:
    def __init__(self,name, age, gender):
        self.__name=name # private var
        self.__age=age # private var
        self.gender=gender # publci var

    def get_name(person):
        return person.__name

person1=Person1("krish",25, gender='male')
print(dir(person1)) ## u WON'T see last 2 items as name and age as they are private vars, you'll see gender listed as it is a public var
for item in dir(person1):
    print(item)

person1.get_name() ## this will not be accessible because name is private var
# person.__name cannot be accessed outside teh class, even derived classes cannot access

# Protected variable - accessible by child class but not from other classes
# Protected variables - using single underscore
class Person2:
    def __init__(self,name, age, gender):
        self._name=name # protected var
        self._age=age # protected var
        self.gender=gender # public var

class Employee(Person2):
    def __init__(self, name, age, gender):
        super().__init__(name,age,gender)

employee=Employee("KRish",22,"male")
print(employee._name) #protected variable is accessible in derived class

