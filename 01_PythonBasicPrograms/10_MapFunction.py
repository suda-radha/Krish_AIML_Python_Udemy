# map() applies a function to all items in list
# map() fn syntax ==== map(fun, iterableList)
# we can combine lambda fn and map ==== map(lambda x:x**2, [1,2,3,4,5])
# always use the map fun inside a list() ull see teh magic of it
from win32comext.shell.shellcon import EXP_LOGO3_ID_SIG

squares = map(lambda x:x**2, [1,2,3,4,5])
for sq in squares:
    print(sq)

# always wrap map() inside list()
squares2 = list(map(lambda x:x**2, [1,2,3,4,5]))
print(squares2) #[1, 4, 9, 16, 25]


#Ex 2
def square(x):
    return x**2
numbers = [9,8,7,6,5]
squares = map(square, numbers)
print(list(squares)) # always wrap map() with list()

#Ex3
# lambda fun with map
# syntax === map(lambda_fun, list)
print(list(map(lambda a,b: a+b, [1,2,3],[4,5,6])))

#Ex4
# using inbuilt fun int
nums_chars = ['1', '2', '3']
nums = list(map(int, nums_chars ))
print(nums)

#Ex5
# using inbuilt fun str.upper
fruits = ["apple", 'banana', "cherry"]
print(list(map(str.upper, fruits)))

#Ex6
# map on dictionary
people = [
    {"name":"Amy", "age": 23},
    {"name": "Ben", "age": 25}
]

def get_name(person):
    return person["name"]

# fun to get names of people
print(list(map(get_name, people)))

