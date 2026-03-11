# list comprehension
# if you want some operation to be done on allelements in a list, u can do it in just one line of code instead of one full method
# 1 methods way
# say u want squares of all nums in list
lst=[]
for x in range(10):
    lst.append(x**2)
print(lst)

#2nd way is list comprehension way [operation for num in nums]
lst2= [x**2 for x in range(10)]
print(lst2)

### List comprehension syntax 3 things
# LC with basic syntax    [operation for-loop]
# LC with condition       [operation for-loop condition]
# nested LC               [operation for-loop forloop ]
# LC with function calls  [function  for-loop]
# you can also have else block

# squares of all nums 1-10
# even numbers of 1-10
# pairs of items from 2 diff list
# get length of words from list


squares = [num**2 for num in range(10)]
print(f"squares: {squares}")

even = [num for num in range(10) if(num%2==0)]
print(f"even: {even}")

lst1 = [1,2,3]
lst2 = ['a', 'b', 'c']
pair = [[l1, l2] for l1 in lst1 for l2 in lst2]
print(f"pair: {pair}")

words = ["hello", "world", "java", "comprehension", "alphabet"]
words_length = [len(word) for word in words]
print(f"words_length: {words_length}")



# 2 ways to create a list
list1 = []
list2 = list()
print(type(list1))
print(type(list2))

# only diff tuples are immutable
# 2 ways to create a tuple
tuple1 = ()
tuple2 = tuple()
print(type(tuple1))
print(type(tuple2))

# dictionary k,v pairs
dict1 = {}
dict2 = dict()
print(type(dict1))
print(type(dict2))

student = {"name": "krish", "age": 21, "salary": 240}
print(student)
print(student.get("name"))
print(student["name"])

print(student.keys())
print(student.values())

print(student.items())
# shows all kv pairs as list of tuples [(k,v), (k,v), (k,v)]


for key in student.keys():
    print(key)

for value in student.values():
    print(value)

for item in student.items():
    print(item)

for key,value in student.items():
    print(f"{key}:{value}")


#shallow copy concept
student_copy = student # here both will ahve same references if one is updated the copy is alos updated
# so go for shallow copy using copy()
student_copy2 = student.copy();

# dictionary comprehension
squares1 = {x:x**2 for x in range(10)}
print(f"squares1: {squares1}")
# DC with basic syntax    {operation_k:v for-loop}

squares_even = {x:x**2 for x in range(10) if x%2==0}
print(f"squares_even: {squares_even}")

#ex1: use dictionary to count teh frequency of nums
# ex in numbers = [1,2,3,3,4,4,4,4,4] display count of each nums
print("Frequency of number in a list:")
numbers3 = [1,2,3,3,4,4,4,4,4]
# 1:1, 2:1, 3:3, 4:5
frequency ={}
for num in numbers3:
    if num in frequency:
        frequency[num]+=1
    else:
        frequency[num]=1
print(frequency)

