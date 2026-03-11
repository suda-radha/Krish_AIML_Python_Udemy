# Filter fun is similar to map fun but it applies a condition fun on a list
# map() and filter() may looks similar in syntax but understand that using map() u appy any fun on a list
# and using a filter() u apply a filter condition method on a list
# filter() syntax === filter(filter-fun, list)

numbers = [1,2,3,4,5,6,7,8,9]

def even(num):
    return num%2==0

print(list(filter(even, numbers)))

#Ex2:
nums = [1,2,3,4,5,6,7,8,9]
nums5 = []
def greater_than_five(nums):
    for num in nums:
        if num>5:nums5.append(num)
    return nums5

print(greater_than_five(nums))

# same using lambda and filter
print(list(filter(lambda x: x>5, nums)))

#Ex3: filter() with lambda and multiple conditions

numbs3 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
even_and_greater_than_five = list(filter(lambda x:x>5 and x%2==0, numbs3))
print(even_and_greater_than_five)


#filter() on dictionaries
# filter people by age >25

people = [
    {"name": "Amy", "age":26},
    {"name": "Ben", "age": 22},
    {"name": "Cam", "age": 27},
    {"name": "Den", "age": 28},
    {"name": "Ely", "age": 25}
]

def age_less_than_25(person):
    return person["age"]>25

print(list(filter(age_less_than_25, people)))