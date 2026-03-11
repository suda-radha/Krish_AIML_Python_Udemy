# Slicing basics : and ::
nums=[1,2,3,4,5,6,7,8,9,0]
print(nums[:5])
print(nums[5:])
print(nums[::]) # :: means everything
print(nums[::2]) # everything with skip size =2
print(nums[::-1]) # everything in reverse with skipsize 1
print(nums[::-2]) # everything in reverse with skipsize 2

# print list items with their index
nums1=[1,2,3,4,5,6,7,8,9,0]
for num in nums1:
    print(num)

# print nums with index - use enumerate()
for index,num in enumerate(nums1):
    print(index,num)



