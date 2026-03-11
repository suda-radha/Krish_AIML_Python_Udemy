def factorial(num):
    fact=1;
    for n in range(num+1):
        if n==0:
            continue
        fact= fact*n
    return fact

print(factorial(4))

#recursive way
def factrecur(n):
    if n ==0:
        return 1
    else:
        return n*factrecur(n-1)

print(factrecur(5))

print(factorial(0))
print(factrecur(0))
print(factorial(2))
print(factrecur(2))
