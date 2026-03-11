year = int(input("Enter the year to check for leap year: "))
if year%4==0:
    if year%100==0:
        if year%400==0:
            print(year, "is a Leap year")
        else:
            print(year, "is NOT a Leap year")
    else:
        print(year, " is a Leap year")
else:
    print(year, " is NOT a Leap year")
