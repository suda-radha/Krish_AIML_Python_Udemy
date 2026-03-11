def isPalindrome(s):
    s=s.lower().replace(" ", "")
    return s==s[::-1]

print(isPalindrome("mart"))
print(isPalindrome("liril"))
print(isPalindrome("malayalam"))
print(isPalindrome("A man a plan a canal panama"))