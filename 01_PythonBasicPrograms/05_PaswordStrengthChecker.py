def is_strong_password(password):
    if len(password)<8:
        return False
    if not any( char.isdigit() for char in password):
        return False
    if not any(char.islower() for char in password):
        return False
    if not any(char.isupper() for char in password):
        return False
    if not any(char in '!@#$%^&*()_+' for char in password):
        return False
    else:
        return True

print(is_strong_password("Hello1*")) #<8 False
print(is_strong_password("Hello123*")) # True

