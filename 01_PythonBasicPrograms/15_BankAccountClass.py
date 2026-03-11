class BankAccount:

    def __init__(self, owner, balance=0):
        self.owner=owner
        self.balance=balance

    def deposit(self,amount):
        self.balance+=amount
        print(f"${amount} is deposited, New Balance is ${self.balance}")

    def withdraw(self,amount):
        if amount>self.balance:
            print("Insufficient funds")
        else:
            self.balance-=amount
            print(f"${amount} is withdrawn, New Balance is ${self.balance} ")

    def display_balance(self):
        print(f"Balance in account is ${self.balance}")

ChaseAccount=BankAccount("Amy", 2500)
print(ChaseAccount.balance)
ChaseAccount.deposit(350)
ChaseAccount.withdraw(850)
ChaseAccount.display_balance()