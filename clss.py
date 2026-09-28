class BankAcc:
    def __init__(self,owner,balance):
        self.owner = owner
        self.balance = float(balance)
    def deposit(self,put_in):
        self.balance ++ put_in
    def withdrawl(self,take_out):
        self.balance -= take_out
    def show(self):
        print(f'Owner: {self.owner}\nBalance: {self.balance}')

own = input('Name: ')
acc = BankAcc(own,1000)

w_or_d = input('what do you want to do? ')

if w_or_d.lower() == 'w':
    w_in = float(input('How much: '))
    acc.withdrawl(w_in)
    acc.show()
if w_or_d.lower() == 'd':
    w_in = float(input('How much: '))
    acc.deposit(w_in)
    acc.show()