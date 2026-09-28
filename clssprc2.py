class Coffee:
    def __init__(self,water,coffee,milk):
        self.water = water
        self.coffee = coffee
        self.milk = milk
    def make(self):
        water_req = 150
        coffee_req = 20
        milk_req = 100

        if self.water >= water_req and self.coffee >= coffee_req and self.milk >= milk_req:
            self.water -= water_req
            self.coffee -= coffee_req
            self.milk -= milk_req
            print(f'Coffee ready. Ingredients remaining: Water: {self.water}ml, Coffee: {self.coffee}gm, Milk: {self.milk}ml')
        else:
            print('Not enough ingreedients')

machine = Coffee(water=1000,coffee=150,milk=1000)

machine.make()

        