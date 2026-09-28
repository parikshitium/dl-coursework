class Temperature:
    def __init__(self,value):
        self.value = value
    def celsius(self):
        return self.value
    def fahren(self):
        return (self.value * 9/5) + 32  
    def kelvin(self):
        return self.value + 273.15

temp = Temperature(100)
print(f'Celsius: {temp.celsius()} \nFahrenheit: {temp.fahren()} \nKelvin: {temp.kelvin()}')  