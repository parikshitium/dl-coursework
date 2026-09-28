class Animal:
    def __init__(self,name):
        self.name = name

    def speak(self):
        pass

class Dog(Animal):
    def __init__(self,name,breed):
        super().__init__(name)
        self.breed = breed
    def speak(self):
        super().speak()
        print(f'{self.name} barks')

dog = Dog('Bruno','Labrador')
dog.speak()