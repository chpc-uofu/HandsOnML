# Ex. 1: KerasTorch (Logistic Regression class Model2)

class LogisticRegressionModel2(nn.Module):

    def __init__(self, num_inputs):

        # The class inherits from the class nn.Module
        super(LogisticRegressionModel2,self).__init__()

        # Define a Single LAYER object which connects
        #     the input with 1 single output
        self.linear = nn.Linear(num_inputs, 1)

    def forward(self, x):

        # Applies the forward propagation
        z = self.linear(x)
        return z

