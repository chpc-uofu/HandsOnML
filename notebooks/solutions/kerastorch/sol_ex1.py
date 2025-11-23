# Ex. 1: KerasTorch (Logistic Regression class ModelEx)

class LogisticRegressionModelEx(nn.Module):

    def __init__(self, num_inputs):

        # The class inherits from the class nn.Module
        super(LogisticRegressionModelEx,self).__init__()

        # Define a Single LAYER object which connects
        #     the input with 1 single output
        self.linear = nn.Linear(num_inputs, 1)

    def forward(self, x):

        # Applies the forward propagation
        z = self.linear(x)
        return z

