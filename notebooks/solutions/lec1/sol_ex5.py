# Code for Logistic Regression

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

def train_model(X_train_tensor, y_train_tensor, model, loss_fn, optim, num_epochs=100000 , delta_print=10000):
    """"
    Function which trains the model
    """

    # Set model to train mode
    # Strictly not necessary for our case
    model.train()

    # Loop over the epochs
    for epoch in range(num_epochs):

        # PART A: FORWARD PROPAGATION ( => )
        # Step 1: Generate the output (activation of the linear layer)
        output = model(X_train_tensor)

        # Step 2: Use the activation of the last layer & the labels
        #         to calculate the loss.
        loss = loss_fn(output, y_train_tensor)

        # Step B: BACK PROPAGATION ( <= )
        # Step 3: Calculate the gradients of the parameters
        optim.zero_grad()   # Init. the gradients to ZERO!!
        loss.backward()     # Calc. grad. of param.

        # Step 4: Adjust the parameters
        optim.step()

        if (epoch+1)%delta_print == 0 or epoch==0:
           print(f"  Epoch {epoch+1}/{num_epochs}  Loss:{loss.item():.6f}")

    return loss.item()


if __name__ == "__main__":

    # Logistic Regression v.2
    model2 = LogisticRegressionModel2(num_inputs=2)
    print(f"  Logistic Model:{model2}")

    # Check the model v.2 ANTE optimization
    for name, param in model2.named_parameters():
        print(f"Name:{name:20s} -> param:{param.shape}")
        print(f"{param.data}\n")

    # BCEWithLogitsLoss
    loss_fn2 = nn.BCEWithLogitsLoss() 

    # Optimization & Training the model
    optim2 = optim.SGD(model2.parameters(), lr=0.005)
    final_loss2 = train_model(X_train_tensor, y_train_tensor, model2, loss_fn2, optim2)
    print(f"Loss in the last step:{final_loss2:.6f}")

    # Check the model v.2 POST optimization
    for name, param in model2.named_parameters():
        print(f"Name:{name:20s} -> param:{param.shape}")
        print(f"{param.data}\n")
