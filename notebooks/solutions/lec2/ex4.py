# Ex. 4: Solution
def optim_2hlayer_model(X_train, y_train, n1, n2, lr=0.005, num_epochs=20, verbosity=0):

    tstart = time.time()
    # A.Set-up the model
    name_layer1 = "hila1_nodes=" + str(n1)  # Hidden Layer 1 with n1 neurons
    name_layer2 = "hila2_nodes=" + str(n2)  # Hidden Layer 2 with n2 neurons
    name_model =  "2HiLa_" + str(n1) + "-" + str(n2)

    print(f"  Starting the optimization for model {name_model}")
    model = keras.Sequential([
                    keras.layers.Input(shape=(2,)),                                      # Input layer: input vector (2 features)
                    keras.layers.Dense(units=n1, activation='relu', name=name_layer1),   # Hidden layer with n1
                    keras.layers.Dense(units=n2, activation='relu', name=name_layer2),   # Hidden layer with n2
                    keras.layers.Dense(units=1, activation='sigmoid',name='outlayer')],  # Output layer: 1 Class
                    name=name_model)

    # B.Compile the model
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])

    # C.Fit the model
    hist = model.fit(X_train, y_train, epochs=num_epochs, verbose=verbosity)

    # Extract loss & accuracy
    loss = hist.history['loss']
    acc = hist.history['accuracy']

    # Return model, loss and accuracy
    d = {"model":model,
         "loss": loss,
         "accuracy": acc}
    tend = time.time()
    print(f"  Finishing the optimization for model {name_model} -> Elapsed Time:{tend-tstart:6.2f} sec\n")

    return d
