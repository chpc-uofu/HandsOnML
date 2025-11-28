# Ex3: Solution

def optim_1hlayer_model( X_train: np.ndarray,  # features matrix (np.ndarray) 
                         y_train: np.ndarray,  # labels (np.ndarray)
                         n1: int,              # number of neurons in layer 1 (integer)
                         lr: float = 0.005,    # learning rate (float)      [0.005]
                         num_epochs: int = 20, # number of epochs (integer) [20]
                         batch_sz: int = 32,   # batch size (integer)       [32]
                         verbosity: int = 0    # verbosity level (integer)  [0]
                       ) -> dict[str, Any]:    
   
    """
    Trains and optimizes a Keras Sequential model with one hidden layer for binary classification.

    The model architecture consists of:
    1. An Input Layer (shape 2).
    2. A Dense Hidden Layer with 'n1' neurons and 'relu' activation.
    3. A Dense Output Layer with 1 neuron and 'sigmoid' activation.

    The model is compiled using the Adam optimizer, binary_crossentropy loss, and accuracy metrics.

    Args:
        X_train (Any): Training feature data (e.g., shape (m, 2), a numpy array).
        y_train (Any): Training target labels (e.g., shape (m, 1), a numpy array).
        n1 (int): The number of neurons in the single hidden layer.
        lr (float, optional): The learning rate for the Adam optimizer. Defaults to 0.005.
        num_epochs (int, optional): The number of epochs for training. Defaults to 20.
        batch_sz (int, optional): The size of a mini-batch. Defaults to 32.
        verbosity (int, optional): Keras training verbosity (0=silent, 1=progress bar, 2=one line per epoch). Defaults to 0.

    Returns:
        dict[str, Any]: A dictionary containing:
          'model' (keras.Model): The trained Keras model object.
          'loss' (list[float]): List of training loss values per epoch.
          'accuracy' (list[float]): List of training accuracy values per epoch.        
    """
    
    tstart = time.time()
    
    # A.Set-up the model
    name_layer1 = "hila1_nodes=" + str(n1)  # Hidden Layer 1 with n1 neurons
    name_model =  "1HiLa_" + str(n1)        # Name of the model
    
    print(f"  Starting the optimization for model {name_model}") 
    model = keras.Sequential([
                  keras.layers.Input(shape=(2,)),                                     # Input layer: input vector (2 features)
                  keras.layers.Dense(units=n1, activation='relu', name=name_layer1),  # Hidden layer with n1   # <-- ADD 
                  keras.layers.Dense(units=1, activation='sigmoid',name='outlayer')], # Output layer: 1 Class  # <-- ADD
                  name=name_model)

    # B.Compile the model
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])   # <--- ADD

    # C.Fit the model
    hist = model.fit(X_train, y_train, epochs=num_epochs, verbose=verbosity, batch_size=batch_sz)  

    # Extract loss & accuracy 
    loss = hist.history['loss']
    acc = hist.history['accuracy']

    # Return model, loss and accuracy
    d = {"model": model, 
         "loss": loss, 
         "accuracy": acc}
    
    tend = time.time()
    print(f"  Finishing the optimization for model {name_model} -> Elapsed Time:{tend-tstart:6.2f} sec\n")
    
    return d
