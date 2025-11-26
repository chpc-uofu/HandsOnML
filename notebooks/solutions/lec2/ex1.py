# Ex. 1
X_val_s, X_test_s, y_val, y_test =\
        train_test_split(X_temp_s, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
print(f"Training/validation/test::")
print(f"  X_train (scaled):{X_train_s.shape}    y_train :{y_train.shape}")
print(f"  X_val (scaled)  :{X_val_s.shape  }    y_val   :{y_val.shape}")
print(f"  X_test (scaled) :{X_test_s.shape }    y_test  :{y_test.shape}")
