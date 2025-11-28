# Ex 5:
# Run the models on the same cross-validation set
lstAcc = [ ]
for item in lstResH2Layers:
    label = item['model'].name
    obj = item['model']
    loss_val, acc_val = obj.evaluate(X_val_s, y_val)
    lstAcc.append(acc_val)
    print(f"Model:{label}")
    print(f"  Loss: {loss_val}")
    print(f"  Acc:  {acc_val}")
