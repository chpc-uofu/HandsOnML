# Ex 6: Solution
print(f"\nRun the test on the model with the highest accuracy")
ind = int(np.argmax(lstAcc))

d = lstResH2Layers[ind]
label = d['model'].name
obj = d['model']
loss_val, acc_val = obj.evaluate(X_test_s, y_test)
print(f"Model:{label}")
print(f"  Loss: {loss_val}")
print(f"  Acc:  {acc_val}")
