# Ex 6: Solution
print(f"\nRun the test on the model with the highest accuracy")
ind = int(np.argmax(lstAcc))

d = lstResH2Layers[ind]
label = d['model'].name
obj = d['model']
loss_test, acc_test = obj.evaluate(X_test_s, y_test)
print(f"Model:{label}")
print(f"  Loss: {loss_test}")
print(f"  Acc:  {acc_test}")
