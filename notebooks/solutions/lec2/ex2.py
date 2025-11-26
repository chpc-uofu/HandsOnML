# Exercise 2:
lr_sk = LogisticRegression(solver='lbfgs', max_iter=2000, random_state=123)
lr_sk.fit(X_train_s,y_train)

y_train_pred = lr_sk.predict(X_train_s)
acc_train = accuracy_score(y_train, y_train_pred)
print(f"  acc. (train): {acc_train:6.2f}")
