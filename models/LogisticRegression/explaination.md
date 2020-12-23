# Logistic Regression
## Preprocessing
1. drop ```?``` attributes
```python=
# drop '?'
notification_df = notification_df.replace('?', pd.NaT)    
notification_df = notification_df.dropna(axis=0)
```
2. Choose the features with string to do ```labelencoding``` and split X and y
```python=
# label encoding
X_categorical_col = ['gender', 'department', 'scenario']
y_categorical_col = ['tol_111111', 'tol_1122', 'tol_123', 'tol_222', 'tol_24', 'tol_15']

X = notification_df.iloc[:, :-6]
y = notification_df.iloc[:, -6:]
```

3. reconstruct the dataframe
    - drop the test where the data is below half an hour
    - use ```ratio``` to represent the corelation
    - drop the test where the total time in sum is below half an hour
    - replace the values to usetime with ```specific usetime ratio```
    - eliminate the triple effect from ```view_order``` by replacing them with one value
 
```python=
for num, i in X.iterrows():
    # print(i['total_usetime'])
    if i['total_usetime'] == 0:
        X.drop(index = [num], inplace = True)
        y.drop(index = [num], inplace = True)
    else:
        tot = int(i['social_usetime']) + int(i['communication_usetime']) + int(i['entertainment_usetime']) + int(i['news_usetime']) + int(i['system_usetime']) + int(i['notification_usetime'])
        if tot == 0:
            X.drop(index = [num], inplace = True)
            y.drop(index = [num], inplace = True)
        else:
            i['social_usetime'] = int(i['social_usetime'])/tot
            i['communication_usetime'] = int(i['communication_usetime'])/tot
            i['entertainment_usetime'] = int(i['entertainment_usetime'])/tot
            i['news_usetime'] = int(i['news_usetime'])/tot
            i['system_usetime'] = int(i['system_usetime'])/tot
            i['notification_usetime'] = int(i['notification_usetime'])/tot
            new['view_order'].append(1 if i['view_order_top'] == 1 else (0 if i['view_order_middle'] == 1 else -1))
```

4. Since there are some test deleted, reset the index and concatenate ```view_order``` and the original X

```python=
X.reset_index(drop=True, inplace=True)
X = pd.concat([X, pd.DataFrame(new)], axis=1)
X = X.drop(['view_order_top', 'view_order_middle', 'view_order_bottom'], axis=1)
```
5. Get the ```labelencoding``` with the features specified above

```python=
labelencoder = LabelEncoder()
for i in X_categorical_col:
    X[i] = labelencoder.fit_transform(X[i])

le_name_mapping = []
for i in y_categorical_col:
    labelencoder.fit(y[i])
    le_name_mapping.append(dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_))))
```

6. Since we are dealing with categorical values, use ```one-hot code``` for model construction

```python=
# handle X with categorical in 'one-hot encode'
X = pd.get_dummies(X, columns = X_categorical_col)
print(X)
```
7. Normalize the features

```python=
X = preprocessing.normalize(X)
```
## Test-and-Split
Use ```Holdout``` to split the data with ratio **7:3**

```python=
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
```
## Model Construction
1. Construct the ```logistic regression``` model and fit in the data for prediction
```python=
clf = LogisticRegression(random_state=666) clf.fit(X_train, y_train)
y_pred = mod.predict(X_test)
```
2. For each different test, do the same thing
## Result
1. Use ```accuracy``` as the method to see the result
```python=
print("Classification accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))

    confusion_mat = metrics.confusion_matrix(y_test, y_pred, labels=class_names)
    plot_confusion_matrix(confusion_mat, class_names)
```
:::success
result:
* tol_111111: 0.0423728813559322
* tol_1122: 0.16101694915254236
* tol_123: 0.21610169491525424
* tol_222: 0.211864406779661
* tol_24: 0.4915254237288136
* tol_15: 0.5466101694915254
:::

2. plot the ```confusion matrix```
* tol_111111
![](https://i.imgur.com/mW8B6JK.png)
* tol_1122
![](https://i.imgur.com/hyu9uBz.png)
* tol_123
![](https://i.imgur.com/ssz5LOu.png)
* tol_222
![](https://i.imgur.com/MgnXq3c.png)
* tol_24:
![](https://i.imgur.com/vX6RoEm.png)
* tol_15:
![](https://i.imgur.com/QaNJl1i.png)
