import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
pd.set_option('display.width', 150)

csv = pd.read_csv('data/notification_sequence.csv')

test = {}
test['gender'] = ('F' if input('Gender (0: Female, 1: Male): ') == '0' else 'M')
test['age'] = (int(input('Age (0: ~15, 1: 16~18, 2: 19~22, 3: 22~28, 4: 29~): ')))
test['department'] = (input('Department (cs, ee, oe, mg, lb, bi, ed, ot): '))
test['total_usetime'] = (int(input('Usetime per day (0: ~1hr, 1: 1~2hrs, 2: 2~3hrs, 3: 3~4hrs, 4: 4hrs~): ')))
test['social_usetime'] = (int(input('Social Usetime per day (0: ~0.5hr, 1: 0.5~1hrs, 2: 1~2hrs, 3: 2~3hrs, 4: 3hrs~): ')))
test['communication_usetime'] = (int(input('Communication Usetime per day (0: ~0.5hr, 1: 0.5~1hrs, 2: 1~2hrs, 3: 2~3hrs, 4: 3hrs~): ')))
test['entertainment_usetime'] = (int(input('Entertainment Usetime per day (0: ~0.5hr, 1: 0.5~1hrs, 2: 1~2hrs, 3: 2~3hrs, 4: 3hrs~): ')))
test['news_usetime'] = (int(input('News Usetime per day (0: ~0.5hr, 1: 0.5~1hrs, 2: 1~2hrs, 3: 2~3hrs, 4: 3hrs~): ')))
test['system_usetime'] = (int(input('System Usetime per day (0: ~0.5hr, 1: 0.5~1hrs, 2: 1~2hrs, 3: 2~3hrs, 4: 3hrs~): ')))
test['notification_usetime'] = (int(input('Notification Usetime per day (0: ~0.5hr, 1: 0.5~1hrs, 2: 1~2hrs, 3: 2~3hrs, 4: 3hrs~): ')))
vo = input('View order (top, middle, bottom) (e.x.: 1 2 3): ').strip().split()
test['view_order_top'] = int(vo[0])
test['view_order_middle'] = int(vo[1])
test['view_order_bottom'] = int(vo[2])
test['scenario'] = (input('Scenario (wakingup, working, resting, commuting): '))

new_csv = {
    'gender': [],
    'age': [],
    'department': [],
    'scenario': [],
    'view-order': [],
    'total': [],
    'social': [],
    'communication': [],
    'entertainment': [],
    'notification': []
}
Y_111111 = []
Y_1122 = []
Y_123 = []
Y_222 = []
Y_24 = []
Y_15 = []

for idx, row in csv.iterrows():
    if row['social_usetime'] == '?':
        continue
    tot = int(row['social_usetime']) + int(row['communication_usetime']) + int(row['entertainment_usetime']) + int(row['news_usetime']) + int(row['system_usetime']) + int(row['notification_usetime'])
    if tot == 0:
        continue
    new_csv['gender'].append(row['gender'])
    new_csv['age'].append(row['age'])
    new_csv['department'].append(row['department'])
    new_csv['scenario'].append(row['scenario'])
    new_csv['view-order'].append(1 if row['view_order_top'] == 1 else (0 if row['view_order_middle'] == 1 else -1))
    new_csv['total'].append(row['total_usetime'])
    new_csv['social'].append(int(row['social_usetime']) / tot)
    new_csv['communication'].append(int(row['communication_usetime']) / tot)
    new_csv['entertainment'].append(int(row['entertainment_usetime']) / tot)
    new_csv['notification'].append(int(row['notification_usetime']) / tot)
    Y_111111.append(row['tol_111111'])
    Y_1122.append(row['tol_1122'])
    Y_123.append(row['tol_123'])
    Y_222.append(row['tol_222'])
    Y_24.append(row['tol_24'])
    Y_15.append(row['tol_15'])
tot = int(test['social_usetime']) + int(test['communication_usetime']) + int(test['entertainment_usetime']) + int(test['news_usetime']) + int(test['system_usetime']) + int(test['notification_usetime'])
new_csv['gender'].append(test['gender'])
new_csv['age'].append(test['age'])
new_csv['department'].append(test['department'])
new_csv['scenario'].append(test['scenario'])
new_csv['view-order'].append(1 if test['view_order_top'] == 1 else (0 if test['view_order_middle'] == 1 else -1))
new_csv['total'].append(test['total_usetime'])
new_csv['social'].append(int(test['social_usetime']) / tot)
new_csv['communication'].append(int(test['communication_usetime']) / tot)
new_csv['entertainment'].append(int(test['entertainment_usetime']) / tot)
new_csv['notification'].append(int(test['notification_usetime']) / tot)

new_csv = pd.get_dummies(pd.DataFrame(new_csv))
X_test = np.array(new_csv.tail(1))
X = np.array(new_csv.values)
X = np.delete(X, 788, 0)
Y_111111 = np.array(Y_111111)
Y_1122 = np.array(Y_1122)
Y_123 = np.array(Y_123)
Y_222 = np.array(Y_222)
Y_24 = np.array(Y_24)
Y_15 = np.array(Y_15)

_mean = np.mean(X, axis=0)
_std = np.std(X, axis=0)
_scale = np.full(X.shape[1], 1.0)
_scale[-4:] *= 0.25
_scale[3:6] *= 2.85

X -= _mean
X /= _std
X *= _scale

knn_111111 = KNeighborsClassifier(n_neighbors=1)
knn_111111.fit(X, Y_111111)
knn_1122 = KNeighborsClassifier(n_neighbors=1)
knn_1122.fit(X, Y_1122)
knn_123 = KNeighborsClassifier(n_neighbors=1)
knn_123.fit(X, Y_123)
knn_222 = KNeighborsClassifier(n_neighbors=1)
knn_222.fit(X, Y_222)
knn_24 = KNeighborsClassifier(n_neighbors=1)
knn_24.fit(X, Y_24)
knn_15 = KNeighborsClassifier(n_neighbors=1)
knn_15.fit(X, Y_15)

print()
print(f'tol_111111 (Strict): {knn_111111.predict(X_test)[0]}')