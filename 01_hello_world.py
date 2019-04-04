from __future__ import print_function
from sklearn import tree

features = [[140,1], [130,1], [150,0], [170, 0]]
# features = [[140,'smooth'], [130,'smooth'], [150,'bump'], [170, 'bump']]

labels = [0, 0, 1, 1]
# labels = ['apple', 'apple', 'orange', 'orange']

clf = tree.DecisionTreeClassifier()

clf = clf.fit(features, labels)

print(clf.predict([[150,0]]))
