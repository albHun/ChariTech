from sklearn.svm import SVC


def svmLearning(X_training, y_training, X_cv, y_cv):
	# Using default svc classifier
	clf = SVC()
	clf.fit(X_training, y_training)
	print("CV score")
	print(clf.score(X_cv, y_cv))