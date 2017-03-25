from sklearn.neural_network import MLPClassifier

def NNlearning(X_training, y_training, X_cv, y_cv):
	# Adjust hidden layer sizes here
	hidden_layer_size_selections = [(220, 110)]
	for hidden_layer_sizes in hidden_layer_size_selections:
		clf = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes, verbose = True, random_state=1, tol = 1e-7, max_iter = 200,
		                    solver= 'adam', learning_rate= 'constant', momentum= 0, alpha = 0.1)
		clf.fit(X_training, y_training)
		print("The hidden layer sizes are", hidden_layer_sizes)
		print(clf.score(X_training, y_training))
		print(clf.score(X_cv, y_cv))