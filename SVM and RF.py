#Define the classifier
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 50, random_state = 42)
RF_model.fit(X_for_RF, y_train) 

from sklearn import svm
SVM_model = svm.SVC(C=1,kernel='rbf',gamma='auto')
SVM_model.fit(X_for_SVM, y_train)

# predict on Test data
# extract features from test data and reshape, just like training data
test_features = feature_extractor(x_test)
test_features = np.expand_dims(test_features, axis=0)

test_for_SVM = np.reshape(test_features, (x_test.shape[0], -1))
test_for_RF = np.reshape(test_features, (x_test.shape[0], -1))

test_prediction_SVM = SVM_model.predict(test_for_SVM)
test_prediction_RF = RF_model.predict(test_for_RF)


#Inverse le transform to get original label back
test_prediction_SVM = le.inverse_transform(test_prediction_SVM)
test_prediction_RF = le.inverse_transform(test_prediction_RF)

# print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, test_prediction_SVM))
print ("Accuracy = ", metrics.accuracy_score(test_labels, test_prediction_RF))
