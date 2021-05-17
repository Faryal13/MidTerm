from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from pandas import read_csv

from sklearn.svm import SVC




def predict(request):
    return render(request, 'predict.html')


def predict_chances(request):

    if request.POST.get('action') == 'post':

        # Receive data from client
        sepal_length = float(request.POST.get('sepal_length'))
        sepal_width = float(request.POST.get('sepal_width'))
        petal_length = float(request.POST.get('petal_length'))
        petal_width = float(request.POST.get('petal_width'))

        df = pd.read_csv(r"C:\Users\ktaim\Desktop\IrisPrediction\iris.csv")
    
        X = df[['sepal_length','sepal_width','petal_length','petal_width']]
        y = df['classification']
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1)
        model = SVC(gamma='auto')
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test) 
        result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        classification = result[0]
        accuracy=100* metrics.accuracy_score(Y_test, predictions)


        return JsonResponse({'result': classification, 'sepal_length': sepal_length,
                             'sepal_width': sepal_width, 'petal_length': petal_length, 'petal_width': petal_width,'accuracy':accuracy},
                            safe=False)


                       
        