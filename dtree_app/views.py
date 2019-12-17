from django.shortcuts import render
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
import multiprocessing
from django.http import HttpResponse, HttpResponseRedirect
import subprocess
import distutils.dir_util
from django.conf import settings
import os, datetime
from django.core.mail import send_mail
from django.core.mail import EmailMessage
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from io import BytesIO
import base64
from matplotlib.pyplot import *

def index(request):
	return render(request,'index.html')

def decision(request):
	if request.POST and request.FILES:
		try:
			global filename, emailid,pred_decision
			emailid = request.POST['emailid']
			filename = request.FILES['csv_file']

			#id = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
			#dir_path = os.path.join('C:\\Users\\Gumbi\\Desktop\\classification\\media\\result', id)

			#folder = os.mkdir(dir_path)

			dataset = pd.read_csv(filename)

			#decision_result=int(pred_decision(dataset))

			X = dataset.iloc[:,[2,3]].values
			y = dataset.iloc[:,4].values

			from sklearn.model_selection import train_test_split
			X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.25, random_state = 0)

			from sklearn.preprocessing import StandardScaler
			sc_X = StandardScaler()
			X_train = sc_X.fit_transform(X_train)
			X_test = sc_X.transform(X_test)

			from sklearn.tree import DecisionTreeClassifier
			classifier = DecisionTreeClassifier(criterion = 'entropy',random_state = 0)
			classifier.fit(X_train,y_train)
			
			y_pred = classifier.predict(X_test)

			from sklearn.metrics import classification_report, confusion_matrix
			a=(classification_report(y_test, y_pred))
			b=(confusion_matrix(y_test, y_pred))

			#with open(os.path.join(dir_path,"one" + ".log"), 'w') as file:
				#for line in a,b:
					#file.write(str(line))
				#file.write("\nSUCCESSFULLY COMPLETED OPERATION")

			#fp = open("media/" + "result/" + str(id) + "/" + "one" + ".log", 'a')
			#fp.write("\nSUCCESSFULLY COMPLETED OPERATION")
			#fp.close()

			from matplotlib.colors import ListedColormap
			X_set, y_set=X_test,y_test
			X1,X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1,
				stop=X_set[:,0].max()+1,step=0.01),np.arange(start=X_set[:,1].min()-1,
				stop=X_set[:,1].max()+1,step=0.01))

			plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
			plt.xlim(X1.min(),X1.max())
			plt.ylim(X2.min(),X2.max())


			for i,j in enumerate(np.unique(y_set)):
			    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
			    	c=ListedColormap(('red','green'))(i),label=j)
			plt.title('Decision tree classifier(Testing set)')
			plt.xlabel('X axis')
			plt.ylabel('Y axis')
			#plt.legend()
			

			#sample = "Test-result"
			#plt.savefig(dir_path + "/" + sample)

			buffer = BytesIO()
			plt.savefig(buffer, format = 'png')
			buffer.seek(0)
			image_png = buffer.getvalue()
			buffer.close()

			graphic = base64.b64encode(image_png)
			graphic = graphic.decode('utf-8')


			return render (request, 'progress.html', {'a':a, 'graphic' : graphic})

            #sample = "Test result"
            #pylab.savefig(dir_path + "/" + sample)			

			return HttpResponseRedirect("/progress")
		except Exception as e:
			print(e)
			print(request.FILES)
			return HttpResponse("""<h3> Oooppss!! Errorr! </h3>""")

	return render(request, 'decision.html');

'''def progress(request):
	fp = open("media/" + "result/" + str(id) + "/" + str(id) + ".log", 'r')
	file=""
	refresh =True
	for i in fp:
		if i == "SUCCESSFULLY COMPLETED OPERATION":
			refresh = False
		file = "      "+ file +  "\n " + i + "\n"
	return render(request,"progress.html",{'file':file})'''