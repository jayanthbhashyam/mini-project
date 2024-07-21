User side views.py
from django.shortcuts import render,HttpResponse
from django.contrib import messages
from users.forms import UserRegistrationForm
from users.models import  UserRegistrationModel,FlightDataModel
import io,csv
from django.conf import settings

from .FlightDataPreproces import DPDataPrePRocess
from .models import FlightDataModel
from django_pandas.io import read_frame
# Create your views here.

def UserRegisterAction(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            # return HttpResponseRedirect('./CustLogin')
            form = UserRegistrationForm()
            return render(request, 'UserRegister.html', {'form': form})
        else:
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegister.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
            # return render(request, 'user/userpage.html',{})
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserUploadForm(request):
    return render(request,'users/uploadform.html',{})

def UserDataUpload(request):
    # declaring template
    template = "users/UserHome.html"
    data = FlightDataModel.objects.all()
    # prompt is a context variable that can have different values      depending on their context
    prompt = {
        'order': 'Order of the CSV should be name, email, address,    phone, profile',
        'profiles': data
    }
    # GET request returns the value of the data with the specified key.
    if request.method == "GET":
        return render(request, template, prompt)
    csv_file = request.FILES['file']
    # let's check if it is a csv file
    if not csv_file.name.endswith('.csv'):
        messages.error(request, 'THIS IS NOT A CSV FILE')
    data_set = csv_file.read().decode('UTF-8')
    try:
        # setup a stream which is when we loop through each line we are able to handle a data in a stream
        io_string = io.StringIO(data_set)
        next(io_string)
        for column in csv.reader(io_string, delimiter=',', quotechar="|"):
            _, created = FlightDataModel.objects.update_or_create(
            DAY = column[1],
            DEPARTURE_TIME= column[2],
            FLIGHT_NUMBER= column[3],
            DESTINATION_AIRPORT= column[4],
            ORIGIN_AIRPORT= column[5],
            DAY_OF_WEEK= column[6],
            TAXI_OUT= column[7]
            )
    except Exception as ex:
        print('error at', ex)
    context = {}

    return render(request, 'users/UserHome.html', context)

def DataPreProcessing(request):
    #dataset = settings.MEDIA_ROOT + "\\" + 'flightsdata.csv'
    qs = FlightDataModel.objects.all()
    dataset = read_frame(qs)
    print("Dataset ",dataset)
    x = DPDataPrePRocess()
    data = x.process_data(datasetname = dataset)

    return render(request, 'users/PreProcessData.html',{'data':qs})

def UsermachineLearning(request):
    qs = FlightDataModel.objects.all()
    dataset = read_frame(qs)
    x = DPDataPrePRocess()
    lg_dict = x.MyLogiSticregression(dataset)
    #lg_dict = {}
    dt_dict = x.MyDecisionTree(dataset)
    rf_dict = x.MyRandomForest(dataset)
    br_dict = x.MyBayesianRidge(dataset)
    gbr_dict = x.MyGradientBoostingRegressor(dataset)

    return render(request,'users/UsrMachineLearningRslt.html',{'lg_dict':lg_dict,'dt_dict':dt_dict,'rf_dict':rf_dict,'br_dict':br_dict,'gbr_dict':gbr_dict})

def UserGraphs(request):
    qs = FlightDataModel.objects.all()
    dataset = read_frame(qs)
    x = DPDataPrePRocess()
    #lg_dict = x.MyLogiSticregression(dataset)
    lg_dict = {}
    dt_dict = x.MyDecisionTree(dataset)
    rf_dict = x.MyRandomForest(dataset)
    br_dict = x.MyBayesianRidge(dataset)
    gbr_dict = x.MyGradientBoostingRegressor(dataset)

    return render(request, 'users/UserGraphs.html',
                  {'lg_dict': lg_dict, 'dt_dict': dt_dict, 'rf_dict': rf_dict, 'br_dict': br_dict,
                   'gbr_dict': gbr_dict})

FlightDataProcess.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
import numpy as np
class DPDataPrePRocess:
    def process_data(self,datasetname):
        #dataset = pd.read_csv(datasetname)
        dataset = datasetname
        dataset = dataset[['DAY', 'DEPARTURE_TIME', 'FLIGHT_NUMBER', 'DESTINATION_AIRPORT', 'ORIGIN_AIRPORT', 'DAY_OF_WEEK','TAXI_OUT']]
        #print(dataset.dtypes)
        dataset.fillna
        dataset.dropna()
        dataset = dataset.fillna(0)
        dataset.fillna(method='ffill')
        print(dataset.isnull().values.any())
        #print(dataset.head(10))
        #plot_corr(dataset)
        #plt.show()
        print(dataset.dtypes)
        dataset.to_csv('file1.csv')
        data_dict = dataset.to_dict()
        return data_dict

    def MyLogiSticregression(self,dataset):
        print("###Logistic Regression####")
        #print('Have a great day ',dataset)
        #dataset = pd.read_csv(dataset)
        dataset = dataset[['DAY','DEPARTURE_TIME','FLIGHT_NUMBER','DESTINATION_AIRPORT','ORIGIN_AIRPORT','DAY_OF_WEEK','TAXI_OUT']]
        #print(dataset.head())
        X = dataset.iloc[:,:1].values
        y = dataset.iloc[:,2].values
        X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=1/3,random_state=0)

        model = LogisticRegression()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)

        #acuracy = accuracy_score(y_pred,y_test)
        #print(acuracy)
        lgDict = {}
        lg_MAE = metrics.mean_absolute_error(y_pred.round(), y_test)
        lg_MSE = metrics.mean_squared_error(y_pred.round(), y_test)
        lg_EVS  = metrics.explained_variance_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')
        lg_MedianAE = metrics.median_absolute_error(y_test, y_pred)
        lg_R2Score = metrics.r2_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')

        lgDict.update({'lg_MAE':lg_MAE,'lg_MSE':lg_MSE,'lg_EVS':lg_EVS,'lg_MedianAE':lg_MedianAE,'lg_R2Score':lg_R2Score})

        print("MAE=",lg_MAE )
        print("MSE=",lg_MSE )
        print("RMSE=", np.sqrt(metrics.mean_squared_error(y_pred.round(), y_test)))
        print("Variance Score ",lg_EVS)
        print("Median Absalute Error=",lg_MedianAE)
        print("R2_Score", lg_R2Score)

        return lgDict

    def MyDecisionTree(self, dataset):
        print("###Decesion Treee####")
        #print('Have a great day ', dataset)
        #dataset = pd.read_csv(dataset)
        dataset = dataset[
            ['DAY', 'DEPARTURE_TIME', 'FLIGHT_NUMBER', 'DESTINATION_AIRPORT', 'ORIGIN_AIRPORT', 'DAY_OF_WEEK',
             'TAXI_OUT']]
        #print(dataset.head())
        X = dataset.iloc[:, :1].values
        y = dataset.iloc[:, 2].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # acuracy = accuracy_score(y_pred,y_test)
        # print(acuracy)
        dtDict = {}
        dt_MAE = metrics.mean_absolute_error(y_pred.round(), y_test)
        dt_MSE = metrics.mean_squared_error(y_pred.round(), y_test)
        dt_EVS =  metrics.explained_variance_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')
        dt_MedianAE = metrics.median_absolute_error(y_test, y_pred)
        dt_R2Score = metrics.r2_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')

        dtDict.update({'dt_MAE':dt_MAE,'dt_MSE':dt_MSE,'dt_EVS':dt_EVS,'dt_MedianAE':dt_MedianAE,'dt_R2Score':dt_R2Score})

        print("MAE=", dt_MAE)
        print("MSE=", dt_MAE)
        print("RMSE=", np.sqrt(metrics.mean_squared_error(y_pred.round(), y_test)))
        print("Variance Score ",dt_EVS)
        print("Median Absalute Error=", dt_MedianAE)
        print("R2_Score", dt_R2Score)
        return dtDict

    def MyRandomForest(self, dataset):
        print("###RadomForest####")
        #print('Have a great day ', dataset)
        #dataset = pd.read_csv(dataset)
        dataset = dataset[
            ['DAY', 'DEPARTURE_TIME', 'FLIGHT_NUMBER', 'DESTINATION_AIRPORT', 'ORIGIN_AIRPORT', 'DAY_OF_WEEK',
             'TAXI_OUT']]
        #print(dataset.head())
        X = dataset.iloc[:, :1].values
        y = dataset.iloc[:, 2].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # acuracy = accuracy_score(y_pred,y_test)
        # print(acuracy)
        rfDict = {}
        rf_MAE = metrics.mean_absolute_error(y_pred.round(), y_test)
        rf_MSE = metrics.mean_squared_error(y_pred.round(), y_test)
        rf_EVS = metrics.explained_variance_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')
        rf_MedianAE = metrics.median_absolute_error(y_test, y_pred)
        rf_R2Score = metrics.r2_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')

        rfDict.update({'rf_MAE': rf_MAE, 'rf_MSE': rf_MSE, 'rf_EVS': rf_EVS, 'rf_MedianAE': rf_MedianAE,
                       'rf_R2Score': rf_R2Score})

        print("MAE=", rf_MAE)
        print("MSE=", rf_MSE)
        print("RMSE=", np.sqrt(metrics.mean_squared_error(y_pred.round(), y_test)))
        print("Variance Score ", rf_EVS)
        print("Median Absalute Error=", rf_MedianAE)
        print("R2_Score", rf_R2Score)
        return rfDict

    def MyBayesianRidge(self, dataset):
        print("###RadomForest####")
        #print('Have a great day ', dataset)
        #dataset = pd.read_csv(dataset)
        dataset = dataset[
            ['DAY', 'DEPARTURE_TIME', 'FLIGHT_NUMBER', 'DESTINATION_AIRPORT', 'ORIGIN_AIRPORT', 'DAY_OF_WEEK',
             'TAXI_OUT']]
        #print(dataset.head())
        X = dataset.iloc[:, :1].values
        y = dataset.iloc[:, 2].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

        model = BayesianRidge()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # acuracy = accuracy_score(y_pred,y_test)
        # print(acuracy)
        brDict = {}
        br_MAE = metrics.mean_absolute_error(y_pred.round(), y_test)
        br_MSE = metrics.mean_squared_error(y_pred.round(), y_test)
        br_EVS = metrics.explained_variance_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')
        br_MedianAE = metrics.median_absolute_error(y_test, y_pred)
        br_R2Score = metrics.r2_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')

        brDict.update({'br_MAE': br_MAE, 'br_MSE': br_MSE, 'br_EVS': br_EVS, 'br_MedianAE': br_MedianAE,
                       'br_R2Score': br_R2Score})

        print("MAE=", br_MAE)
        print("MSE=", br_MSE)
        print("RMSE=", np.sqrt(metrics.mean_squared_error(y_pred.round(), y_test)))
        print("Variance Score ", br_EVS)
        print("Median Absalute Error=", br_MedianAE)
        print("R2_Score", br_R2Score)
        return brDict

    def MyGradientBoostingRegressor(self, dataset):
        print("###GradientBoostingRegressor####")
        #print('Have a great day ', dataset)
        #dataset = pd.read_csv(dataset)
        dataset = dataset[
            ['DAY', 'DEPARTURE_TIME', 'FLIGHT_NUMBER', 'DESTINATION_AIRPORT', 'ORIGIN_AIRPORT', 'DAY_OF_WEEK',
             'TAXI_OUT']]
        #print(dataset.head())
        X = dataset.iloc[:, :1].values
        y = dataset.iloc[:, 2].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

        model = GradientBoostingRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # acuracy = accuracy_score(y_pred,y_test)
        # print(acuracy)
        gbrDict = {}
        gbr_MAE = metrics.mean_absolute_error(y_pred.round(), y_test)
        gbr_MSE = metrics.mean_squared_error(y_pred.round(), y_test)
        gbr_EVS = metrics.explained_variance_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')
        gbr_MedianAE = metrics.median_absolute_error(y_test, y_pred)
        gbr_R2Score = metrics.r2_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')

        gbrDict.update({'gbr_MAE': gbr_MAE, 'gbr_MSE': gbr_MSE, 'gbr_EVS': gbr_EVS, 'gbr_MedianAE': gbr_MedianAE,
                       'gbr_R2Score': gbr_R2Score})

        print("MAE=", gbr_MAE)
        print("MSE=", gbr_MSE)
        print("RMSE=", np.sqrt(metrics.mean_squared_error(y_pred.round(), y_test)))
        print("Variance Score ", gbr_EVS)
        print("Median Absalute Error=", gbr_MedianAE)
        print("R2_Score", gbr_R2Score)
        return gbrDict




def plot_corr(data_frame, size=11):
    corr = data_frame.corr()  # data frame correlation function
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)  # color code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)), corr.columns)  # draw x tick marks
    plt.yticks(range(len(corr.columns)), corr.columns)  # draw y tick marks

Urls.py
"""FlightDelays URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from FlightDelays import views as mainview
from users import views as usr
from admins import views as admins


urlpatterns = [
    path('admin/', admin.site.urls),
    path('',mainview.index,name='index'),
    path('UserRegister/',mainview.UserRegister, name='UserRegister'),
    path('UserLogin/',mainview.UserLogin, name='UserLogin'),
    path('AdminLogin/', mainview.AdminLogin, name='AdminLogin'),
    path('Logout/', mainview.Logout, name='Logout'),

    ### User Based URLS
    path('UserRegisterAction/',usr.UserRegisterAction, name='UserRegisterAction'),
    path('UserLoginCheck/',usr.UserLoginCheck, name='UserLoginCheck'),
    path('UserUploadForm/', usr.UserUploadForm, name='UserUploadForm'),
    path('UserDataUpload/', usr.UserDataUpload, name='UserDataUpload'),
    path('DataPreProcessing/',usr.DataPreProcessing, name='DataPreProcessing'),
    path('UsermachineLearning/', usr.UsermachineLearning, name='UsermachineLearning'),
    path('UserGraphs/', usr.UserGraphs, name='UserGraphs'),

    ### Admin Based Urls
    path('AdminLoginCheck/',admins.AdminLoginCheck, name='AdminLoginCheck'),
    path('ViewUsers/', admins.ViewUsers, name='ViewUsers'),
    path('AdminActivaUsers/', admins.AdminActivaUsers, name='AdminActivaUsers'),
    path('AdmimnAddData/',admins.AdmimnAddData, name='AdmimnAddData'),
    path('AdminAddingFlightData/',admins.AdminAddingFlightData, name='AdminAddingFlightData'),
    path('AdminViewData/',admins.AdminViewData, name='AdminViewData'),
    path('AdminFindArrivalDelay/', admins.AdminFindArrivalDelay, name='AdminFindArrivalDelay'),
    path('AdminGraphs/',admins.AdminGraphs, name='AdminGraphs'),




]
Adminside Views.py
from django.shortcuts import render,HttpResponse
from django.contrib import messages
from users.models import UserRegistrationModel,FlightDataModel
from .forms import FlightDataForms
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.conf import settings
# Create your views here.
from .CalculationArrivalDelay import ArrivalDelay

def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')

        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})

def ViewUsers(request):
    data = UserRegistrationModel.objects.all()
    return render(request, 'admins/ViewUsers.html',{'data':data})

def AdminActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        UserRegistrationModel.objects.filter(id=id).update(status=status)
        data = UserRegistrationModel.objects.all()
        return render(request,'admins/ViewUsers.html',{'data':data})

def AdmimnAddData(request):
    form = FlightDataForms()
    return render(request,'admins/AddDataForm.html',{'form':form})

def AdminAddingFlightData(request):
    if request.method == 'POST':
        form = FlightDataForms(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'Data Added Successfull')
            form = FlightDataForms()
            return render(request, 'admins/AddDataForm.html', {'form': form})
        else:
            print("Invalid form")
    else:
        form = FlightDataForms()
    return render(request, 'admins/AddDataForm.html', {'form': form})


def AdminViewData(request):
    data_list = FlightDataModel.objects.all()
    page = request.GET.get('page', 1)

    paginator = Paginator(data_list, 60)
    try:
        users = paginator.page(page)
    except PageNotAnInteger:
        users = paginator.page(1)
    except EmptyPage:
        users = paginator.page(paginator.num_pages)

    return render(request, 'admins/AdminViewFlightData.html', {'users': users})

def AdminFindArrivalDelay(request):
    dataset = settings.MEDIA_ROOT + "\\" + 'flightsdata.csv'
    obj = ArrivalDelay()
    lg_dict = obj.MyLogiSticregression(dataset)
    #lg_dict = {}
    dt_dict = obj.MyDecisionTree(dataset)
    rf_dict = obj.MyRandomForest(dataset)
    br_dict = obj.MyBayesianRidge(dataset)
    gbr_dict = obj.MyGradientBoostingRegressor(dataset)

    return render(request, 'admins/AdminMachineLearningRslt.html',
                  {'lg_dict': lg_dict, 'dt_dict': dt_dict, 'rf_dict': rf_dict, 'br_dict': br_dict,
                   'gbr_dict': gbr_dict})


def AdminGraphs(request):
    dataset = settings.MEDIA_ROOT + "\\" + 'flightsdata.csv'
    obj = ArrivalDelay()
    #lg_dict = x.MyLogiSticregression(dataset)
    lg_dict = {}
    dt_dict = obj.MyDecisionTree(dataset)
    rf_dict = obj.MyRandomForest(dataset)
    br_dict = obj.MyBayesianRidge(dataset)
    gbr_dict = obj.MyGradientBoostingRegressor(dataset)

    return render(request, 'admins/AdminGraphs.html',
                  {'lg_dict': lg_dict, 'dt_dict': dt_dict, 'rf_dict': rf_dict, 'br_dict': br_dict,
                   'gbr_dict': gbr_dict})

Calculation ArrivalDelay.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
import numpy as np
class ArrivalDelay:
    def process_data(self,datasetname):
        #dataset = pd.read_csv(datasetname)
        dataset = datasetname
        dataset = dataset[['DAY', 'DEPARTURE_TIME', 'FLIGHT_NUMBER', 'DESTINATION_AIRPORT', 'ORIGIN_AIRPORT', 'DAY_OF_WEEK','TAXI_OUT']]
        #print(dataset.dtypes)
        dataset.fillna
        dataset.dropna()
        dataset = dataset.fillna(0)
        dataset.fillna(method='ffill')
        print(dataset.isnull().values.any())
        #print(dataset.head(10))
        #plot_corr(dataset)
        #plt.show()
        print(dataset.dtypes)
        dataset.to_csv('file1.csv')
        data_dict = dataset.to_dict()
        return data_dict

    def MyLogiSticregression(self,dataset):
        print("###Logistic Regression####")
        #print('Have a great day ',dataset)
        dataset = pd.read_csv(dataset)
        dataset = dataset[['DAY','DEPARTURE_TIME','FLIGHT_NUMBER','ARRIVAL_DELAY','DESTINATION_AIRPORT','ORIGIN_AIRPORT','DAY_OF_WEEK','TAXI_OUT']]
        #print(dataset.head())
        dataset.fillna
        dataset.dropna()
        dataset = dataset.fillna(0)
        X = dataset.iloc[:,:3].values
        y = dataset.iloc[:,2].values
        X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=1/3,random_state=0)

        model = LogisticRegression()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)

        #acuracy = accuracy_score(y_pred,y_test)
        #print(acuracy)
        lgDict = {}
        lg_MAE = metrics.mean_absolute_error(y_pred.round(), y_test)
        lg_MSE = metrics.mean_squared_error(y_pred.round(), y_test)
        lg_EVS  = metrics.explained_variance_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')
        lg_MedianAE = metrics.median_absolute_error(y_test, y_pred)
        lg_R2Score = metrics.r2_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')

        lgDict.update({'lg_MAE':round(lg_MAE,2),'lg_MSE':round(lg_MSE,2),'lg_EVS':round(lg_EVS,2),'lg_MedianAE':round(lg_MedianAE,2),'lg_R2Score':round(lg_R2Score,2)})

        print("MAE=",lg_MAE )
        print("MSE=",lg_MSE )
        print("RMSE=", np.sqrt(metrics.mean_squared_error(y_pred.round(), y_test)))
        print("Variance Score ",lg_EVS)
        print("Median Absalute Error=",lg_MedianAE)
        print("R2_Score", lg_R2Score)

        return lgDict

    def MyDecisionTree(self, dataset):
        print("###Decesion Treee####")
        #print('Have a great day ', dataset)
        dataset = pd.read_csv(dataset)
        dataset = dataset[['DAY','DEPARTURE_TIME','FLIGHT_NUMBER','ARRIVAL_DELAY','DESTINATION_AIRPORT','ORIGIN_AIRPORT','DAY_OF_WEEK','TAXI_OUT']]
        #print(dataset.head())
        dataset.fillna
        dataset.dropna()
        dataset = dataset.fillna(0)
        X = dataset.iloc[:, :3].values
        y = dataset.iloc[:, 2].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # acuracy = accuracy_score(y_pred,y_test)
        # print(acuracy)
        dtDict = {}
        dt_MAE = metrics.mean_absolute_error(y_pred.round(), y_test)
        dt_MSE = metrics.mean_squared_error(y_pred.round(), y_test)
        dt_EVS =  metrics.explained_variance_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')
        dt_MedianAE = metrics.median_absolute_error(y_test, y_pred)
        dt_R2Score = metrics.r2_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')

        dtDict.update({'dt_MAE':round(dt_MAE,2),'dt_MSE':round(dt_MSE,2),'dt_EVS':round(dt_EVS,2),'dt_MedianAE':round(dt_MedianAE,2),'dt_R2Score':round(dt_R2Score,2)})

        print("MAE=", dt_MAE)
        print("MSE=", dt_MAE)
        print("RMSE=", np.sqrt(metrics.mean_squared_error(y_pred.round(), y_test)))
        print("Variance Score ",dt_EVS)
        print("Median Absalute Error=", dt_MedianAE)
        print("R2_Score", dt_R2Score)
        return dtDict

    def MyRandomForest(self, dataset):
        print("###RadomForest####")
        #print('Have a great day ', dataset)
        dataset = pd.read_csv(dataset)
        dataset = dataset[
            ['DAY', 'DEPARTURE_TIME', 'FLIGHT_NUMBER', 'ARRIVAL_DELAY', 'DESTINATION_AIRPORT', 'ORIGIN_AIRPORT',
             'DAY_OF_WEEK', 'TAXI_OUT']]
        # print(dataset.head())
        dataset.fillna
        dataset.dropna()
        dataset = dataset.fillna(0)
        X = dataset.iloc[:, :3].values
        y = dataset.iloc[:, 2].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # acuracy = accuracy_score(y_pred,y_test)
        # print(acuracy)
        rfDict = {}
        rf_MAE = metrics.mean_absolute_error(y_pred.round(), y_test)
        rf_MSE = metrics.mean_squared_error(y_pred.round(), y_test)
        rf_EVS = metrics.explained_variance_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')
        rf_MedianAE = metrics.median_absolute_error(y_test, y_pred)
        rf_R2Score = metrics.r2_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')

        rfDict.update({'rf_MAE': round(rf_MAE,2), 'rf_MSE': round(rf_MSE,2), 'rf_EVS': round(rf_EVS,2), 'rf_MedianAE': round(rf_MedianAE,2),
                       'rf_R2Score': round(rf_R2Score,2)})

        print("MAE=", rf_MAE)
        print("MSE=", rf_MSE)
        print("RMSE=", np.sqrt(metrics.mean_squared_error(y_pred.round(), y_test)))
        print("Variance Score ", rf_EVS)
        print("Median Absalute Error=", rf_MedianAE)
        print("R2_Score", rf_R2Score)
        return rfDict

    def MyBayesianRidge(self, dataset):
        print("###RadomForest####")
        #print('Have a great day ', dataset)
        dataset = pd.read_csv(dataset)
        dataset = dataset[
            ['DAY', 'DEPARTURE_TIME', 'FLIGHT_NUMBER', 'ARRIVAL_DELAY', 'DESTINATION_AIRPORT', 'ORIGIN_AIRPORT',
             'DAY_OF_WEEK', 'TAXI_OUT']]
        # print(dataset.head())
        dataset.fillna
        dataset.dropna()
        dataset = dataset.fillna(0)
        X = dataset.iloc[:, :3].values
        y = dataset.iloc[:, 2].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

        model = BayesianRidge()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # acuracy = accuracy_score(y_pred,y_test)
        # print(acuracy)
        brDict = {}
        br_MAE = metrics.mean_absolute_error(y_pred.round(), y_test)
        br_MSE = metrics.mean_squared_error(y_pred.round(), y_test)
        br_EVS = metrics.explained_variance_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')
        br_MedianAE = metrics.median_absolute_error(y_test, y_pred)
        br_R2Score = metrics.r2_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')

        brDict.update({'br_MAE': round(br_MAE,2), 'br_MSE': round(br_MSE,2), 'br_EVS': round(br_EVS,2), 'br_MedianAE': round(br_MedianAE,2),
                       'br_R2Score': round(br_R2Score,2)})

        print("MAE=", br_MAE)
        print("MSE=", br_MSE)
        print("RMSE=", np.sqrt(metrics.mean_squared_error(y_pred.round(), y_test)))
        print("Variance Score ", br_EVS)
        print("Median Absalute Error=", br_MedianAE)
        print("R2_Score", br_R2Score)
        return brDict

    def MyGradientBoostingRegressor(self, dataset):
        print("###GradientBoostingRegressor####")
        #print('Have a great day ', dataset)
        dataset = pd.read_csv(dataset)
        dataset = dataset[
            ['DAY', 'DEPARTURE_TIME', 'FLIGHT_NUMBER', 'ARRIVAL_DELAY', 'DESTINATION_AIRPORT', 'ORIGIN_AIRPORT',
             'DAY_OF_WEEK', 'TAXI_OUT']]
        # print(dataset.head())
        dataset.fillna
        dataset.dropna()
        dataset = dataset.fillna(0)
        X = dataset.iloc[:, :3].values
        y = dataset.iloc[:, 2].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

        model = GradientBoostingRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # acuracy = accuracy_score(y_pred,y_test)
        # print(acuracy)
        gbrDict = {}
        gbr_MAE = metrics.mean_absolute_error(y_pred.round(), y_test)
        gbr_MSE = metrics.mean_squared_error(y_pred.round(), y_test)
        gbr_EVS = metrics.explained_variance_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')
        gbr_MedianAE = metrics.median_absolute_error(y_test, y_pred)
        gbr_R2Score = metrics.r2_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')

        gbrDict.update({'gbr_MAE': round(gbr_MAE,2), 'gbr_MSE': round(gbr_MSE,2), 'gbr_EVS': round(gbr_EVS,2), 'gbr_MedianAE': round(gbr_MedianAE,2),
                       'gbr_R2Score': round(gbr_R2Score,2)})

        print("MAE=", gbr_MAE)
        print("MSE=", gbr_MSE)
        print("RMSE=", np.sqrt(metrics.mean_squared_error(y_pred.round(), y_test)))
        print("Variance Score ", gbr_EVS)
        print("Median Absalute Error=", gbr_MedianAE)
        print("R2_Score", gbr_R2Score)
        return gbrDict




def plot_corr(data_frame, size=11):
    corr = data_frame.corr()  # data frame correlation function
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)  # color code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)), corr.columns)  # draw x tick marks
    plt.yticks(range(len(corr.columns)), corr.columns)  # draw y tick marks

models.py
from django.db import models


# Create your models here.

class UserRegistrationModel(models.Model):
    name = models.CharField(max_length=100)
    loginid = models.CharField(unique=True, max_length=100)
    password = models.CharField(max_length=100)
    mobile = models.CharField(max_length=100)
    email = models.CharField(max_length=100)
    locality = models.CharField(max_length=100)
    address = models.CharField(max_length=1000)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    status = models.CharField(max_length=100)

    def __str__(self):
        return self.loginid

    class Meta:
        db_table = 'AviationUsers'


class FlightDataModel(models.Model):
    DAY =models.IntegerField(default=0)
    DEPARTURE_TIME =models.FloatField(default=0.0)
    FLIGHT_NUMBER =models.IntegerField(default=0)
    DESTINATION_AIRPORT =models.CharField(max_length=100)
    ORIGIN_AIRPORT =models.CharField(max_length=100)
    DAY_OF_WEEK =models.IntegerField(default=0)
    TAXI_OUT =models.FloatField(default=0.0)
    def __str__(self):
        return self.id

    class Meta:
        db_table = "FlighDelayData"

