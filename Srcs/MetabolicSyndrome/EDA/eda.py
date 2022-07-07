import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# koges dataset
ko = pd.read_csv('/Data2/Dataset/KoGES/KoGES_followup/KoGES_total.csv')
#print("koges ===== ", ko)

# myungi dataset
myungi = pd.read_excel('/Data2/Dataset/Myongji/Myongji_2019-2021.xlsx', sheet_name = ['3년','2년'],engine='openpyxl')
#print("myungi =====", myungi)

# koges dataset
def koges1():
	print(" ")

# myungi dataset
def myungi1():
	myungi_1 = myungi['2년']
	#print(myungi_1)
	
	# change english
	myungi_english = myungi_1.rename(columns={'성별MF':'SEX', '나이(년)':'AGE', '음주(빈칸이면 비음주)':'DRINK',
                       '담배(1이면 노담배, 2이면 현재흡연)':'SMOKE', '고강도운동(0이면 운동안함)':'HARDEXERCUR',
                       '중강도운동(0이면 운동안함)':'SOFTEXERCUR', '1.고혈압약물치료(1이면약물치료)':'TREATD1',
                       '1.당뇨 약물치료(1이면 약물치료)':'TREATD2', '혈당(식전)':'GLU0_ORI',
                       '총콜레스테롤':'TCHL_ORI', 'HDL-콜레스테롤':'HDL_ORI', '중성지방(TG)':'TRIGLY_ORI',
                       '허리둘레(단위:cm)':'WAIST', '체질량지수(BMI)':'BMI', '혈압(최고)':'SBP'}, inplace=True)
	# drink null change
	myungi_1.DRINK.replace(np.nan,'0', inplace=True)
	# nan column delete
	myungi_1_2 = myungi_1.drop(['Unnamed: 0','Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18'],axis=1)
	#print(myungi_1_2)
	
	# change type = 'DRINK' column
	myungi_1_2['DRINK'] = myungi_1_2['DRINK'].astype(float)
	
	# xgboost
	from xgboost import XGBClassifier
	from xgboost import plot_importance
	import pandas as pd
	import numpy as np
	from sklearn.datasets import load_breast_cancer
	from sklearn.model_selection import train_test_split
	import warnings
	warnings.filterwarnings('ignore')
	from sklearn.preprocessing import LabelEncoder

	X = myungi_1_2.drop(['SEX'], axis = 1)
	y = myungi_1_2['SEX']
	encoder = LabelEncoder()
	encoder.fit(y)
	labels = encoder.transform(y)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 156)

	evals = [X_test, y_test]
	xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate = 0.1, max_depth=3)

	xgb_wrapper.fit(X_train, y_train)

	w_pred = xgb_wrapper.predict(X_test)

	w_pred = xgb_wrapper.predict_proba(X_test)[:, 1]
	
	model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_valid, y_valid)], 
												early_stopping_rounds = 100, verbose = False)

	

if __name__ == '__main__':
	myungi1()
	#koges1()
