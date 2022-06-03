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
	ko

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

	# pearson
	corr = myungi_1_2.corr()
	#print(corr)
	


if __name__ == '__main__':
	myungi1()
	#koges1()
