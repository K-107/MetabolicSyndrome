import pandas as pd
import numpy as np
import math

def Exer_level(x1, x2):
    ## 0: No exercise, 1: Medium, 2: High
    if x1 == 2:
        if x2 == 2:
            return int(2)
        elif x2 == 1:
            return int(1)
    else:
        return int(0)

    
def drink_level(x1, x2, x3, x4):
    if x1 < 3:
        return int(0)
    else:
        if x4 == 1 and x3 <= 65:
            ## 65세 이하 남성
            if x2 > 42:
                # 음주량 42g 기준
                return int(2)
            elif x2 <= 42:
                return int(1)
        elif x4 == 1 and x3 > 65:
            if x2 > 28:
                return int(2)
            elif x2 <= 28:
                return int(1)
        elif x4 == 2 and x3 <= 65:
            if x2 > 28:
                return int(2)
            elif x2 <= 28:
                return int(1)
        elif x4 == 2 and x3 > 65:
            if x2 > 14:
                return int(2)
            elif x2 <= 14:
                return int(1)
        
            
def factor_calc(fn):
    def diabetes_factor(*args):
        factor = []
        factor += fn(args[0])
        
        if args[1] > 2 or args[0] > 1000 or args[1] < 1:
            return 99999
        else:
            try:
                return factor[args[1]-1]
            except IndexError as e:
                print(e)
                print(args[0])
                print(args[1])
                
    return diabetes_factor


@factor_calc
def BMI_factor(x):
    bmi_factor = []
    if x < 18.5:
        bmi_factor = [0.07343423, 0.01107232]
    elif x < 23 and x >= 18.5:
        bmi_factor = [0.0, 0.0]
    elif x < 25 and x >= 23:
        bmi_factor = [0.33567685, 0.32325289]
    elif x < 30 and x >= 25:
        bmi_factor = [0.65489924, 0.64505894]
    elif x >= 30:
        bmi_factor = [1.25414597, 1.06455937]
    
    return bmi_factor


@factor_calc
def Smoking_factor(x):
    s_factor = []
    if x == 1:
        s_factor = [0.0, 0.0]
    elif x == 2:
        s_factor = [0.06879353, 0.23742451]
    elif x == 3:
        s_factor = [0.39075797, 0.27228951]
    
    return s_factor


@factor_calc
def FH_factor(x):    
    return [0.40539468*(x-1), 0.45270876*(x-1)]


@factor_calc
def Age_factor(x):
    return [0.03429806*x, 0.03267195*x]


@factor_calc
def Alcohol_factor(x):
    alc_factor = []
    if x == 2:
        alc_factor = [0.05447258, 0.0]
    elif x == 1:
        alc_factor = [-0.08385316, 0.0]
    elif x == 0:
        alc_factor = [0.0, 0.0]
    return alc_factor


@factor_calc
def Exer_factor(x):
    ex_factor = []
    if x == 2:
        ex_factor = [-0.13733219, -0.02239017]
    elif x == 1:
        ex_factor = [-0.12335606, -0.06736686]
    elif x == 0:
        ex_factor = [0.0, 0.0]
    return ex_factor
        
    
@factor_calc
def anti_HTN(x):
    drug = []
    if x == 1:
        drug = [0.0, 0.0]
    elif x == 2:
        drug = [0.32880315, 0.38249871]
    
    return drug


@factor_calc
def Statin(x):
    drug = []
    if x == 1:
        drug = [0.0, 0.0]
    elif x == 2:
        drug = [0.26482252, 0.31324724]
    
    return drug


@factor_calc
def HT_factor(x):
    ht_factor = []
    if x < 100:
        ht_factor = [-0.12135365, -0.17698000]
    elif x < 120 and x >= 100:
        ht_factor = [0.0, 0.0]
    elif x < 140 and x >= 120:
        ht_factor = [0.15236789, 0.18910012]
    elif x < 160 and x >= 140:
        ht_factor = [0.25680655, 0.29230610]
    elif x >= 160:
        ht_factor = [0.34104001, 0.36250700]
        
    return ht_factor


@factor_calc
def TC_factor(x):
    tc_factor = []
    if x < 240:
        tc_factor = [0.0, 0.0]
    elif x >= 240:
        tc_factor = [0.17977916, 0.10497807]
    
    return tc_factor


@factor_calc
def FSG_factor(x):
    fsg_factor = []
    if x < 70:
        fsg_factor = [-0.06092024, -0.02473416]
    elif x < 85 and x >= 70:
        fsg_factor = [0.0, 0.0]
    elif x < 100 and x >= 85:
        fsg_factor = [0.21862977, 0.22737061]
    elif x < 110 and x >= 100:
        fsg_factor = [0.69352537, 0.78101158]
    elif x >= 110:
        fsg_factor = [1.29451275, 1.34396826]
    
    return fsg_factor


@factor_calc
def Log_GGT(x):
    return [0.0, math.log(x)*0.48718396]



def KRS(x1, x2):    
    '''
    x1: KRS SUM
    x2 : 성별
    return : KRS score, 결측치가 포함되어 잘못된 경우에는 999999
    '''
    if x2 == 1:
        try:
            score = 1 - math.pow(0.966054, math.exp(x1-1.755900))
            return score
        except OverflowError as e:
            return 999999         
    elif x2 == 2:
        try:
            score = 1 - math.pow(0.974148, math.exp(x1-3.079393)) 
            return score
        except OverflowError as e:
            return 999999
    else:
        return 999999



def main():
    '''
    Calculate Korean Risk Score based on Ha et al (2018) https://doi.org/10.4093/dmj.2018.0014
    
    '''
    KoGES = pd.read_csv('KoGES_total_v1.csv')
    Diabetes_Col = ['DIST_ID', 'SEX', 'AGE', 'DRINK', 'TOTALC', 'EXER', 'EXERCUR', 'DRUGHT', 'DRUGLP', 'HEIGHT', 'WEIGHT', 'FMDM', 'R_GTP_ORI', 'GLU0_ORI', 'TCHL_ORI', 'SBP_L', 'SBP_R', 'SMOKE']
    KoGES_DB = KoGES[Diabetes_Col]
    
    ######## Add New Columns #############
    # BMI
    KoGES_DB['BMI'] = KoGES['WEIGHT'] / ((KoGES['HEIGHT']/100)**2)
    # Mean Systolic Blood Pressure
    KoGES_DB['SBP'] = (KoGES['SBP_L'] + KoGES['SBP_R']) / 2
    # Excercise Level
    KoGES_DB['EXERCISE'] = KoGES.apply(lambda x: Exer_level(x['EXERCUR'], x['EXER']), axis=1)
    # Drink Level
    KoGES_DB['ALCOHOL'] = KoGES.apply(lambda x: drink_level(x['DRINK'], x['TOTALC'], x['AGE'], x['SEX']), axis=1)
    
    
    AGE = KoGES_DB.apply(lambda x: Age_factor(x['AGE'], x['SEX']), axis=1)
    BMI = KoGES_DB.apply(lambda x: BMI_factor(x['BMI'], x['SEX']), axis=1)
    SMOKING = KoGES_DB.apply(lambda x: Smoking_factor(x['SMOKE'], x['SEX']), axis=1)
    PA = KoGES_DB.apply(lambda x: Exer_factor(x['EXERCISE'], x['SEX']), axis=1)
    FH = KoGES_DB.apply(lambda x: FH_factor(x['FMDM'], x['SEX']), axis=1)
    ALCOHOL = KoGES_DB.apply(lambda x: Alcohol_factor(x['ALCOHOL'], x['SEX']), axis=1)
    HTN = KoGES_DB.apply(lambda x: HT_factor(x['SBP'], x['SEX']), axis=1)
    TC = KoGES_DB.apply(lambda x: TC_factor(x['TCHL_ORI'], x['SEX']), axis=1)
    FSG = KoGES_DB.apply(lambda x: FSG_factor(x['GLU0_ORI'], x['SEX']), axis=1)
    STA = KoGES_DB.apply(lambda x: Statin(x['DRUGLP'], x['SEX']), axis=1)
    ANT_HTN = KoGES_DB.apply(lambda x: anti_HTN(x['DRUGHT'], x['SEX']), axis=1)
    GGT = KoGES_DB.apply(lambda x: Log_GGT(x['R_GTP_ORI'], x['SEX']), axis=1)
    
    
    x_sum = AGE + BMI + SMOKING + PA + FH + ALCOHOL + HTN + TC + FSG + STA + ANT_HTN + GGT
    KoGES_DB['KRS_SUM'] = x_sum
    KoGES_DB.apply(lambda x: KRS(x['KRS_SUM'], x['SEX']), axis=1)
    
    KoGES_DB['KRS_SCORE'] = KoGES_DB.apply(lambda x: KRS(x['KRS_SUM'], x['SEX']), axis=1)    
    KoGES_DB.to_csv("./KoGES_Diabetes.csv", encoding='utf-8', header=True, index=False)
    
    # Apply to Original Data
    KoGES['KRS_SCORE'] = KoGES_DB.apply(lambda x: KRS(x['KRS_SUM'], x['SEX']), axis=1)
    KoGES.to_csv("./KoGES_Diabetes_score.csv", encoding='utf-8', header=True, index=False)
    
    # Filtering
    KoGES_filter = KoGES[KoGES['KRS_SCORE'] < 99999]
    KoGES_filter.to_csv("./KoGES_filter_Diabetes_score.csv", encoding='utf-8', header=True, index=False)
    
    
    
if __name__=="__main__":
    main()
    
    
    
    
    