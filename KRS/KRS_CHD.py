import pandas as pd
import numpy as np
import math

def factor_calc(fn):
    def CHD_factor(*args):
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
                
    return CHD_factor


@factor_calc
def Age_factor(x):
    age_factor = [0.13759*(x-45.7991), 0.12962*(x-45.5808)]    
    return age_factor


@factor_calc
def Age_Square(x):
    age_factor = [-0.0006964*(x*x-2186.58), -0.0003965*(x*x-2363.65)]    
    return age_factor


@factor_calc
def Smoking_factor(x):
    smoking_factor = []
    if x == 1:
        smoking_factor = [0.0, 0.0]        
    elif x == 2:
        smoking_factor = [-0.00207*(1-0.23029), 0.23099*(1-0.03970)]
        
    elif x == 3:
        smoking_factor = [0.60138*(1-0.53016), 0.67653*(1-0.05079)]
        
    return smoking_factor


@factor_calc
def Total_Chol(x):
    tc_factor = []
    if x < 200 and x >= 160:
        tc_factor = [0.30303*(math.log(x)-0.43540), 0.20005*(math.log(x)-0.41642)]
        
    elif x < 240 and x >= 200:
        tc_factor = [0.72508*(math.log(x)-0.31439), 0.44176*(math.log(x)-0.29841)]
        
    elif x < 280 and x >= 240:
        tc_factor = [1.02770*(math.log(x)-0.09640), 0.52267*(math.log(x)-0.09640)]
        
    elif x >= 280:
        tc_factor = [1.51018*(math.log(x)-0.01387), 1.035735*(math.log(x)-0.02196)]
        
    elif x < 160:
        tc_factor = [0.0, 0.0]
        
    return tc_factor


@factor_calc
def HDL_factor(x):
    hdl_factor = []
    if x < 45 and x >= 35:
        hdl_factor = [-0.41580*(math.log(x)-0.31063), -0.28121*(math.log(x)-0.18651)]
        
    elif x < 50 and x >= 45:
        hdl_factor = [-0.59809*(math.log(x)-0.22692), -0.18543*(math.log(x)-0.16015)]
        
    elif x < 60 and x >= 50:
        hdl_factor = [-0.80256*(math.log(x)-0.27050), -0.47018*(math.log(x)-0.30597)]
        
    elif x >= 60:
        hdl_factor = [-1.13973*(math.log(x)-0.11410), -0.72046*(math.log(x)-0.31451)]
        
    elif x < 35:
        hdl_factor = [0.0, 0.0]
                
    return hdl_factor


@factor_calc
def DM_factor(x):
    dm_factor = []
    if x == 1:
        dm_factor = [0.49443*(1-0.08389), 0.58729*(1-0.06026)]
        
    elif x == 0:
        dm_factor = [0, 0]
                
    return dm_factor


@factor_calc
def HT_factor(x):
    ht_factor = []
    if x < 140 and x >= 120:
        ht_factor = [0.24130*(math.log(x)-0.40678), 0.41491*(math.log(x)-0.32308)]
        
    elif x < 160 and x >= 140:
        ht_factor = [0.54176*(math.log(x)-0.18005), 0.66187*(math.log(x)-0.14102)]
        
    elif x >= 160:
        ht_factor = [0.79091*(math.log(x)-0.06823), 1.10282*(math.log(x)-0.06657)]
        
    elif x < 120:
        ht_factor = [0.0, 0.0]
        
    return ht_factor


def KRS(x1, x2):    
    '''
    x1: KRS SUM
    x2 : 성별
    return : KRS score, 결측치가 포함되어 잘못된 경우에는 999999
    '''
    if x2 == 1:
        try:
            score = 1 - math.pow(0.99313, math.exp(x1))
            return score
        except OverflowError as e:
            return 999999         
    elif x2 == 2:
        try:
            score = 1 - math.pow(0.99815, math.exp(x1)) 
            return score
        except OverflowError as e:
            return 999999
    else:
        return 999999
    


def main():
    KoGES = pd.read_csv('KoGES_total_v1.csv')
    CHD_Col = ['DIST_ID', 'SEX', 'AGE', 'HTN', 'DM', 'TCHL_ORI', 'HDL_ORI', 'SMOKE']
    KoGES_DB = KoGES[CHD_Col]
    
    AGE = KoGES_DB.apply(lambda x: Age_factor(x['AGE'], x['SEX']), axis=1)
    AGESQ = KoGES_DB.apply(lambda x: Age_Square(x['AGE'], x['SEX']), axis=1)
    SMOKING = KoGES_DB.apply(lambda x: Smoking_factor(x['SMOKE'], x['SEX']), axis=1)
    TOT_CHOL = KoGES_DB.apply(lambda x: Total_Chol(x['TCHL_ORI'], x['SEX']), axis=1)
    HDL = KoGES_DB.apply(lambda x: HDL_factor(x['HDL_ORI'], x['SEX']), axis=1)
    DM = KoGES_DB.apply(lambda x: DM_factor(x['DM'], x['SEX']), axis=1)
    HTN = KoGES_DB.apply(lambda x: HT_factor(x['HTN'], x['SEX']), axis=1)
    
    x_sum = AGE + AGESQ + SMOKING + TOT_CHOL + HDL + DM + HTN
    KoGES_DB['KRS_SUM'] = x_sum
    KoGES_DB.apply(lambda x: KRS(x['KRS_SUM'], x['SEX']), axis=1)
    
    KoGES_DB['KRS_SCORE'] = KoGES_DB.apply(lambda x: KRS(x['KRS_SUM'], x['SEX']), axis=1)
    KoGES_DB.to_csv("./KoGES_CHD.csv", encoding='utf-8', header=True, index=False)
    
    # Apply to Original Data
    KoGES['KRS_SCORE'] = KoGES_DB.apply(lambda x: KRS(x['KRS_SUM'], x['SEX']), axis=1)
    KoGES.to_csv("./KoGES_CHD_score.csv", encoding='utf-8', header=True, index=False)
    
    # Filtering
    KoGES_filter1 = KoGES[KoGES['KRS_SCORE'].isna()==False].copy()
    KoGES_filter2 = KoGES_filter1[KoGES_filter1['KRS_SCORE'] < 99999]
    KoGES_filter2.to_csv("./KoGES_fil_CHD_score.csv", encoding='utf-8', header=True, index=False)
    
    
    
if __name__=="__main__":
    main()
    
    
    
    
    