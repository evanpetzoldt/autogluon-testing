from functools import reduce
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

def doublelift(df, y_model1, y_model2, y_actual, model_type="additive", weight=None, p_tile=10, y_denom=None, 
               rescale=False, y_name="Agg"):
    '''Creates a Double Lift Chart given 2 model prediction sets'''
    
    global df_DblLft
    global df_ChartAgg
    global mergelist
    if y_model1==y_model2: 
        df.rename(columns={y_model1: y_model1 + "_model1"}, inplace=True)
        df.rename(columns={y_model2: y_model2 + "_model2"}, inplace=True)
    
    mergelist=[y_model1, y_model2, y_actual]
    if y_denom is not None: 
        mergelist.append(y_denom)
    if weight is not None: 
        mergelist.append(weight)
           
    df_DblLft=df[mergelist].copy()
    print(df_DblLft.head())
 
    if weight is None:
        df_DblLft["weight"]=1
        weight="weight"
    df_DblLft["ModelsRatio"]=df_DblLft[y_model1]/df_DblLft[y_model2]
    df_DblLft.sort_values(by="ModelsRatio", inplace=True)

    df_DblLft["runtot_weight"]=df_DblLft[weight].cumsum()
    
    df_DblLft["p_tile"]=pd.qcut(df_DblLft["runtot_weight"],p_tile, labels=False)
        
    ###Aggregate to display level
    
    if model_type=="additive":
        df_ChartAgg=df_DblLft.groupby("p_tile").agg(model1_agg=(y_model1, 'mean'),
                                                model2_agg=(y_model2, 'mean'),
                                                actual_agg=(y_actual, 'mean')                                                 
                                                )
    
    
    ###Handle Ratio Style
    if model_type=="ratio": 
        df_DblLft["y_actual_num"]=df_DblLft[y_actual]*df_DblLft[y_denom]
        df_DblLft["y_model1_num"]=df_DblLft[y_model1]*df_DblLft[y_denom]
        df_DblLft["y_model2_num"]=df_DblLft[y_model2]*df_DblLft[y_denom]
        df_ChartAgg=df_DblLft.groupby("p_tile").agg(model1_num_agg=("y_model1_num", sum),
                                                model2_num_agg=("y_model2_num", sum),
                                                actual_num_agg=("y_actual_num", sum),
                                                y_denom_agg=(y_denom, sum)    
                                                )
        
        df_ChartAgg["model1_agg"]=df_ChartAgg.model1_num_agg/df_ChartAgg.y_denom_agg
        df_ChartAgg["model2_agg"]=df_ChartAgg.model2_num_agg/df_ChartAgg.y_denom_agg     
        df_ChartAgg["actual_agg"]=df_ChartAgg.actual_num_agg/df_ChartAgg.y_denom_agg
    

    df_ChartAgg=df_ChartAgg.filter(items=["actual_agg", "model1_agg", "model2_agg", "p_tile"])
        
    ###Handle Rescaling
    if rescale==True: 
        df_ChartAgg.model1_agg= df_ChartAgg.model1_agg/df_ChartAgg.actual_agg
        df_ChartAgg.model2_agg= df_ChartAgg.model2_agg/df_ChartAgg.actual_agg
        df_ChartAgg.actual_agg=1
        
        
    ### Create Title Names
    TitleDict={5:"Quintile", 10:"Decile", 25:"Quartile"}
    if p_tile in TitleDict.keys():
        Title= TitleDict[p_tile]
    else: 
        Title=p_tile + "%'iles"
    if y_name==None: y_name=model_type
    
    df_ChartAgg.reset_index(inplace=True)
    if y_name!="Agg": 
        for colname in df_ChartAgg.columns:
            df_ChartAgg.rename(columns={colname:colname.replace('_agg','_'+y_name)}, inplace=True)
    
    FinalDf=df_ChartAgg.melt("p_tile",var_name='Model',value_name=y_name)
    FinalDf["p_tile"]=FinalDf["p_tile"]+1
    plt.xlabel(Title)
    
    DblLftCht= sns.catplot(x="p_tile",y=y_name,hue="Model",data=FinalDf,kind='point')
    DblLftCht.set_xlabels(Title)
    plt.title("Double Lift Chart")
    
    return DblLftCht