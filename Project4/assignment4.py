#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from numpy import asarray
import pandas as pd
frequent_item_list=list()
max_confidence_list=list()
min_confidence_list=list()
for i in range(1,6):
    pat1_series= pd.read_csv('DataFolder//CGMSeriesLunchPat'+str(i)+'.csv')
    pat1_bolus= pd.read_csv('DataFolder//InsulinBolusLunchPat'+str(i)+'.csv')
    pat_series=pat1_series
    pat_bolus=pat1_bolus

    pat_series.fillna(0,inplace=True)
    pat_bolus.fillna(0.0,inplace=True)
    pat1_b=pat_bolus.to_numpy()
    bolus=list()
    for ele in pat1_b:

        bolus.append(max(ele))
    cgmm=list()
    cgm0=list()
    pat1_s=pat_series.to_numpy()
    for ele in pat1_s:
        cgmm.append(max(ele))
        cgm0.append(ele[5])
    def create_bins(lower_bound, width, quantity):
        """ create_bins returns an equal-width (distance) partitioning.
            It returns an ascending list of tuples, representing the intervals.
            A tuple bins[i], i.e. (bins[i][0], bins[i][1])  with i > 0
            and i < quantity, satisfies the following conditions:
                (1) bins[i][0] + width == bins[i][1]
                (2) bins[i-1][0] + width == bins[i][0] and
                    bins[i-1][1] + width == bins[i][1]
        """

        bins = []
        for low in range(lower_bound,
                         lower_bound + quantity * width + 1, width):
            bins.append((low, low + width))
        return bins

    bins = create_bins(lower_bound=40,
                       width=10,
                       quantity=36)


    def find_bin(value, bins):
        """ bins is a list of tuples, like [(0,20), (20, 40), (40, 60)],
            binning returns the smallest index i of bins so that
            bin[i][0] <= value < bin[i][1]
        """

        for i in range(0, len(bins)):
            if value==0:
                return 0
            elif bins[i][0] < value <= bins[i][1]:
                return i+1
        return -3

    binned_weights = []
    bin_result=[]
    #bin_dataframe=pd.DataFrame()
    for value in cgmm:
        bin_index = find_bin(value, bins)
        bin_result.append(bin_index)
    sum=0
    total=len(cgmm)
    for i in range(0,len(cgmm)):
        if cgmm[i]==bin_result[i]:
            sum=sum+1
    #print(sum/total)
    #print("carb")
    #print(carbsdatalabel)
    cgmm_bins=bin_result 
    binned_weights = []
    bin_result=[]
    #bin_dataframe=pd.DataFrame()
    for value in cgm0:
        bin_index = find_bin(value, bins)
        bin_result.append(bin_index)

    sum=0
    total=len(cgm0)
    for i in range(0,len(cgm0)):
        if cgm0[i]==bin_result[i]:
            sum=sum+1

    cgm0_bins=bin_result

    import numpy as np
    x=list()
    for i in range(len(cgmm_bins)):
        x.append(['cgmm'+str(cgmm_bins[i]),'cgm0'+str(cgm0_bins[i]),bolus[i]])
    x=np.array(x)

    x_list=x
    from apyori import apriori
    association_rules = apriori(x_list,min_support=0.01,min_confidence=0)
    association_results = list(association_rules)
    frequent_itemsets=list()
    fi=list()
    for ele in association_results:
        if len(ele[0])==3:
            x=sorted([list(ele[0])[0],list(ele[0])[1],list(ele[0])[2]],reverse=True)
            fi.append([int(x[0][4:]),int(x[1][4:]),float(x[2])])
    fi_final=list()
    for ele in fi:
        fi_final.append(['{'+str(ele[0])+','+str(ele[1])+','+str(ele[2])+'}'][0])
    for ele in fi_final:
        frequent_item_list.append(ele)
    # freq=pd.DataFrame(fi_final)
    # freq.to_csv('frequent_itemsets.csv',index=False,header=None,mode='a')

    ma=0
    mi=2
    m_c=list()
    for ele in association_results:
        if len(ele[2])==7:
            if ele[2][6][2]>ma:
                ma=ele[2][6][2]
            if ele[2][6][2]<mi:
                mi=ele[2][6][2]
    for ele in association_results:
        if len(ele[2])==7:
            if ele[2][6][2]==ma:
                m_c.append([int(list(ele[2][6][0])[0][4:]),int(list(ele[2][6][0])[1][4:]),float(list(ele[2][6][1])[0])])
    m_final=list()
    for ele in m_c:
        m_final.append(['{'+str(ele[0])+','+str(ele[1])+'->'+str(ele[2])+'}'][0])
    for ele in m_final:
        max_confidence_list.append(ele)

    anomalous=list()
    if mi==ma:
        mi=2
    for ele in association_results:
        if len(ele[2])==7:
            if ele[2][6][2]==mi:
                anomalous.append([int(list(ele[2][6][0])[0][4:]),int(list(ele[2][6][0])[1][4:]),float(list(ele[2][6][1])[0])])
    anomalous_final=list()
    for ele in anomalous:
        anomalous_final.append(['{'+str(ele[0])+','+str(ele[1])+'->'+str(ele[2])+'}'][0])
    for ele in anomalous_final:
        min_confidence_list.append(ele)

freq=pd.DataFrame(frequent_item_list)
freq.to_csv('frequent_itemsets.csv',index=False,header=None)
mf=pd.DataFrame(max_confidence_list)
mf.to_csv('most_confident_rules.csv',index=False,header=None)
af=pd.DataFrame(min_confidence_list)
af.to_csv('anomalous_rules.csv',index=False,header=None)
# In[ ]:




