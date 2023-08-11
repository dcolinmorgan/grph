import os
from collections import Counter
import cProfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pstats import Stats
import cudf
from time import time
import warnings
warnings.filterwarnings('ignore')
import dask_cudf as dc
from dask.utils import parse_bytes
import dask
from dask.delayed import delayed
import rmm
import pandas as pd
import dask.dataframe as dd
pd.set_option('display.max_colwidth', 200)

import dirty_cat, umap

os.chdir('/home/graphistry/notebooks/daniel/pygraphistry')
import graphistry
os.chdir('..')
graphistry.register(api=3,protocol="https", server="hub.graphistry.com", username='dcolinmorgan', password='fXjJnkE3Gik6BWy')
graphistry.__version__

# os.chdir('cu-cat')
# import cu_cat
# os.chdir('..')
import dirty_cat, umap

pool = rmm.mr.PoolMemoryResource(
    rmm.mr.ManagedMemoryResource(),
    initial_pool_size=parse_bytes("12GB"),
    maximum_pool_size=parse_bytes("14GB")
)
rmm.mr.set_current_device_resource(pool) #https://docs.rapids.ai/api/rmm/stable/basics.html

# filtered_df=ddf # pd.read_csv("../../data-pv/sample-state-with-severity.csv").drop(columns=["Unnamed: 0"])
ddf=dc.read_csv("../../data-pv/ddf_gby/*")

A=pd.DataFrame(columns=['state','drug','gpu','dc-cuml'])


states = ["AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
for s in states:
    ddf=dc.read_csv("../../data-pv/ddf_gby/*")
    state_df = ddf[ddf["BUYER_STATE"].isin([s])]
    filtered_state_df=state_df.compute().to_pandas()
    for d in pd.unique(filtered_state_df.DRUG_NAME):
        filtered_state_df["DRUG_NAME"] = filtered_state_df["DRUG_NAME"].str.replace('\t', '', regex=True)
        filtered_df = filtered_state_df[(filtered_state_df['DRUG_NAME']==d)]

        filtered_df=filtered_df.rename(columns={"('CALC_BASE_WT_IN_GM', 'sum')":"calc_weight_per_month_grams","('DOSAGE_UNIT', 'sum')":"pills_per_month" })
        filtered_df["month"]=pd.to_datetime(filtered_df["month_date"],format='%Y-%m').dt.month
        filtered_df["year"]=pd.to_datetime(filtered_df["month_date"],format='%Y-%m').dt.year
        filtered_df.month_date = filtered_df.month_date.values.astype('datetime64[M]')
        filtered_df=filtered_df.set_index("month_date")
        filtered_df = filtered_df.sort_index()

        filtered_df["trailing six month avg"]=filtered_df.groupby(["BUYER_DEA_NO","DRUG_NAME","dos_str","month_date"])['pills_per_month'].transform("sum").rolling("183d",min_periods=1).mean()
        filtered_df["yearly_average"]=filtered_df.groupby(["BUYER_DEA_NO","DRUG_NAME","dos_str","year"])['pills_per_month'].transform('mean')
        filtered_df['2*TRAILING12']=2*filtered_df.groupby(["BUYER_DEA_NO","DRUG_NAME","dos_str","month_date"])['pills_per_month'].transform("sum").rolling(window="365D",min_periods=1).mean()
        filtered_df['3*TRAILING12']=3*filtered_df.groupby(["BUYER_DEA_NO","DRUG_NAME","dos_str","month_date"])['pills_per_month'].transform("sum").rolling(window="365D",min_periods=1).mean()
        filtered_df["Max monthly dosage units"]=8000
        filtered_df=filtered_df.reset_index()
        filtered_df["exceeds_max_monthly_dosage_unit_threshhold"] = filtered_df['pills_per_month'] > filtered_df["Max monthly dosage units"]
        filtered_df["exceeds_trailing_6_month_avg"] = filtered_df['pills_per_month'] > filtered_df["trailing six month avg"]
        filtered_df["exceeds_3*TRAILING12_threshold"] = filtered_df['pills_per_month'] > filtered_df['3*TRAILING12']
        filtered_df["exceeds_2*TRAILING12_threshold"] = filtered_df['pills_per_month'] > filtered_df['2*TRAILING12']
        filtered_df["exceeds_yearly_avg"] = filtered_df['pills_per_month'] > filtered_df['yearly_average']


        sev_lst = dict()
        for index in filtered_df.index:
            sev_lst[index] = (filtered_df["exceeds_max_monthly_dosage_unit_threshhold"][index].sum() +
                              filtered_df["exceeds_trailing_6_month_avg"][index].sum() +
                              (2*filtered_df["exceeds_3*TRAILING12_threshold"][index].sum()) +
                              filtered_df["exceeds_2*TRAILING12_threshold"][index].sum() +
                              filtered_df["exceeds_yearly_avg"][index].sum())/5
        filtered_df["severity_level"] = sev_lst
        
        
        
        rows=filtered_df.groupby(['month','year','BUYER_DEA_NO','dos_str']).sum().reset_index()
        
        
        t=time()
        # g1 = g.umap(feature_engine='cu_cat', engine='cuml',memoize=False)
        t1=time()-t

        # t=time()
        # g2 = g.umap(feature_engine='dirty_cat', engine='cuml',memoize=False)
        # t2=time()-t
        # print(['dirtycat+cuml time:'+str(t3)])
        g = graphistry.nodes((rows))
        t=time()
        g1=g.featurize(feature_engine='dirty_cat',memoize=True)
        # g2 = g1.umap(engine='cuml')
        t3=time()-t
        
        df2 = pd.DataFrame([[s,d,t1,t3,rows.shape]], columns=['state','drug','gpu','dc-cuml','dims'], index=[d+'_'+s])
        print(df2)
        A=A.append(df2)
        df2.to_csv('cpu_feat_loop_bench.txt',sep='\t',mode='a')
        del g3, ddf,filtered_state_df,state_df
A.to_csv('cpu_full_bench2.txt',sep='\t')