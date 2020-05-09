import  pandas as pd
import warnings
warnings.filterwarnings('ignore')
import json


train_path ="../v2/train_sample.csv"
test_path ="../v2/test_sample.csv"

train_data=  pd.read_csv( train_path )
test_data= pd.read_csv(test_path )

print ( len(train_data))

train_data.drop('Unnamed: 0' ,axis =1 , inplace= True)
test_data.drop('Unnamed: 0' ,axis =1 , inplace= True)
test_data.insert( 0 , 'click' , 0 )
tr_ts = pd.concat([test_data, train_data], axis=0).reset_index()


def convert(x , topk , threshold ) :
    if topk:
        print (" topk filter " , x.name  )
        category ={}
        v_counts = x.value_counts().index.tolist()[:topk]
        v_counts.append("others")
        for idx , value  in enumerate(x):
            value_bak = value
            if value not in v_counts :
                x[idx] = 'others'
                value_bak = 'others'
            index = v_counts.index( value_bak )
            category.setdefault( str(index) ,[])
            category[str(index)].append(value)
        json_path =x.name+ '_converted.csv'
        json.dump( category ,  open( json_path , 'w'))
        print ("处理 "+ x.name +" 成功")
        print( "写入" ,json_path , '成功')
    else :
        print ("threhold filter " , x.name )
        category ={}
        i_list= x.value_counts().index.tolist()
        i_list.append("others")
        v_counts = x.value_counts()
        for idx , value  in enumerate(x):
            if   v_counts[value] >=threshold :
                index = i_list.index( value )
                category.setdefault( str(index) ,[])
                category[str(index)].append(value)
            else :
                x[idx] = 'others'
                index = i_list.index( 'others' )
                category.setdefault( str(index) ,[])
                category[str(index)].append(value)
        json_path =x.name+ '_converted.csv'
        json.dump( category ,  open( json_path , 'w'))
        print ("处理 "+ x.name +" 成功")
        print( "写入" ,json_path , '成功')

# 详细分析过程见Notebook
convert(tr_ts.app_id , topk= 1 ,threshold=None )
convert(tr_ts.device_id , topk=1 , threshold= None )
convert(tr_ts.app_domain , topk= None , threshold= 750 )
convert(tr_ts.app_category , topk= 1 , threshold= None )
convert(tr_ts.site_category ,topk= None ,  threshold= 200 )
convert(tr_ts.device_ip , None , 200 )
convert(tr_ts.device_model , topk=None ,threshold= 500 )
convert(tr_ts.site_domain , topk=None , threshold=500 )
convert(tr_ts.banner_pos , topk=2 , threshold=None )

# 时间处理

tr_ts['hour'] =tr_ts['hour'].astype("str")
tr_ts['day'] = tr_ts['hour'].apply(lambda x: x[-4:-2])
tr_ts['hour'] = tr_ts['hour'].apply(lambda x: x[-2:])


tr_ts.banner_pos  = tr_ts['banner_pos'].astype("str")
tr_ts.device_ip  = tr_ts['device_ip'].astype("str")
tr_ts.app_category  = tr_ts['app_category'].astype("str")

from sklearn import preprocessing
lenc = preprocessing.LabelEncoder()
C_fields = [ 'hour', 'day' ,'device_ip' ,'banner_pos', 'site_category', 'app_domain', 'app_category',
            'device_conn_type', 'C14',"C15" ,'C16' ,'C18', 'C19', 'C20','C21',
            'device_id', 'app_id', 'site_id','site_domain', 'device_model', 'device_type']
for f, column in enumerate(C_fields):
    print("convert " + column + "...")
    tr_ts[column] = lenc.fit_transform(tr_ts[column])


dummies_site_category = pd.get_dummies(tr_ts['site_category'], prefix = 'site_category')
dummies_app_category = pd.get_dummies(tr_ts['app_category'], prefix = 'app_category')



tr_ts_new = pd.concat([tr_ts, dummies_site_category, dummies_app_category], axis=1)
tr_ts_new.drop(['site_category', 'app_category'], axis = 1, inplace=True)
tr_ts_new.id = tr_ts['id'].astype("int")

tr_ts_new.iloc[:test_data.shape[0],].to_csv('test_features.csv')
tr_ts_new.iloc[test_data.shape[0]:,].to_csv('train_features.csv')


print ('ok!')