# -*- coding: utf-8 -*-
# @Time    : 2020/3/2 23:28
# @Author  : hooker5427


import requests
import json
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

text = requests.get("https://lab.isaaclin.cn/nCoV/api/area", verify=False).text

alldata = json.loads(text, encoding='utf-8')
data = alldata.get('results')

infos = [] 
for line  in data :
    try :
        for city in line['cities']:
            info ={}
            for  k in  city.keys() :
                if k != "cityName" :
                    info["city_"+k] =  city[k]
                else :
                    info[k] = city[k]
            others = [ k for k in line.keys() if k != "cities"]
            for k in others :
                info[k] = line[k]
            infos.append(info)
    except Exception :
        # 有点粗暴
        pass 


import  csv 
def savetocsv(data , savepath ) :
    with open (savepath  ,'w' , newline= "" ,  encoding= "utf-8") as file : 
        fieldnames = data[0].keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for line in  data :
            writer.writerow(line)

savetocsv( infos , "area.csv" )

