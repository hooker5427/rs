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
for ele in data:

    info = {}
    for key in ['continentName', "countryName", 'provinceName', \
                'confirmedCount', 'confirmedCount', 'suspectedCount', \
                'curedCount', 'deadCount', 'updateTime']:
        info[key] = ele.get(key)

        try:
            city_keys = ["cityName", "currentConfirmedCount",\
                         "confirmedCount", "confirmedCount",\
                         "curedCount", "deadCount", "locationId"]
            for city in ele.get('cities'):
                for key in city_keys:
                    if key == "cityName":
                        info[key] = city.get(key)
                    else:
                        info["city_" + key] = city.get(key)
        except Exception:
            pass
        infos.append(info)

data = pd.DataFrame(infos)
columns = ['continentName', "countryName", 'provinceName', 'confirmedCount', 'confirmedCount',
           'suspectedCount', 'curedCount', 'deadCount', 'updateTime'] + \
          ["cityName", "city_currentConfirmedCount", "city_confirmedCount", "city_confirmedCount",
           "city_curedCount", "city_deadCount", "city_locationId"]
data.to_csv("area.csv", columns=columns, index=None, )
