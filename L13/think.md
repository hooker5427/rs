#### 当我们思考数据源的时候，都有哪些维度，如果你想要使用爬虫抓取数据，都有哪些工具 ？
+ 数据的来源:
    - 政府网站，比如统计局等
    - 企业数据共享 ，比如天池，kaggle等比赛数据集
    - 公开数据集 比如imageNet, Coco数据集，Mnist等
    - 开源社区 ,比如github别人采集的数据、
    - 爬虫 ，自己抓取互联网上数据
+ 爬虫工具
    - urllib(python内置)，requests库，简单易用
    - scrapy 爬虫框架，异步高并发， 且有分布式版本 scrapy-redis 
    - selenium 爬取动态网页
    - 八爪鱼等工具，优点在于不用写代码（或者编写少量的xpath表达式）


#### 什么是时序数据库？为什么时间序列数据成为增长最快的数据类型之一？ 
+ 按照维基百科解释，时间序列数据库（TSDB）是一个为了用于处理时间序列数据而优化的软件系统，其按时间数值或时间范围进行索引。
+ 时间序列数据
    随着大数据时代的到来，数据的采集越来越丰富多彩。数据的形式往往伴随着时间属性，比如金融数据，交通流量,传感器数据，智能运维数据等。在数据随着时间而增长的过程中，时间序列数据成为增长最快的数据类型之一。


#### 开源是当前重要的Trend，我们使用的statsmodels.tsa，tensorflow/keras都是开源工具
+ AI相关的开源工具？
    - tensorflow ,pytorch , keras 等深度学习框架
    - lightgbm/xgboost/libfm/skleran等机器学习库
+ 阿里，微软，百度 都有哪些和AI相关的开源工具（包括LightGBM）
+   - 微软: lightgbm , learning to rank（ranklib）
    - 谷歌:tensorflow  ，tensorflow-hub 
    - 百度:paddle ，百度AI平台也可以调一些借口

+ 了解和使用这些工具，对于我们有哪些价值？
    + 快速上手，实现项目
    + 阅读源码进行学习(比如deepctr工具)


