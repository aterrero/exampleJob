from __future__ import print_function
import sys
from pyspark.sql import SparkSession,SQLContext
from pyspark.sql.functions import col, greatest, least, when, mean
import random
import numpy as np


spark = SparkSession\
        .builder\
        .appName("nbatest")\
        .getOrCreate()

    #Data loading and preprocessing
data = spark.read.csv("/nbashottest3.csv")
data2 = data.rdd.map(lambda x: (x._c19, x._c13, x._c8, x._c11, x._c16)).toDF()
data3 = data2.fillna("0")

    #construction of initial centroids; centroids are [shotclock,shotdist,closedefdist]
centroid1 =np.array([random.uniform(0,24),random.uniform(0,47.2),random.uniform(0,53.2)])
centroid2 =np.array([random.uniform(0,24),random.uniform(0,47.2),random.uniform(0,53.2)])
centroid3 =np.array([random.uniform(0,24),random.uniform(0,47.2),random.uniform(0,53.2)])
centroid4 =np.array([random.uniform(0,24),random.uniform(0,47.2),random.uniform(0,53.2)])

len = data3.count()
previouspredictions = ['c0' for i in range(len)]
counter = 0

for i in range(0,30):
    counter = counter + 1
    progress = (float(counter)/float(30))*100
#    print("this is iteration ", counter)
    print("Progress:",progress,"%")

    #distance computations
    data4 = data3.withColumn('distc1',((data3._3 - centroid1[0])**2 + (data3._4 - centroid1[1])**2 + (data3._5 - centroid1[2])**2)**0.5 )
    data5 = data4.withColumn('distc2',((data3._3 - centroid2[0])**2 + (data3._4 - centroid2[1])**2 + (data3._5 - centroid2[2])**2)**0.5 )
    data6 = data5.withColumn('distc3',((data3._3 - centroid3[0])**2 + (data3._4 - centroid3[1])**2 + (data3._5 - centroid3[2])**2)**0.5 )
    data7 = data6.withColumn('distc4',((data3._3 - centroid4[0])**2 + (data3._4 - centroid4[1])**2 + (data3._5 - centroid4[2])**2)**0.5 )
    data8 = data7.select(data7._1 \
                           , data7._2 \
                           , data7._3 \
                           , data7._4 \
                           , data7._5 \
                           , data7.distc1 \
                           , data7.distc2 \
                           , data7.distc3 \
                           , data7.distc4 \
                           , least("distc1","distc2","distc3","distc4").alias('mindis'))

    #assigning clusters
    data9 = data8.withColumn('prediction',when(data8.mindis == data8.distc1,"c1") \
                                             .when(data8.mindis == data8.distc2,"c2") \
                                             .when(data8.mindis == data8.distc3,"c3") \
                                             .otherwise("c4"))

    newpredictions = data9.select("prediction").rdd.flatMap(lambda x: x).collect()
    newpredictions = [str(s) for s in newpredictions]

    if newpredictions == previouspredictions:
        break
    else:
        previouspredictions = newpredictions

    #creating SQLContext and registering previous df as a table to build new centroids
    sqlcontext = SQLContext(spark)
    sqlcontext.registerDataFrameAsTable(data9,"data9")

    #building new centroid1
    c1df = sqlcontext.sql("SELECT  * FROM data9 WHERE data9.prediction = 'c1'")
    c1dfmeans = c1df.select(mean(col("_3")).alias("c1shotclockmean") \
                              , mean(col("_4")).alias("c1shotdistmean") \
                              , mean(col("_5")).alias("c1closedefmean")).collect()

    c1shotclock = c1dfmeans[0]["c1shotclockmean"]
    c1shotdist = c1dfmeans[0]["c1shotdistmean"]
    c1closedef = c1dfmeans[0]["c1closedefmean"]
    centroid1 = np.array([c1shotclock,c1shotdist,c1closedef])

    #building new centroid2
    c2df = sqlcontext.sql("SELECT  * FROM data9 WHERE data9.prediction = 'c2'")
    c2dfmeans = c2df.select(mean(col("_3")).alias("c2shotclockmean") \
                              , mean(col("_4")).alias("c2shotdistmean") \
                              , mean(col("_5")).alias("c2closedefmean")).collect()

    c2shotclock = c2dfmeans[0]["c2shotclockmean"]
    c2shotdist = c2dfmeans[0]["c2shotdistmean"]
    c2closedef = c2dfmeans[0]["c2closedefmean"]
    centroid2 = np.array([c2shotclock,c2shotdist,c2closedef])

    #building new centroid3
    c3df = sqlcontext.sql("SELECT  * FROM data9 WHERE data9.prediction = 'c3'")
    c3dfmeans = c3df.select(mean(col("_3")).alias("c3shotclockmean") \
                              , mean(col("_4")).alias("c3shotdistmean") \
                              , mean(col("_5")).alias("c3closedefmean")).collect()

    c3shotclock = c3dfmeans[0]["c3shotclockmean"]
    c3shotdist = c3dfmeans[0]["c3shotdistmean"]
    c3closedef = c3dfmeans[0]["c3closedefmean"]
    centroid3 = np.array([c3shotclock,c3shotdist,c3closedef])

    #building new centroid 4
    c4df = sqlcontext.sql("SELECT  * FROM data9 WHERE data9.prediction = 'c4'")
    c4dfmeans = c4df.select(mean(col("_3")).alias("c4shotclockmean") \
                              , mean(col("_4")).alias("c4shotdistmean") \
                              , mean(col("_5")).alias("c4closedefmean")).collect()

    c4shotclock = c4dfmeans[0]["c4shotclockmean"]
    c4shotdist = c4dfmeans[0]["c4shotdistmean"]
    c4closedef = c4dfmeans[0]["c4closedefmean"]
    centroid4 = np.array([c4shotclock,c4shotdist,c4closedef])

print("Kmeans clustering converged after ",counter," iterations")

spark.stop()
