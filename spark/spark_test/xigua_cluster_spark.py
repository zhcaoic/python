# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 19:07:02 2018

@author: zhcao
"""

from pyspark.sql import SparkSession
import pyspark.sql.types as typ
import pyspark.ml.feature as ft
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline

if __name__ == "__main__":
    
    spark = SparkSession.builder.appName("XiGuaKMeans").getOrCreate()
    
    labels = [
            ('Num', typ.IntegerType()),
            ('VIB1', typ.FloatType()),
            ('VIB2', typ.FloatType())
    ]
    
    schema = typ.StructType([
            typ.StructField(e[0], e[1], False) for e in labels
    ])
    
    data = spark.read.csv("file:///home/hadoop/zhaco/workspace/spark_test/watermelon.csv",
                               header = True,
                               schema = schema)
    
    data.createOrReplaceTempView("data_clu")
    data.printSchema()
    data.cache()
    data.show()
    
    featuresCreator = ft.VectorAssembler(inputCols = ['VIB1', 'VIB2'], outputCol = 'features')
    
    kmeans = KMeans(featuresCol = 'features', k = 3, seed = 1)
    
    pipeline = Pipeline(stages = [
            featuresCreator,
            kmeans
    ])
    
    model = pipeline.fit(data)
    result = model.transform(data)
    print(type(result))
    
    result.collect()
    result.show(10, False)
    
    spark.stop()
