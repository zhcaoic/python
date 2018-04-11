# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 19:40:41 2018

@author: zhcao
"""

from pyspark.sql import SparkSession
import pyspark.sql.types as typ
import pyspark.ml.feature as ft
from pyspark.ml import Pipeline


if __name__ == "__main__":
    
    spark = SparkSession.builder.appName("PCATest2").getOrCreate()
    
    labels = [
            ('Num', typ.IntegerType()),
            ('VIB1', typ.FloatType()),
            ('VIB2', typ.FloatType())
    ]
    schema = typ.StructType([
            typ.StructField(e[0], e[1], False) for e in labels        
    ])
    
    data = spark.read.csv("file:///home/hadoop/zhcao/workspace/spark_test/watermelon.csv",
                          header = True,
                          schema = schema)
    data.createOrReplaceTempView("data_temp")
    data.printSchema()
    data.cache()
    data.show()
    
    #
    featuresCreator = ft.VectorAssembler(
            inputCols = ['VIB1', 'VIB2'],
            outputCol = 'features'
    )
    
    pca = ft.PCA(k = 1,
                 inputCol = 'features',
                 outputCol = 'pca_features'
                )
    
    pipeline = Pipeline(stages = [
            featuresCreator,
            pca
    ])
    
    model = pipeline.fit(data)
    
    result = model.transform(data)
    print(type(result))
    
    result.collect()
    print(type(result))
    
    result.show(10, False)
    
    spark.stop()

