# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 18:52:33 2018

@author: zhcao
"""

from pyspark.sql import SparkSession
import pyspark.sql.types as typ
import pyspark.ml.feature as ft

if __name__ == "__main__":
    
    spark = SparkSession.builder.appName("PCATest").getOrCreate()
    
    labels = [
            ('Num', typ.IntegerType()),
            ('VIB1', typ.FloatType()),
            ('VIB2', typ.FloatType())
    ]
    schema = typ.StructType([
            typ.StructField(e[0], e[1], False) for e in labels        
    ])
    
    data = spark.read.csv('file:///home/hadoop/zhcao/workspace/spark_test/watermelon.csv',
                          header = True,
                          schema = schema)
    data.createOrReplaceTempView("data_temp")
    data.printSchema()
    data.cache()
    
    featuresCreator = ft.VectorAssembler(
            inputCols = ['VIB1', 'VIB2'],
            outputCol = 'features'
    )

    data = featuresCreator.transform(data)
    data.show(3, False)
    print(type(data))

    pca = ft.PCA(k = 1,
                 inputCol = 'features',
                 outputCol = 'pca_features'
                )

    model = pca.fit(data)
    result = model.transform(data)

    result.collect()
    result.show(10, False)

    spark.stop()

