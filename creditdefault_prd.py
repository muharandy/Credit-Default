from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml import PipelineModel

spark = SparkSession.builder \
      .appName("Credit Default Prediction") \
      .master("local[*]") \
      .getOrCreate()
  
model = PipelineModel.load("file:///home/cdsw/models/spark") 

features = ["LIMIT_BAL","MARITAL","EDUCATION","SEX","AGE","DELAY1","DELAY2","BILLING1","BILLING2","PAYMENT1","PAYMENT2"]
                            
def predict(args):
  account = args["param"].split(",")
  feature = spark.createDataFrame([map(float,account[:1]) + account[1:4] + map(float,account[4:12])], features)
  result = model.transform(feature).collect()[0].prediction
  return {"result" : result}

# arg= {"param":"30000,Married,GradSch,Female,40,0,0,24607,24430,1700,1600"}
# print(predict(arg))