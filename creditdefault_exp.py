## Import Packages
# Spark DF profiling and preparation
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType, DoubleType, ShortType, DecimalType
import pyspark.sql.functions as func
from pyspark.sql.functions import isnull
from pyspark.sql.functions import mean
from pyspark.sql.types import Row
import matplotlib.pyplot as plt
plt.ioff()

# Pandas DF operation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling + Evaluation
from pyspark.ml.feature import VectorAssembler, VectorIndexer, OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder 
from sklearn.metrics import roc_curve, auc

import cdsw

# # Preparation
# Create Spark Session
spark = SparkSession.builder.appName("Credit Scoring Example").getOrCreate()

# Read Data from HDFS
df = spark.read.csv('/tmp/fajar/bank-class.csv', inferSchema=False, header=True)

# Column rename and type casting
df = df.withColumn("LIMIT_BAL", df["LIMIT_BAL"].cast(IntegerType()))
df = df.withColumn("AGE", df["AGE"].cast(IntegerType()))
df = df.withColumn("DELAY1", df["DELAY1"].cast(IntegerType()))
df = df.withColumn("DELAY2", df["DELAY2"].cast(IntegerType()))
df = df.withColumnRenamed("TARGET", "label")                        # Change column name 'TARGET' to 'label' to ease modelling purpose
df = df.withColumn("label", df["label"].cast(IntegerType()))
df = df.withColumn("BILLING1", df["BILLING1"].cast(DoubleType()))
df = df.withColumn("BILLING2", df["BILLING2"].cast(DoubleType()))
df = df.withColumn("PAYMENT1", df["PAYMENT1"].cast(DoubleType()))
df = df.withColumn("PAYMENT2", df["PAYMENT2"].cast(DoubleType()))

LIMIT_BAL_means = df.select(mean("LIMIT_BAL")).collect()[0][0]
df = df.na.fill({"LIMIT_BAL": LIMIT_BAL_means})
df = df.na.drop(subset=("MARITAL", "EDUCATION", "SEX"))

# String to Index Conversion
MARITALindexer  = StringIndexer(inputCol="MARITAL", outputCol="MARITALindex")
EDUCATIONindexer= StringIndexer(inputCol="EDUCATION", outputCol="EDUCATIONindex")
SEXindexer      = StringIndexer(inputCol="SEX", outputCol="SEXindex")


# Index to Binary Vector Conversion
MARITALencoder  = OneHotEncoder(inputCol="MARITALindex", outputCol="MARITALvec")
EDUCATIONencoder= OneHotEncoder(inputCol="EDUCATIONindex", outputCol="EDUCATIONvec")
SEXencoder      = OneHotEncoder(inputCol="SEXindex", outputCol="SEXvec")

# Create features vector
assembler = VectorAssembler(inputCols=["LIMIT_BAL","MARITALvec","EDUCATIONvec","SEXvec","AGE","DELAY1","DELAY2","BILLING1","BILLING2","PAYMENT1","PAYMENT2"],
                            outputCol="features")
                            
# Split Train-Test
train, test = df.randomSplit([0.7, 0.3], seed=1000)

## Modeling

# Set Parameter
param_numTrees = int(sys.argv[1]) #10
param_maxDepth = int(sys.argv[2]) #5
param_impurity = sys.argv[3] #'gini'

cdsw.track_metric("numTrees",param_numTrees)
cdsw.track_metric("maxDepth",param_maxDepth)
cdsw.track_metric("impurity",param_impurity)

# Declare RF Model and Pipeline
rf = RandomForestClassifier(labelCol = 'label', 
                                    featuresCol = 'features', 
                                    numTrees = param_numTrees, 
                                    maxDepth = param_maxDepth,  
                                    impurity = param_impurity)
pipe_rf = Pipeline(stages=[MARITALindexer,EDUCATIONindexer,SEXindexer,MARITALencoder,EDUCATIONencoder,SEXencoder,assembler,rf])     
model = pipe_rf.fit(train)

predictions = model.transform(test)
evaluator = BinaryClassificationEvaluator()
auroc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
aupr = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})
"The AUROC is %s and the AUPR is %s" % (auroc, aupr)

cdsw.track_metric("auroc", auroc)
cdsw.track_metric("aupr", aupr)

model.write().overwrite().save("models/spark")

!ls
!rm -r -f models/spark
!rm -r -f models/spark_rf.tar
!ls
!hdfs dfs -get models/spark models/
!tar -cvf models/spark_rf.tar models/spark

cdsw.track_file("models/spark_rf.tar")

spark.stop()
