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

# # Preparation
# Create Spark Session
spark = SparkSession.builder.appName("Credit Scoring Example").getOrCreate()

# Read Data from HDFS
df = spark.read.csv('/tmp/fajar/bank-class.csv', inferSchema=False, header=True)
type(df)

df.show(5)
df.printSchema()

# Column rename and type casting
df = df.withColumn("LIMIT_BAL", df["LIMIT_BAL"].cast(IntegerType()))
df = df.withColumn("AGE", df["AGE"].cast(IntegerType()))
df = df.withColumn("DELAY1", df["DELAY1"].cast(IntegerType()))
df = df.withColumn("DELAY2", df["DELAY2"].cast(IntegerType()))
df = df.withColumnRenamed("default payment next month", "label")                        # Change column name 'TARGET' to 'label' to ease modelling purpose
df = df.withColumnRenamed("MARRIAGE", "MARRIAGE")  
df = df.withColumn("label", df["TARGET"].cast(IntegerType()))

df = df.withColumn("BILLING1", df["BILLING1"].cast(DoubleType()))
df = df.withColumn("BILLING2", df["BILLING2"].cast(DoubleType()))
df = df.withColumn("PAYMENT1", df["PAYMENT1"].cast(DoubleType()))
df = df.withColumn("PAYMENT2", df["PAYMENT2"].cast(DoubleType()))

df = df.select("ID","LIMIT_BAL","MARITAL","EDUCATION","SEX","AGE","DELAY1","DELAY2","BILLING1","BILLING2","PAYMENT1","PAYMENT2","label")

df.printSchema()

rowNum = df.count()
colNum = len(df.columns)

print("Data after type conversion :", "Column Number = %i" %colNum, ", Row Number = %i" %rowNum)
df.show(10)

# # Data Profiling
# Show dimension statistics
df.select(["LIMIT_BAL","AGE","BILLING1","BILLING2","PAYMENT1","PAYMENT2"]).describe().show()

df.groupBy("MARITAL").count().show()
df.groupBy("EDUCATION").count().show()
df.groupBy("SEX").count().show()
df.groupBy("DELAY1").count().show() 
df.groupBy("DELAY2").count().show() 
df.groupBy("label").count().show()

# Missing value analysis
df_missing = df.select(isnull("LIMIT_BAL").alias("LIMIT_BAL-miss"), 
                        isnull("MARITAL").alias("MARITAL-miss"), 
                        isnull("EDUCATION").alias("EDUCATION-miss"),
                        isnull("SEX").alias("SEX-miss"), 
                        isnull("AGE").alias("AGE-miss"), 
                        isnull("DELAY1").alias("DELAY1-miss"), 
                        isnull("DELAY2").alias("DELAY2-miss"),
                        isnull("BILLING1").alias("BILLING1-miss"), 
                        isnull("BILLING2").alias("BILLING2-miss"), 
                        isnull("PAYMENT1").alias("PAYMENT1-miss"), 
                        isnull("PAYMENT2").alias("PAYMENT2-miss"), 
                        isnull("label").alias("label-miss"))

df_missing.groupBy("LIMIT_BAL-miss").count().show()
df_missing.groupBy("MARITAL-miss").count().show(), 
df_missing.groupBy("EDUCATION-miss").count().show(), 
df_missing.groupBy("SEX-miss").count().show(), 
df_missing.groupBy("AGE-miss").count().show(), 
df_missing.groupBy("DELAY1-miss").count().show(), 
df_missing.groupBy("DELAY2-miss").count().show(), 
df_missing.groupBy("BILLING1-miss").count().show(), 
df_missing.groupBy("BILLING2-miss").count().show(), 
df_missing.groupBy("PAYMENT1-miss").count().show(), 
df_missing.groupBy("PAYMENT2-miss").count().show(), 
df_missing.groupBy("label-miss").count().show()

LIMIT_BAL_means = df.select(mean("LIMIT_BAL")).collect()[0][0]
print("LIMIT_BAL means = %.2f" %LIMIT_BAL_means)
df = df.na.fill({"LIMIT_BAL": LIMIT_BAL_means})

df = df.na.drop(subset=("MARITAL", "EDUCATION", "SEX"))

rowNum = df.count()
colNum = len(df.columns)

print("Data dimension after missing value handling :", "Column Number = %i" %colNum, ", Row Number = %i" %rowNum )

df_pd = df.sample(False, 0.1, 1000).toPandas()
type(df_pd)

# # Data Profiling Visualization
plt.figure(figsize=(10,3))
plt.subplot(121)
df_pd["LIMIT_BAL"].plot.hist(bins=15, title="LIMIT_BAL")
plt.subplot(122)
df_pd["AGE"].plot.hist(bins=15, title="Age")
plt.show()

plt.figure(figsize=(20,3))
plt.subplot(141)
df_pd["PAYMENT1"].plot.hist(bins=20, title="PAYMENT1 -  Jan")
plt.subplot(142)
df_pd["PAYMENT2"].plot.hist(bins=20, title="PAYMENT2 - Feb")
plt.subplot(143)
df_pd["BILLING1"].plot.hist(bins=20, title="BILLING1 - Jan")
plt.subplot(144)
df_pd["BILLING2"].plot.hist(bins=20, title="BILLING2 - Feb")
plt.show()

plt.figure(figsize=(20,3))
plt.subplot(141)
df_pd["PAYMENT1"].apply(np.cbrt).plot.hist(bins=20, title="PAYMENT1 - Jan (cbrt)")
plt.subplot(142)
df_pd["PAYMENT2"].apply(np.cbrt).plot.hist(bins=20, title="PAYMENT2 - Feb (cbrt)")
plt.subplot(143)
df_pd["BILLING1"].apply(np.cbrt).plot.hist(bins=20, title="BILLING1 - Jan (cbrt)")
plt.subplot(144)
df_pd["BILLING2"].apply(np.cbrt).plot.hist(bins=20, title="BILLING2 - Feb (cbrt)")
plt.show()

plt.figure(figsize=(16,3))
plt.subplot(131)
df_pd[["LIMIT_BAL","BILLING1","BILLING2"]].boxplot(sym='r*', grid=False)
plt.subplot(132)
df_pd[["PAYMENT1","PAYMENT2"]].boxplot(sym='b+', grid=True)
plt.subplot(133)
df_pd[["AGE"]].boxplot(sym='g-*', grid=False)
plt.show()

plt.figure(figsize=(25,4))
plt.subplot(161)
df_pd["MARITAL"].value_counts().sort_index().plot.bar(rot=0, title="MARITAL Status")
plt.subplot(162)
df_pd["SEX"].value_counts().sort_index().plot.bar(rot=0, title="SEX")
plt.subplot(163)
df_pd["EDUCATION"].value_counts().sort_index().plot.bar(rot=0, title="EDUCATION")
plt.subplot(164)
df_pd["DELAY1"].value_counts().sort_index().plot.bar(rot=0, title="DELAY Jan")
plt.subplot(165)
df_pd["DELAY2"].value_counts().sort_index().plot.bar(rot=0, title="DELAY Feb")
plt.subplot(166)
df_pd["label"].value_counts(normalize=True).plot.bar(rot=0, title="label Proportion")
plt.show()


plt.figure(figsize=(10,3))
plt.subplot(121)
df_pd.groupby("label").LIMIT_BAL.plot.density(alpha=0.5, legend=True, title="LIMIT_BAL VS label")
plt.subplot(122)
df_pd.groupby("label").AGE.plot.density(alpha=0.5, legend=True, title="AGE VS label")
plt.show()

plt.figure(figsize=(30,4))
plt.subplot(141)
df_pd.groupby("label").PAYMENT1.plot.density(alpha=0.5, legend=True, title="Payment Jan VS label")
plt.subplot(142)
df_pd.groupby("label").PAYMENT2.plot.density(alpha=0.5, legend=True, title="Payment Feb VS label")
plt.subplot(143)
df_pd.groupby("label").BILLING1.plot.density(alpha=0.5, legend=True, title="Billing Jan VS label")
plt.subplot(144)
df_pd.groupby("label").BILLING1.plot.density(alpha=0.5, legend=True, title="Billing Feb VS label")
plt.show()

plt.figure(figsize=(30,4))
df_pd2 = df_pd.copy(deep=True)
plt.subplot(141)
df_pd2["PAYMENT1"]=np.cbrt(df_pd2["PAYMENT1"])
df_pd2.groupby("label").PAYMENT1.plot.density(alpha=0.5, legend=True, title="PAYMENT Jan VS label")
plt.subplot(142)
df_pd2["PAYMENT2"]=np.cbrt(df_pd2["PAYMENT2"])
df_pd2.groupby("label").PAYMENT2.plot.density(alpha=0.5, legend=True, title="PAYMENT Feb VS label")
plt.subplot(143)
df_pd2["BILLING1"]=np.cbrt(df_pd2["BILLING1"])
df_pd2.groupby("label").BILLING1.plot.density(alpha=0.5, legend=True, title="BILLING Jan VS label")
plt.subplot(144)
df_pd2["BILLING2"]=np.cbrt(df_pd2["BILLING2"])
df_pd2.groupby("label").BILLING1.plot.density(alpha=0.5, legend=True, title="BILLING Feb VS label")
plt.show()

pd.crosstab(df_pd["MARITAL"], df_pd["label"],normalize='index').plot.bar(rot=0,stacked=True,figsize=(4,3),title="MARITAL VS label")
pd.crosstab(df_pd["EDUCATION"], df_pd["label"],normalize='index').plot.bar(rot=0,stacked=True,figsize=(4,3),title="EDUCATION VS label")
pd.crosstab(df_pd["SEX"], df_pd["label"],normalize='index').plot.bar(rot=0,stacked=True,figsize=(4,3),title="SEX VS label")
pd.crosstab(df_pd["DELAY1"], df_pd["label"],normalize='index').plot.bar(rot=0,stacked=True,figsize=(4,3),title="DELAY1 VS label")
pd.crosstab(df_pd["DELAY2"], df_pd["label"],normalize='index').plot.bar(rot=0,stacked=True,figsize=(4,3),title="DELAY2 VS label")

df_pd_num = df_pd.drop(["MARITAL","SEX","EDUCATION","ID"], axis=1)
colname = df_pd_num.columns

corr = df_pd_num.corr()

f, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)

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

print("Train Data Row Number = ", train.count())
print("Test Data Row Number = ", test.count())

## MODELLING and TUNING

# Declare LR Model and Pipeline
lr = LogisticRegression(featuresCol='features', labelCol="label", maxIter=10) 
pipe_lr = Pipeline(stages=[MARITALindexer,EDUCATIONindexer,SEXindexer,MARITALencoder,EDUCATIONencoder,SEXencoder,assembler,lr])     

# Hyper-Parameter Tuning
paramGrid_lr = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.elasticNetParam, [0.8, 0.7]) \
    .build()
crossval_lr = CrossValidator(estimator=pipe_lr,
                             estimatorParamMaps=paramGrid_lr,
                             evaluator=BinaryClassificationEvaluator(),
                             numFolds=3) 

# Build model and predict Test Data
lr_model    = crossval_lr.fit(train) 
lr_result   = lr_model.transform(test)  

lr_eval = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="label")
lr_eval2= MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")
lr_AUC  = lr_eval.evaluate(lr_result)
lr_ACC  = lr_eval2.evaluate(lr_result, {lr_eval2.metricName:"accuracy"})
lr_f1   = lr_eval2.evaluate(lr_result, {lr_eval2.metricName:"f1"})

print("Logistic Regression Performance Measure")
print("Accuracy = %0.2f" % lr_ACC)
print("AUC = %.2f" % lr_AUC)

cm_lr = lr_result.crosstab("prediction", "label")
cm_lr = cm_lr.toPandas()
cm_lr

TP = cm_lr["1"][0]
FP = cm_lr["0"][0]
TN = cm_lr["0"][1]
FN = cm_lr["1"][1]
Accuracy = (TP+TN)/(TP+FP+TN+FN)
Sensitivity = TP/(TP+FN)
Specificity = TN/(TN+FP)
Precision = TP/(TP+FP)

print ("Accuracy = %0.2f" %Accuracy )
print ("Sensitivity = %0.2f" %Sensitivity )
print ("Specificity = %0.2f" %Specificity )
print ("Precision = %0.2f" %Precision )