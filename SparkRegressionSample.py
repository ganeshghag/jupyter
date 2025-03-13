# Databricks notebook source
pip install xgboost

# COMMAND ----------

#%restart_python
from pyspark.sql import SparkSession
#from sparkxgb import SparkXGBRegressor  # Import XGBoost for PySpark
from xgboost.spark import SparkXGBRegressor as sparkxgb
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

# 1️⃣ Initialize Spark Session
#spark = SparkSession.builder.appName(PySparkXGBoostRegression).getOrCreate()

# 2️⃣ Load Dataset (California Housing dataset)
#data = spark.read.csv("https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv",
#                      header=True, inferSchema=True)

from pyspark.sql import SparkSession
#from sparkxgb import SparkXGBRegressor  # Import XGBoost for PySpark
from xgboost.spark import SparkXGBRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler

data = spark.read.format("csv").option("header", True).option("inferSchema", True).load("dbfs:/FileStore/tables/housing.csv")
#data = spark.read.format("csv").option("header", "true").load("https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv")
# 3️⃣ Check Schema
data.printSchema()

# 4️⃣ Handle Missing Values (Drop Nulls)
data = data.dropna()

# 5️⃣ Feature Engineering - Assemble Features into a Single Column
feature_columns = data.columns[:-1]  # Exclude target column (median_house_value)
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data).select("features", "median_house_value")

# 6️⃣ Train-Test Split (80-20)
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# 7️⃣ Initialize & Train XGBoost Regressor
xgb = SparkXGBRegressor(features_col="features", label_col="median_house_value",
                        maxDepth=6, eta=0.1, objective="reg:squarederror")
model = xgb.fit(train_data)

# 8️⃣ Model Evaluation on Test Data
predictions = model.transform(test_data)

evaluator = RegressionEvaluator(labelCol="median_house_value", metricName="rmse")
rmse = evaluator.evaluate(predictions)

# 9️⃣ Print Model Performance
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# COMMAND ----------

data = spark.read.format("csv").option("header", True).option("inferSchema", True).load("dbfs:/FileStore/tables/housing.csv")

data.describe(["median_house_value"]).show()


# COMMAND ----------


