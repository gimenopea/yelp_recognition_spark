# Databricks notebook source
# MAGIC %run "./PROJECT_INIT"

# COMMAND ----------

stop_all_streams()

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Inbound Streaming #1: Load incoming JSON feed to a delta source 'repository' table

# COMMAND ----------

#setting schema for incoming json stream

schema = spark.read.json('dbfs:/yelp_dump/class_demo',multiLine=True).schema
schema.fields

# COMMAND ----------

#deltapath = '/tmp/final_proj_delta/'
#dbutils.fs.rm(deltapath,recurse=True)

# COMMAND ----------

#set up readstream 
job_1_ingest_json = spark.readStream.schema(schema).json('dbfs:/yelp_dump/class_demo')

deltapath = '/tmp/final_proj_delta/'

#drop dupes
job_1_ingest_json.dropDuplicates(subset=['id','name'])
job_1_ingest_json = job_1_ingest_json.drop('_corrupt_record')
job_1_ingest_json = job_1_ingest_json.filter(job_1_ingest_json.text.isNotNull())
job_1_ingest_json.createOrReplaceTempView('raw_streaming_data')

#start writestream
job_1_ingest_json.writeStream.format("delta").outputMode("append").queryName("yelp_delta_ingest").trigger(processingTime="5 seconds").option("checkpointLocation",f'{deltapath}checkpoint').start(deltapath)

# COMMAND ----------

#number of records in the delta table
spark.sql(f"select COUNT(*) from delta.`{deltapath}` ").show()


# COMMAND ----------

display(job_1_ingest_json)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Inbound Streaming 2: Delta Table to Vectorized Delta Table for Machine Learning Algorithms

# COMMAND ----------

#initialize our second streaming reader

job_2_delta_ml = spark.readStream.format("delta").option("ignoreChanges", "true").load(deltapath)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Load Pipeline used in 2 other notebooks

# COMMAND ----------

from pyspark.ml import PipelineModel
from pyspark.ml.feature import IDF,StringIndexer,StopWordsRemover,CountVectorizer,Tokenizer, VectorAssembler

pipeline_model = PipelineModel.load('/final_project/pipeline/')
pipeline_model.stages

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Load our best pipeline model and transform incoming stream

# COMMAND ----------

best_model_uri = 'dbfs:/databricks/mlflow-tracking/1736550529104970/3cee8be31ded483ab8e28194788f994c/artifacts/final_project_models/model-Logistic Regression'

#load model

loaded_model = mlflow.spark.load_model(best_model_uri)

#transform pipeline
stream_vectorized = pipeline_model.transform(job_2_delta_ml)

#transform model
stream_predictions = loaded_model.transform(stream_vectorized)


# COMMAND ----------

stream_predictions.isStreaming

# COMMAND ----------

#dbutils.fs.rm('/final_projet/scored_delta',recurse=True)

# COMMAND ----------

#start writestream
delta_ml_path = '/final_projet/scored_delta'
checkpoint_ml_path = f'{delta_ml_path}/checkpoint/'
stream_predictions.writeStream.format("delta").outputMode("append").queryName("predictions_delta").trigger(processingTime="5 seconds").option("checkpointLocation",delta_ml_path).start(delta_ml_path)

# COMMAND ----------

display(stream_predictions.select('text','prediction').filter(stream_predictions.prediction == 1))

# COMMAND ----------


