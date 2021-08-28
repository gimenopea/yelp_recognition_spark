



# MAGIC %run "./PROJECT_INIT"



# MAGIC %md 
# MAGIC ### Figure out the best model in the experiment



#URI functions to reference the best model

import mlflow
import ast

EXPERIMENT_ID = '1736550529104970' #or put this in an env_var?


def query_best_model_uri(minimum_accuracy):
  
  query_best_model = mlflow.search_runs(experiment_ids = EXPERIMENT_ID, filter_string=f'metrics.accuracy > {minimum_accuracy}').sort_values('metrics.accuracy', ascending=False)
  best = query_best_model[['artifact_uri','run_id','metrics.accuracy','params.type','tags.mlflow.log-model.history']]
  return best

def get_best_model_uri(query_best_model,x):
  return query_best_model['artifact_uri'].iloc[x] +'/'+ast.literal_eval(filtered_models['tags.mlflow.log-model.history'].iloc[x])[0]['artifact_path']
  



#dbutils.fs.rm('/yelp_dump/',recurse=True)



from datetime import datetime
today = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')



'''
#writes json in a dir for class demo

t = sample.set_index('name',drop=False)
businesses = t.index.unique().values
import time

for i,n in enumerate(businesses):
  
  tmp_df = pd.DataFrame()
  tmp_df = t.loc[n]
  json_df = tmp_df.to_json(orient='records',lines=True)    
  dbutils.fs.put(f"/yelp_dump/file-{i}.json",json_df,overwrite=True);

'''

#

# MAGIC %md
# MAGIC 
# MAGIC ### Query all my runs with accuracy score above 60%



filtered_models = query_best_model_uri(0.6)
filtered_models.reset_index().drop('index',axis=1)




# MAGIC %md
# MAGIC 
# MAGIC ### Get the URI of the best model i choose

# COMMAND ----------

best_model_uri = get_best_model_uri(filtered_models,1)
best_model_uri

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Load the best model in a variable for the new data stream

# COMMAND ----------

loaded_model = mlflow.spark.load_model(best_model_uri)

# COMMAND ----------

# MAGIC %md Load set for batch screening

# COMMAND ----------

#load 1000 comment samples from another area, will take at least 3 minutes
#sample = inputstream('restaurant','denver, CO', 1)
#sample.info()

# COMMAND ----------

#dfa = spark.read.parquet('dbfs:/final_project/training_dataset/yelp_train.parquet/')
dfa = spark.createDataFrame(sample)
#dfa = spark.read.json('dbfs:/yelp_dump/file2020_11_29_00_33_28-931.json')


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Load the pipeline transformers saved in the training notebook

# COMMAND ----------

from pyspark.ml import PipelineModel

pipeline_model = PipelineModel.load('/final_project/pipeline/')
pipeline_model.stages

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### transform the batch set and run predictions

# COMMAND ----------

vectorized_df = pipeline_model.transform(dfa)
result = loaded_model.transform(vectorized_df)

# COMMAND ----------

sample[sample['cats'].str.contains('pizza')]

# COMMAND ----------

from pyspark.sql.functions import lower

matches = result.select(lower('text').alias('text'),'rating','prediction').filter(result.prediction == 1)
display(matches)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 
# MAGIC # Start Server Here

# COMMAND ----------

start_class_demo()

# COMMAND ----------

class_demo_cleanup()

# COMMAND ----------


