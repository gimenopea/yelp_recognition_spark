# Databricks notebook source
# MAGIC %md 
# MAGIC 
# MAGIC ### Run init notebook

# COMMAND ----------

# MAGIC %run "./PROJECT_INIT"

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Create a training dataset for target reviews
# MAGIC 
# MAGIC Goal: obtain dataset of restaurant reviews is of target category, in this case we are choosing, mexican

# COMMAND ----------

#call yelp api to return 1000 businesses and review

training_dataset = inputstream('pizza','Washington, DC',1)
training_dataset.info()

# COMMAND ----------

training_dataset[training_dataset['cats'].str.contains('pizza')]

# COMMAND ----------

# add a label column for restaurants with a rating of 3 and below
# save transformed df as spark df 
# write to file in parquet format

path = 'final_project/training_dataset/'

def save_training_df(df, filename):
  'pd.DataFrame() -> spark.DataFrame()'
  
  #create label column if rating is below a 3
  df['label'] = df['cats'].apply(lambda x: 1 if 'pizza' in x else 0)
  
  print(f'Converting dataframe to spark dataframe...')
  train = spark.createDataFrame(df)
  print(f'Complete')
  print(f'Writing to parquet file in overwrite mode to {path}')
  train.write.parquet(f'{path}{filename}.parquet',mode='overwrite')
  display(train)
  return train

save_training_df(training_dataset,'yelp_train')

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Load and train the dataset

# COMMAND ----------

df = spark.read.parquet('dbfs:/final_project/training_dataset/yelp_train.parquet/')
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Setting up the transformers in a pipeline and save for later re-use

# COMMAND ----------

from pyspark.ml.feature import IDF,StringIndexer,StopWordsRemover,CountVectorizer,Tokenizer, VectorAssembler
from pyspark.ml import Pipeline

#train test split before applying vectorizers transformers

training, test = clean_data.randomSplit([0.7,0.3])
  
#text preprocessing steps



tokenizer = Tokenizer(inputCol='text',outputCol='token_text')
stop_remove = StopWordsRemover(inputCol='token_text', outputCol='stop_token')
count_vec = CountVectorizer(inputCol='stop_token', outputCol='c_vec')
idf = IDF(inputCol='c_vec',outputCol='tf_idf')
vector_assembler = VectorAssembler(inputCols=['tf_idf'],outputCol='features')

#instantiate the pipeline
pipeline = Pipeline(stages=[tokenizer,stop_remove,count_vec,idf,vector_assembler])

pipeline_model = pipeline.fit(df)

#save this pipeline to the project
pipeline_model.write().overwrite().save('final_project/pipeline/')
clean_data = pipeline_model.transform(df)


display(clean_data)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Model Selection

# COMMAND ----------

from sklearn.metrics import confusion_matrix
import mlflow
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier, LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

model_selections = [{'model': LogisticRegression(), 'name': 'Logistic Regression', 'experiment_id': '1736550529104970'},
               {'model': NaiveBayes(), 'name': 'Naive Bayes','experiment_id': '1736550529104970'},
               {'model': RandomForestClassifier(), 'name': 'Random Forest Classifier','experiment_id': '1736550529104970'}]

#path to save final project models

for model_select in model_selections:
  modelpath = f"final_project_models/model-{model_select['name']}"

  with mlflow.start_run(run_name=model_select['name'], experiment_id=model_select.get('experiment_id',None)):

    print(f'running training set on.. {model_select["name"]} ...')
    model = model_select['model']

    
  
    #transform, fit and extract performance
    mlflow.log_param('type', model)
    top_notch_model = model.fit(training)
    results = top_notch_model.transform(test)
    model_performance = results.toPandas()
    model_performance['actual'] = model_performance['cats'].apply(lambda x: 1 if 'pizza' in x else 0)
    model_performance = model_performance[['actual','prediction']]
    
    #save the model in this iteration
    mlflow.spark.save_model(top_notch_model,modelpath)
    mlflow.spark.log_model(top_notch_model,modelpath)
    
   
    #log performance
    tn, fp, fn, tp = confusion_matrix(model_performance['actual'], model_performance['prediction']).ravel()
   
    accuracy = round((tn+tp)/(tn+fp+fn+tp),2)
    mlflow.log_metric(key = 'accuracy', value = accuracy)
    
    error_rate = round((fn+fp)/(tn+fp+fn+tp),2)
    mlflow.log_metric(key = 'error_rate', value = error_rate)
    
    f1_score = tp / (tp +.5*(fp+fn))
    mlflow.log_metric(key ='f1_score',value=f1_score)
       
    
    print(f'true negative: {tn} | true positive {tp} | false negative {fn} | false positive {fp} | accuracy {accuracy}% | error rate: {error_rate}% ')
    print('Experiment run complete')

# COMMAND ----------

model_performance = results.toPandas()

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Training Data EDA

# COMMAND ----------

dfp = df.toPandas()

# COMMAND ----------

dfp.info()

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

fig,ax = plt.subplots(figsize=(12,8))
plt.hist(dfp['label'])

# COMMAND ----------


