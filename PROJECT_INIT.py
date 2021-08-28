# Databricks notebook source
import requests
import pandas as pd
from datetime import datetime
today = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
import random
import mlflow

  
api_key = ''
client_id = ''
headers = {'Authorization': 'Bearer {}'.format(api_key)}

def get_business(term, location):
  df = pd.DataFrame()
  today = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
  url = 'https://api.yelp.com/v3/businesses/search'  
  
  for offset in range(0, 1000, 50):    
    params = {'limit': 50, 'location': location.replace(' ', '+'), 'offset': offset, 'term': term.replace(' ', '+')    
              }
  
    response = requests.get(url = url, headers=headers, params=params)
    
    def join_categories(df):
      vals=[]
      for key,item in enumerate(df):
        vals.append(item['alias'])

      return '|'.join(vals)
      
    if response.status_code == 200:
      
      tmp = pd.json_normalize(response.json()['businesses'])      
      tmp['cats'] = tmp['categories'].apply(lambda x: join_categories(x))      
      tmp_reduced = tmp[['id','name','review_count','rating','cats']]
      
      df = df.append(tmp_reduced,ignore_index=True)
      
    elif response.status_code == 400:
      print('400 Bad Request')
      break       
    
  print(f"length of businesses returned: {len(df['id'])}")
  return df.sort_values(by='rating', ascending=False)
  

def filter_business(df, min_rating = 1):
  ''' df ---> list() of ids '''

  return df[df['rating'] >= min_rating][['id','name','rating','cats']]
  
  
def get_reviews(biz_df):
  review_df = pd.DataFrame()
   
  for id in biz_df['id'].values:
        
    #http request
    url = f'https://api.yelp.com/v3/businesses/{id}/reviews'    
    response = requests.get(url = url,headers=headers)     
    r = response.json()    
    try:
      resp = pd.json_normalize(r['reviews'])
      
      #appending new cols to review df
      resp['name'] = biz_df[biz_df['id'] == id]['name'].iat[0]   
      resp['rating'] = biz_df[biz_df['id'] == id]['rating'].iat[0]
      resp['cats'] = biz_df[biz_df['id'] == id]['cats'].iat[0]
      resp = resp[['id','name','text','rating','cats']]     
      review_df = review_df.append(resp, ignore_index=True)
      
    except KeyError:
      print('key error')
      continue
    
      
       
  return review_df  

def inputstream(term, location, minimum_rating):
  
  biz = get_business(term,location)
  biz_rev = get_reviews(biz)
  
  
    
  return biz_rev



# COMMAND ----------

import time

#due to API caps i incurred during the day, i will mimic calls using incremental copies instead

def start_class_demo():
  for n in range(1,939):
    time.sleep(1)     
    dbutils.fs.cp(f'dbfs:/yelp_dump/file-{n}.json',f'dbfs:/yelp_dump/class_demo/file-{n}.json.json')

def class_demo_cleanup():
  dbutils.fs.rm('dbfs:/yelp_dump/class_demo',recurse=True)
  
def stop_all_streams():
  for stream in spark.streams.active:
    print(f"Stopping {stream.name}")
    stream.stop() # Stop the stream

# COMMAND ----------


