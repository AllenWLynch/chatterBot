
#%%
import findspark
findspark.init()

import pyspark

spark = pyspark.sql.SparkSession.builder.appName("Chatbot prep").getOrCreate()

#%%
from pyspark.sql import functions
from pyspark.sql import types
from pyspark.sql import Window
from collections import defaultdict
import data_utils
import sentencepiece as spm
import importlib
#%%
data_utils = importlib.reload(data_utils)
#%%
SAVE_PATH = './data/spark_checkpoints'
spark.sparkContext.setCheckpointDir(SAVE_PATH)
#%%

#1. Load in data
data = spark.read.csv('./data/twcs/twcs.csv', inferSchema = True, header = True)
data = data.na.fill('none', ['response_tweet_id','in_response_to_tweet_id'])

data.count()
#%%
#2. Encode authors
data = data.withColumn('author_name', functions.col('author_id'))
data = data.withColumn('author_id', data_utils.AUTHORS_UDF('author_id'))

#3. filter for english tweets
data = data.withColumn('is_english', data_utils.ENGLISH_UDF('text'))
data = data.filter(data['is_english'] == "True")

#4. Substitute regex tokens
#data = data.withColumn('regex_filtered', data_utils.FILTER_UDF('text'))

#5. Filter out short responses
data = data.withColumn('wordcount', functions.size(functions.split('text', " ")))

data.persist()
data.count()
# %%
#6 load in conversation chains

samples = spark.read.csv('./data/conversation_chains.csv', inferSchema = True, header = True)

context = samples.select('id','sender','context', functions.size(functions.split('sender', ",")).alias('length')).orderBy(functions.desc("length"))

response = samples.select('id','response', 'author_id')
#%%
#add regex_filtered response to response side of samples
response = response.join(data.select('tweet_id','text','wordcount'), response.response == data.tweet_id, 'left')
response = response.filter(response.wordcount > 2)
response.show()

#%%
#pivot context and senders
context = context.select('id', functions.posexplode(functions.arrays_zip(
        functions.split('sender',","),
        functions.split('context',","))).alias('pos','zipped'))\
        .select('id','pos', functions.col('zipped').getItem("0").alias('sender'), functions.col('zipped').getItem("1").alias('tweet_id'))
context.show()

#%%
context = context.join(data.select('tweet_id','regex_filtered'), 'tweet_id', 'left')
context.persist()

data.unpersist()
#%%

window = Window.partitionBy('id').orderBy('pos')
context = context.repartition('id').select(
            'id',
            functions.collect_list('sender').over(window).alias('sender')
            functions.collect_list('context').over(window).alias('context')
            )

context.persist()
context.show()
#%%

reconstructed = context.join(response, 'id', 'inner')
reconstructed.show()

#%%
reconstructed.coalesce(1).write.json('data/samples')