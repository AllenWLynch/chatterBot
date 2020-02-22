
#%%
import findspark
findspark.init()

import pyspark

spark = pyspark.sql.SparkSession.builder.appName("Chatbot prep").getOrCreate()

#%%
from pyspark.sql import functions
from pyspark.sql import types
from data_utils import apply_filters
import os

#%%

data = spark.read.csv('./data/twcs/twcs.csv', inferSchema = True, header = True)
data.show()
#data = data.withColumn('response_tweet_id', functions.explode(functions.split(data.response_tweet_id, ',')))

data = data.na.fill('none', ['response_tweet_id','in_response_to_tweet_id'])
data.createOrReplaceTempView('data')

#%%

characters = data.select(functions.explode(functions.split('text', '')).alias('character'))\
    .groupBy('character').count().orderBy(characters['count'].desc())
    
characters = characters.collect()
#%%

authors = data.filter(data.inbound == 'False').select('author_id').distinct().filter(functions.col('author_id').rlike("^[^0-9]*$"))

authors_df = authors.collect()

# %%

filter_udf = functions.udf(apply_filters, types.StringType())
#%%

# %%

def is_english(text, threshold = 0.4):

    if not text:
        return 'False'

    return str(sum([(char.isalpha() and ord(char) <= 127) for char in text])/len(text) > threshold)

negative_test=  'おいいいいい！発売する三ヶ月前から'
positive_test = 'Hello my name is bert'

is_english_udf = functions.udf(is_english, types.StringType())

# %%

data = data.withColumn('is_english', is_english_udf('text'))
data.show()

#%%
data = data.filter(data['is_english'] == "True")

# %%
data = data.withColumn('regex_filtered', filter_udf('text'))

data.show()

# %%

#data.select('regex_filtered').coalesce(5).write.format("com.databricks.spark.csv").option("header","false").save("data/text_corpus.csv")

# %%
