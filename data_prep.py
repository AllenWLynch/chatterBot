
#%%
import findspark
findspark.init()

import pyspark

spark = pyspark.sql.SparkSession.builder.appName("Chatbot prep").getOrCreate()

#%%
from pyspark.sql import functions
from pyspark.sql import types

#%%

data = spark.read.csv('./data/twcs/twcs.csv', inferSchema = True, header = True)
data.show()
data = data.withColumn('response_tweet_id', functions.explode(functions.split(data.response_tweet_id, ',')))

#%%
data.createOrReplaceTempView('data')
# %%

roots = spark.sql('''SELECT tweet_id, inbound, response_tweet_id
                    FROM data
                    WHERE inbound=True AND in_response_to_tweet_id IS NULL''')
roots.show()
# %%
roots.createOrReplaceTempView('roots')

# %%
first_join = spark.sql('''
SELECT data.author_id,
    CONCAT(roots.inbound,",",data.inbound) AS inbound,
    CONCAT(roots.tweet_id, ",", data.tweet_id) AS conversation, 
    data.response_tweet_id AS strand
FROM roots
INNER JOIN data ON roots.response_tweet_id=data.tweet_id''')

first_join.show()
# %%
first_join.createOrReplaceTempView('conversations')

# %%

## Need to add some monitor for finishing the joins
iterative_join = spark.sql('''
SELECT conversations.author_id, 
    CONCAT(conversations.inbound, ",", COALESCE(data.inbound,"")) AS inbound,
    CONCAT(conversations.conversation,",",COALESCE(data.tweet_id,"")) AS conversation, 
    data.response_tweet_id AS strand
FROM conversations
LEFT JOIN data ON conversations.strand=data.tweet_id
''')

iterative_join.createOrReplaceTempView('conversations')
iterative_join.show()
#%%