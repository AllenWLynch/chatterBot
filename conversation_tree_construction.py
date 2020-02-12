
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
#data = data.withColumn('response_tweet_id', functions.explode(functions.split(data.response_tweet_id, ',')))
#%%
data = data.na.fill('none', ['response_tweet_id','in_response_to_tweet_id'])
data.createOrReplaceTempView('data')
# %%

roots = spark.sql('''SELECT tweet_id AS on_deck, inbound, EXPLODE_OUTER(SPLIT(response_tweet_id, ",")) AS next, author_id
                    FROM data
                    WHERE inbound=True AND in_response_to_tweet_id IS NULL''')

roots = roots.withColumn('conversation', functions.lit("").cast(types.StringType()))
roots = roots.withColumn('speaker', functions.lit("").cast(types.StringType()))

roots.show()
roots.select('next').filter(functions.isnull('next')).count()

#%%
terminated_chains = roots.filter(functions.isnull('next'))
growing_chains = roots.filter(roots.next.isNotNull())
#%%
##Iterative part begins

#1. flatten out node fans if there are multiple tweets by the same author
growing_chains = growing_chains.groupBy('speaker','conversation','author_id','next').agg(
        functions.concat_ws(',', functions.collect_list('on_deck')).alias('on_deck'),
        functions.concat_ws(',', functions.collect_list('inbound')).alias('inbound'),
)

growing_chains.show()
#%%
#2. Push the on-deck tweet info onto the conversation and speaker stacks
growing_chains = growing_chains.select(
    functions.concat_ws(',', 'speaker','inbound').alias('speaker'),
    functions.concat_ws(',','conversation','on_deck').alias('conversation'),
    'next'    
    )

#%%
#3. Append new tweets on the next key from data
growing_chains = growing_chains.join(data, growing_chains.next == data.tweet_id, 'left')\
.select(
    growing_chains.speaker,
    growing_chains.conversation,
    data.tweet_id.alias('on_deck'),
    data.author_id,
    functions.explode_outer(functions.split(data.response_tweet_id, ",")).alias('next'),
    data.inbound,
)

#%%
terminated_chains = terminated_chains.unionAll(growing_chains.filter(functions.isnull('next')))
growing_chains = growing_chains.filter(growing_chains.next.isNotNull())

print('Num growing chains:', growing_chains.count())
print('Terminated chains:', terminated_chains.count())

growing_chains.show()

# %%
