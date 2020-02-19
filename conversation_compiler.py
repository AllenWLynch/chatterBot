#%%
import findspark
findspark.init()

import pyspark

spark = pyspark.sql.SparkSession.builder.appName("Chatbot prep")\
    .config("spark.executor.memory","10g")\
    .config("spark.driver.memory","10g")\
    .getOrCreate()

#%%
from pyspark.sql import functions
from pyspark.sql import types
from pyspark.sql import Window
from collections import defaultdict
import data_utils
#%%

data = spark.read.csv('./data/twcs/twcs.csv', inferSchema = True, header = True)
data = data.na.fill('none', ['response_tweet_id','in_response_to_tweet_id'])

#%%

SAVE_PATH = './data/spark_checkpoints'
spark.sparkContext.setCheckpointDir(SAVE_PATH)

#load in companies to encode speakers

data = data.withColumn('is_english', data_utils.ENGLISH_UDF('text'))
data = data.filter(data['is_english'] == "True")

data = data.select('tweet_id','response_tweet_id','author_id', 'in_response_to_tweet_id', 'text')

data.persist()
#data.persist()
# %%
#initialize chains w/ context, response, sender, author_id, next
chains = data.filter((data.in_response_to_tweet_id=="none") & ~(data.response_tweet_id=="none"))\
    .select(
        functions.array('author_id').alias('sender'),
        functions.array('text').alias('tweets'),
        functions.col('response_tweet_id').alias('next')
    )

chains.persist()
#%%
#initialize empty samples DF w/ context, sender, response, author_id
#fields = ['sender','tweets','response','author_id']
samples_schema = types.StructType([
    types.StructField('sender', types.ArrayType(types.IntegerType()), True),
    types.StructField('tweets', types.ArrayType(types.StringType()), True),
    types.StructField('response', types.StringType(), True),
    types.StructField('author_id', types.StringType(), True),])

samples = spark.createDataFrame([], samples_schema)
     

#%%
MAX_DEPTH = 20
depth = 0

format_str = '{:10} | {:10}| {:10}'
print(format_str.format('Iteration', 'Chains','Samples'))

#%%

def tuck(context):
    return context[:-2] + [' '.join(context[-2 : ])]

tuck_udf = functions.udf(tuck, types.ArrayType(types.StringType()))

#%%
while chains.count() > 0 and depth < MAX_DEPTH:

#%%
    if depth > 0:

        chains = chains.select(
            functions.when(chains['self_response'], 
                    tuck_udf(functions.concat('tweets', functions.array('response')))
                ).otherwise(
                    functions.concat('tweets',functions.array('response')),
                ).alias('tweets'),
            functions.when(chains['self_response'],
                    functions.col('sender')
                ).otherwise(
                    functions.concat('sender', functions.array('author_id'))
                ).alias('sender'),
            'next',
        )

#%%
    chains = chains.withColumn('next', functions.explode(functions.split('next', ",")))

    chains.persist()
    
#%%
    chains = chains.join(data, chains.next == data.tweet_id, 'inner')\
        .select(
            'sender',
            'tweets',
            data.text.alias('response'),
            data.author_id,
            data.response_tweet_id.alias('next'),
        )
#%%
    chains = chains.withColumn('self_response', functions.element_at('sender', -1) == functions.col('author_id'))
#%%
    chains.persist()
    #Add to samples
    samples = samples.unionAll(chains.filter(~chains['self_response']).select('sender','tweets','response','author_id'))
    # Remove finished chains
    samples = samples.checkpoint()
    
#%%
    chains = chains.filter(chains.next != 'none')

    chains.persist()

    print(format_str.format(str(depth+1), str(chains.count()), str(samples.count())))
    chains = chains.checkpoint()
    #checkpoint here
    depth += 1


#%%

samples = samples.withColumn('id', functions.monotonically_increasing_id())
#save samples
print('Saving!')
samples.coalesce(5)\
    .write.json("data/mined_conversations")

print('Done! Congratulations!')