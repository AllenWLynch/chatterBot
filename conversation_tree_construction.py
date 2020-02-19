
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

data = data.withColumn('is_english', data_utils.ENGLISH_UDF('text'))
data = data.filter(data['is_english'] == "True")

data = data.select('tweet_id','response_tweet_id', 'in_response_to_tweet_id', 'text', 'author_id')

data.persist()

#data.persist()
# %%
#initialize chains w/ context, response, sender, author_id, next
chains = data.filter((data.in_response_to_tweet_id=="none") & ~(data.response_tweet_id=="none"))\
    .select(
        functions.col('tweet_id').alias('response'),
        functions.col('response_tweet_id').alias('next'),
        'text',
        'author_id',
    )

chains = chains.withColumn('context', functions.lit("").cast(types.StringType()))
chains = chains.withColumn('sender', functions.lit("").cast(types.StringType()))
chains = chains.withColumn('rpos', functions.lit(1).cast(types.IntegerType()))
chains = chains.withColumn('tweets', functions.array().cast("array<string>"))

chains.persist()

#initialize empty samples DF w/ context, sender, response, author_id
fields = ['sender','context','response','author_id']
samples_schema = types.StructType([types.StructField(field_name, types.StringType(), True) for field_name in fields])

samples = spark.createDataFrame([], samples_schema)
#%%
MAX_DEPTH = 20
depth = 0

format_str = '{:10} | {:10}| {:10}'
print(format_str.format('Iteration', 'Chains','Samples'))


#ALGO
'''
1. Find roots, add lit column of rpos
2. explode next w/ pos -> next, npos
3. groupby context, author_id, next over rpos, agg npos with min
4. push to stack
5. join with data
'''
#%%
while chains.count() > 0 and depth < MAX_DEPTH:

#%%

    #Flatten out fans using ordered groupby
    WINDOW_ON = ['context','author_id','next']
    window = Window.partitionBy(WINDOW_ON).orderBy('rpos')

    chains = chains.select('context','sender','tweets','response','author_id','rpos', 'text', functions.posexplode(functions.split('next',","))
        .alias('npos','next'))\
        .repartition(*WINDOW_ON)\
        .select('context','sender','tweets','author_id','next', 
            functions.concat_ws(',', (functions.collect_list('response').over(window))).alias('response'),
            functions.first('npos').over(window).alias('rpos'),
            functions.array(functions.collect_list('text').over(window)).alias('text'),
        )

    chains.persist()
#%% 
    if depth == 0:
        chains = chains.select(
            functions.concat('context','response').alias('context'),
            functions.concat('sender','author_id').alias('sender'),
            'rpos',
            'next',
            functions.col('text').alias('tweets')
        )
    else:
        chains = chains.select(
            functions.concat_ws(',', 'context','response').alias('context'),
            functions.concat_ws(',','sender','author_id').alias('sender'),
            functions.array_union('tweets','text'),
            'rpos',
            'next'
        )
    
#%%
    chains = chains.join(data, chains.next == data.tweet_id, 'inner')\
        .select(
            'sender',
            'context',
            'tweets',
            'rpos',
            data.tweet_id.alias('response'),
            data.author_id,
            data.response_tweet_id.alias('next'),
            data.text.alias('text')
        )

#%%
    chains.persist()
    #Add to samples
    samples = samples.unionAll(chains.select('sender','context','response','author_id'))
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
samples.coalesce(1)\
    .write.format("com.databricks.spark.csv")\
    .option("header", "true")\
    .save("data/conversation_chains.csv")\

print('Done! Congratulations!')