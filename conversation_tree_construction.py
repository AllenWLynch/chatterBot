
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

#%%

data = spark.read.csv('./data/twcs/twcs.csv', inferSchema = True, header = True)
data = data.na.fill('none', ['response_tweet_id','in_response_to_tweet_id'])

#%%

SAVE_PATH = './data/conversation_trees'
spark.sparkContext.setCheckpointDir(SAVE_PATH)

#load in companies to encode speakers
with open('./data/bot_names.csv', 'r') as f:
    authors = [line.strip() for line in f.readlines()]

authors = { author : i+1 for (i, author) in enumerate(authors)}

#%%

def encode_authors(author):
    return str(authors.get(author, int(0)))

encode_authors_udf = functions.udf(encode_authors, types.StringType())

data = data.withColumn('author_name', functions.col('author_id'))
data = data.withColumn('author_id', encode_authors_udf('author_id'))

def is_english(text, threshold = 0.4):
    if not text:
        return 'False'
    return str(sum([(char.isalpha() and ord(char) <= 127) for char in text])/len(text) > threshold)

is_english_udf = functions.udf(is_english, types.StringType())
data = data.withColumn('is_english', is_english_udf('text'))
data = data.filter(data['is_english'] == "True")

data = data.select('tweet_id','response_tweet_id','author_id', 'in_response_to_tweet_id')

data.persist()
#%%
data.createOrReplaceTempView('data')

#data.persist()
# %%
#initialize chains w/ context, response, sender, author_id, next
roots = spark.sql('''
    SELECT tweet_id AS response, author_id, response_tweet_id AS next
    FROM data
    WHERE in_response_to_tweet_id='none' AND NOT response_tweet_id='none'
''')

#%%
chains = roots

chains = chains.withColumn('context', functions.lit("").cast(types.StringType()))
chains = chains.withColumn('sender', functions.lit("").cast(types.StringType()))

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


#NEW ALGO
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
    '''chains = chains.select(
        'sender','context','author_id', 'response',
        functions.explode(functions.split('next',",")).alias('next')
        ).groupBy('sender','context','author_id','next').agg(
            functions.concat_ws(',', functions.collect_list('response')).alias('response'),
    )'''

    #Flatten out fans using ordered groupby
    WINDOW_ON = ['context','author_id','next']
    window = Window.partitionBy(WINDOW_ON).orderBy('pos')

    chains = chains.select('context','sender','response','author_id', functions.posexplode(functions.split('next',","))
        .alias('pos','next'))\
        .repartition(*WINDOW_ON)\
        .select('context','sender','author_id','next', 
            functions.concat_ws(',', (functions.collect_list('response').over(window))).alias('response')
        )

    if depth == 0:
        chains = chains.select(
            functions.concat('context','response').alias('context'),
            functions.concat('sender','author_id').alias('sender'),
            'next'
        )
    else:
        chains = chains.select(
            functions.concat_ws(',', 'context','response').alias('context'),
            functions.concat_ws(',','sender','author_id').alias('sender'),
            'next'
        )
    
    chains = chains.join(data, chains.next == data.tweet_id, 'inner')\
        .select(
            'sender',
            'context',
            data.tweet_id.alias('response'),
            data.author_id,
            data.response_tweet_id.alias('next'),
        )

    chains.persist()
    #Add to samples
    samples = samples.unionAll(chains.filter(chains.next == 'none').select('sender','context','response','author_id'))
    # Remove finished chains
    samples = samples.checkpoint()
    
    chains = chains.filter(chains.next != 'none')

    chains.persist()

    print(format_str.format(str(depth+1), str(chains.count()), str(samples.count())))
    print(chains.take(1))

    chains = chains.checkpoint()
    #checkpoint here
    depth += 1


#%%

#save samples
print('Saving!')
samples.coalesce(1)\
    .write.format("com.databricks.spark.csv")\
    .option("header", "true")\
    .save("data/conversation_chains.csv")\

print('Done! Congratulations!')