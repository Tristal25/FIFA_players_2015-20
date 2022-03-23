import findspark
findspark.init(spark_home = "/Users/tristalli/spark-3.2.0-bin-hadoop3.2",
               python_path = "/opt/anaconda3/bin/python3")
import pyspark
from pyspark.sql import SparkSession, SQLContext, DataFrame
from pyspark.ml.feature import Imputer
from pyspark.sql.functions import *
from functools import reduce

import os
os.chdir("/Users/tristalli/Desktop/36652/course-project-Tristal25")
secret = open('secret.txt', 'r')
pw = secret.read()
secret.close()

appName = "Course-project-36652"
master = "local"

conf = pyspark.SparkConf()\
    .set('spark.driver.host','127.0.0.1')\
    .setAppName(appName)\
    .setMaster(master)

sc = SparkContext.getOrCreate(conf=conf)

sqlContext = SQLContext(sc)

spark = sqlContext.sparkSession.builder.getOrCreate()

#Ingest data from the players CSV into Spark Dataframe. What is dataframe?
players15_raw_df = (spark.read
         .format("csv")
         .option("inferSchema", "true")
         .option("header","true")
         .load("data/players_15.csv")
      )

players16_raw_df = (spark.read
         .format("csv")
         .option("inferSchema", "true")
         .option("header","true")
         .load("data/players_16.csv")
      )

players17_raw_df = (spark.read
         .format("csv")
         .option("inferSchema", "true")
         .option("header","true")
         .load("data/players_17.csv")
      )

players18_raw_df = (spark.read
         .format("csv")
         .option("inferSchema", "true")
         .option("header","true")
         .load("data/players_18.csv")
      )

players19_raw_df = (spark.read
         .format("csv")
         .option("inferSchema", "true")
         .option("header","true")
         .load("data/players_19.csv")
      )

players20_raw_df = (spark.read
         .format("csv")
         .option("inferSchema", "true")
         .option("header","true")
         .load("data/players_20.csv")
      )

players15_raw_df.printSchema()
players16_raw_df.printSchema()
players17_raw_df.printSchema()
players18_raw_df.printSchema()
players19_raw_df.printSchema()
players20_raw_df.printSchema()

players15_year = players15_raw_df.withColumn("year", lit(2015))
players16_year = players15_raw_df.withColumn("year", lit(2016))
players17_year = players15_raw_df.withColumn("year", lit(2017))
players18_year = players15_raw_df.withColumn("year", lit(2018))
players19_year = players15_raw_df.withColumn("year", lit(2019))
players20_year = players15_raw_df.withColumn("year", lit(2020))

players15_year.columns == players16_year.columns == players17_year.columns == players18_year.columns == players19_year.columns == players20_year.columns

players_unioned = reduce(DataFrame.unionByName, [players15_year,players16_year,players17_year,players18_year,players19_year,players20_year])

players_unioned.printSchema()

# Missing values

nrow = players_unioned.count()

## Check for missing values
players_unioned.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in players_unioned.columns]).show()

## Delete columns with > 50% missing values
players_unioned.select([(count(when(isnan(c) | col(c).isNull(), c))/nrow > 0.5).alias(c) for c in players_unioned.columns]).show()

players_dropped_na50 = players_unioned.drop("release_clause_eur", "player_tags", "loaned_from", "nation_position", "nation_jersey_number", \
                                            "gk_diving", "gk_handling", "gk_kicking", "gk_reflexes", "gk_speed", "gk_positioning", \
                                            "player_traits", "mentality_composure")

players_dropped_na50.select([(count(when(isnan(c) | col(c).isNull(), c))/nrow > 0.5).alias(c) for c in players_dropped_na50.columns]).show()

## clean + and -

col_pm = ['attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys',
          'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 'movement_acceleration',
          'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping',
          'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression', 'mentality_interceptions', 'mentality_positioning',
          'mentality_vision', 'mentality_penalties', 'defending_marking', 'defending_standing_tackle', 'defending_sliding_tackle',
          'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes',
          'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm', 'cdm',
          'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb']


players_pm = players_dropped_na50

for c in col_pm:
    players_pm = players_pm.withColumn("{}_int".format(c), regexp_extract(col(c), "^(\d+)", 1).cast("integer")) \
        .withColumn("{}_plus".format(c), regexp_extract(col(c), "^(\d+)\+(\d)", 2).cast("integer")) \
        .withColumn("{}_plus".format(c), when(col("{}_plus".format(c)).isNull(), 0).otherwise(col("{}_plus".format(c))).cast("integer")) \
        .withColumn("{}_minus".format(c), regexp_extract(col(c), "^(\d+)\-(\d)", 2).cast("integer")) \
        .withColumn("{}_minus".format(c), when(col("{}_minus".format(c)).isNull(), 0).otherwise(col("{}_minus".format(c))).cast("integer")) \
        .withColumn("{}_total".format(c), col("{}_int".format(c)) + col("{}_plus".format(c)) - col("{}_minus".format(c))) \
        .drop(c, "{}_int".format(c), "{}_plus".format(c), "{}_minus".format(c)) \
        .withColumnRenamed("{}_total".format(c), c)\
        .withColumn(c, when((0 <= col(c)) & (col(c) <= 100), col(c)).otherwise(lit(None)))

players_pm.select(col_pm).show()

players_pm.select(['goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes']).show()

## Impute the data

players_pm.select([count(when((isnan(c) | col(c).isNull()), c)).alias(c) for c in players_dropped_na50.columns]).show()

[c for c in players_dropped_na50.columns if dict(players_dropped_na50.dtypes)[c] == "int"]

num_null_col = ['team_jersey_number', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic'] + col_pm

imputer_players = Imputer(
    inputCols = num_null_col,
    outputCols = ["{}_imputed".format(c) for c in num_null_col]
    ).setStrategy("mean")

players_imputed = imputer_players.fit(players_pm).transform(players_pm)

for c in num_null_col:
    players_imputed = players_imputed.drop(c).withColumnRenamed("{}_imputed".format(c), c)

players_imputed.select([count(when((isnan(c) | col(c).isNull()), c)).alias(c) for c in players_imputed.columns]).show()

### Three cols with missing values left: team_position: 1392, join: 6906, contract_valid_until: 1434

# Cast types

players_ctype = players_imputed.withColumn("dob", to_date(col("dob"), "yyyy-mm-dd"))\
    .withColumn("joined", to_date(col("joined"), "yyyy-mm-dd"))

# Now the data is ready for import

db_properties={}
db_properties['username']="postgres"
db_properties['password']=pw

db_properties['url']= "jdbc:postgresql://localhost:5432/postgres"
db_properties['driver']="org.postgresql.Driver"

players_ctype.write.format("jdbc")\
.mode("overwrite")\
.option("url", "jdbc:postgresql://localhost:5432/postgres")\
.option("dbtable", "FIFA.players")\
.option("user", "postgres")\
.option("password", pw)\
.option("Driver", "org.postgresql.Driver")\
.save()











