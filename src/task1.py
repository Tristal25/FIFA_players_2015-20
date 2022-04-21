import findspark
import pyspark
from pyspark.sql import SparkSession, SQLContext, DataFrame
from pyspark.ml.feature import Imputer
from pyspark.sql.functions import *
from functools import reduce
import psycopg2
import os

findspark.init(spark_home = "/Users/tristalli/spark-3.2.0-bin-hadoop3.2",
               python_path = "/opt/anaconda3/bin/python3")

os.chdir("/Users/tristalli/Desktop/36652/course-project-Tristal25")
secret = open('secret.txt', 'r')
pw = secret.read()
secret.close()

def init_spark():
    '''
    Initiate spark session
    :return: spark and sqlContext
    '''
    appName = "Course-project-36652"
    master = "local"

    conf = pyspark.SparkConf()\
        .set('spark.driver.host','127.0.0.1')\
        .setAppName(appName)\
        .setMaster(master)

    sc = SparkContext.getOrCreate(conf=conf)

    sqlContext = SQLContext(sc)

    spark = sqlContext.sparkSession.builder.getOrCreate()
    return spark, sqlContext

def read_data(spark):
    '''
    Read FIFA data from 2015 to 2020 and union them into one dataset
    :param spark: spark session produced by init_spark()
    :return: unioned raw data in spark
    '''

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

    players15_year = players15_raw_df.withColumn("year", lit(2015))
    players16_year = players16_raw_df.withColumn("year", lit(2016))
    players17_year = players17_raw_df.withColumn("year", lit(2017))
    players18_year = players18_raw_df.withColumn("year", lit(2018))
    players19_year = players19_raw_df.withColumn("year", lit(2019))
    players20_year = players20_raw_df.withColumn("year", lit(2020))

    # Union data
    #players15_year.columns == players16_year.columns == players17_year.columns == players18_year.columns == players19_year.columns == players20_year.columns

    players_unioned = reduce(DataFrame.unionByName, [players15_year,players16_year,players17_year,players18_year,players19_year,players20_year])
    # players_unioned.printSchema()
    return players_unioned




# Missing values

def clean_data(players_unioned):
    '''
    Drop the columns with >50% NA values, deal with +/- in some columns
    :param players_unioned: Unioned data 2015-2020
    :return: cleaned data in spark
    '''

    #nrow = players_unioned.count()

    ## Check for missing values
    #players_unioned.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in players_unioned.columns]).show()

    ## Delete columns with > 50% missing values
    #players_unioned.select([(count(when(isnan(c) | col(c).isNull(), c))/nrow > 0.5).alias(c) for c in players_unioned.columns]).show()

    players_dropped_na50 = players_unioned.drop("release_clause_eur", "player_tags", "loaned_from", "nation_jersey_number", \
                                                "gk_diving", "gk_handling", "gk_kicking", "gk_reflexes", "gk_speed", "gk_positioning", \
                                                "player_traits")

    #players_dropped_na50.select([(count(when(isnan(c) | col(c).isNull(), c))/nrow > 0.5).alias(c) for c in players_dropped_na50.columns]).show()
    ### nation_position is kept for task 2

    ## clean + and -

    col_pm = ['attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys',
              'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 'movement_acceleration',
              'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping',
              'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression', 'mentality_interceptions', 'mentality_positioning',
              'mentality_vision', 'mentality_penalties', 'mentality_composure', 'defending_marking', 'defending_standing_tackle', 'defending_sliding_tackle',
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
    # players_pm.select(col_pm).show()
    return players_pm



## Impute the data

def impute_data(players_pm):
    '''
    Impute NA values for numeric columns with mean
    :param players_pm: cleaned players data
    :return: imputed data in spark
    '''
    # players_pm.select([count(when((isnan(c) | col(c).isNull()), c)).alias(c) for c in players_pm.columns]).show()

    # [c for c in players_pm.columns if dict(players_pm.dtypes)[c] == "int"]
    num_null_col = ['team_jersey_number', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'skill_dribbling',
                    'movement_acceleration', 'movement_sprint_speed', 'movement_agility','movement_balance','power_jumping',
                    'power_stamina', 'power_strength', 'power_long_shots', 'mentality_positioning', 'mentality_composure',
                    'defending_marking', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
                    'goalkeeping_positioning', 'goalkeeping_reflexes', 'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw',
                    'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb',
                    'cb', 'rcb', 'rb']

    imputer_players = Imputer(
        inputCols = num_null_col,
        outputCols = ["{}_imputed".format(c) for c in num_null_col]
        ).setStrategy("mean")

    players_imputed = imputer_players.fit(players_pm).transform(players_pm)

    for c in num_null_col:
        players_imputed = players_imputed.drop(c).withColumnRenamed("{}_imputed".format(c), c)
    return players_imputed

#players_imputed.select([count(when((isnan(c) | col(c).isNull()), c)).alias(c) for c in players_imputed.columns]).show()

### Three cols with missing values left:
###     team_position: 1326, joined: 8038, contract_valid_until: 1333, nation_position: 94470 (kept for task 2)

# Cast types

def cast_type(players_imputed):
    '''
    Cast the data into correct data types
    :param players_imputed: imputed data
    :return: final players data ready for analysis
    '''
    players_ctype = players_imputed.withColumn("dob", to_date(col("dob"), "yyyy-mm-dd"))\
        .withColumn("joined", to_date(col("joined"), "yyyy-mm-dd"))
    return players_ctype

# Now the data is ready for import

def delete_db():
    '''
    Delete schema Fifa in the database
    '''
    conn = psycopg2.connect(database="postgres", user='postgres', password=pw, host='127.0.0.1', port='5432')
    cursor = conn.cursor()
    call = '''Drop schema if exists FIFA cascade;'''
    cursor.execute(call)
    conn.commit()
    conn.close()
    print("Database deleted!")

def init_db():
    '''
    initialize the database
    '''
    conn = psycopg2.connect(database="postgres", user='postgres', password=pw, host='127.0.0.1', port='5432')
    cursor = conn.cursor()
    call = '''CREATE SCHEMA IF NOT EXISTS FIFA;
    create table if not exists FIFA.players(
    sofifa_id integer primary key,
    player_url varchar,
    short_name varchar,
    long_name varchar,
    age integer,
    dob date,
    height_cm integer,
    weight_kg integer,
    nationality varchar,
    club varchar,
    overall integer,
    potential integer,
    value_eur integer,
    wage_eur integer,
    player_positions varchar,
    preferred_foot varchar,
    international_reputation integer,
    weak_foot integer,
    skill_moves integer,
    work_rate varchar,
    body_type varchar,
    real_face varchar,
    team_position varchar,
    joined date,
    contract_valid_until integer,
    attacking_crossing integer,
    attacking_finishing integer,
    attacking_heading_accuracy integer,
    attacking_short_passing integer,
    attacking_volleys integer,
    skill_dribbling integer,
    skill_curve integer,
    skill_fk_accuracy integer,
    skill_long_passing integer,
    skill_ball_control integer,
    movement_acceleration integer,
    movement_sprint_speed integer,
    movement_agility integer,
    movement_reactions integer,
    movement_balance integer,
    power_shot_power integer,
    power_jumping integer,
    power_stamina integer,
    power_strength integer,
    power_long_shots integer,
    mentality_aggression integer,
    mentality_interceptions integer,
    mentality_positioning integer,
    mentality_vision integer,
    mentality_penalties integer,
    defending_marking integer,
    defending_standing_tackle integer,
    defending_sliding_tackle integer,
    goalkeeping_diving integer,
    goalkeeping_handling integer,
    goalkeeping_kicking integer,
    goalkeeping_positioning integer,
    goalkeeping_reflexes integer,
    year integer,
    team_jersey_number integer,
    pace integer,
    shooting integer,
    passing integer,
    dribbling integer,
    defending integer,
    physic integer,
    ls integer,
    st integer,
    rs integer,
    lw integer,
    lf integer,
    cf integer,
    rf integer,
    rw integer,
    lam integer,
    cam integer,
    ram integer,
    lm integer,
    lcm integer,
    cm integer,
    rcm integer,
    rm integer,
    lwb integer,
    ldm integer,
    cdm integer,
    rdm integer,
    rwb integer,
    lb integer,
    lcb integer,
    cb integer,
    rcb integer,
    rb integer
    );'''
    cursor.execute(call)
    conn.commit()
    conn.close()
    print("Database initialized!")

def data_import(data):
    '''
    Import the data into SQL database
    :param data: Final data produced by cast_type()
    '''
    db_properties={}
    db_properties['username']="postgres"
    db_properties['password']=pw

    db_properties['url']= "jdbc:postgresql://localhost:5432/postgres"
    db_properties['driver']="org.postgresql.Driver"

    data.write.format("jdbc")\
    .mode("overwrite")\
    .option("url", "jdbc:postgresql://localhost:5432/postgres")\
    .option("dbtable", "FIFA.players")\
    .option("user", "postgres")\
    .option("password", pw)\
    .option("Driver", "org.postgresql.Driver")\
    .save()
    print("Data imported!")

def run_task1_module():
    '''
    Run all the steps in Task 1, include reading data, union, data cleaning, imputation, and (re)construct database
    '''
    spark,sqlContext = init_spark()
    players_unioned = read_data(spark)
    players_pm = clean_data(players_unioned)
    players_imputed = impute_data(players_pm)
    players_ctype = cast_type(players_imputed)
    delete_db()
    init_db()
    data_import(players_ctype)










