import findspark
import pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext, DataFrame
from pyspark.ml.feature import Imputer
from pyspark.sql.functions import *
from functools import reduce
import psycopg2
import os
from src.task1 import *
from src.task2 import *

findspark.init(spark_home = "/Users/tristalli/spark-3.2.0-bin-hadoop3.2",
               python_path = "/opt/anaconda3/bin/python3")


if __name__ == "__main__":
    # Task I
    run_task1_module()
    # Task II
    players = load_data()
    print(top_players(players, 4))
    print(largest_club_2021(players, 5, 2017))
    print(largest_club(players, 6, 2018))
    print(popular_nation_team(players, 2019))
    print(popular_nationality(players, 2020))


























