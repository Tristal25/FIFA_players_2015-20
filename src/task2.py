import findspark
import pyspark
from pyspark.sql import SparkSession, SQLContext, DataFrame
from pyspark.sql.functions import *
import os
from src.task1 import init_spark, pw
from functools import reduce

findspark.init(spark_home = "/Users/tristalli/spark-3.2.0-bin-hadoop3.2",
               python_path = "/opt/anaconda3/bin/python3")

def load_data():
    '''
    Load the players table from SQL database
    :return: Players dataset ready for analysis
    '''
    spark, sqlContext = init_spark()
    players = sqlContext.read.format("jdbc") \
        .option("url", "jdbc:postgresql://localhost:5432/postgres") \
        .option("dbtable", "FIFA.players") \
        .option("user", "postgres") \
        .option("password", pw) \
        .option("Driver", "org.postgresql.Driver") \
        .load()
    return players


## 1

def top_players(players, x):
    '''
    List the x players who achieved highest improvement across all skillsets. The steps are:
    1. Overall skill scores are calculated by aggregated scores of 40 columns.
    2. The improvement is the overall skill score in 2020 minus the one in 2015.
    3. Choose the players with top  x improvement scores.
    :param players: the dataset players
    :param x: the number of players to output
    :return: a list of full names of players with top x improvements
    '''

    def column_add(a, b):
        return a.__add__(b)

    if not isinstance(x, int):
        raise TypeError("Input should be an integer value")

    if x <= 0:
        raise TypeError("Input should be positive")

    try:
        skill_cols = ["skill_dribbling", "skill_curve", "skill_fk_accuracy", "skill_long_passing", "skill_ball_control",
                      "pace", "shooting", "passing", "defending", "physic", "dribbling", "attacking_crossing", "power_shot_power",
                      "attacking_finishing", "attacking_heading_accuracy", "attacking_short_passing", "attacking_volleys",
                      "movement_acceleration", "movement_sprint_speed", "movement_agility", "movement_balance","movement_reactions",
                      "power_jumping", "power_stamina", "power_strength", "power_long_shots", "mentality_positioning",
                      "mentality_composure", "mentality_aggression", "mentality_interceptions", "mentality_vision",
                      "mentality_penalties", "defending_marking", "defending_standing_tackle", "defending_sliding_tackle",
                      "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking", "goalkeeping_positioning",
                      "goalkeeping_reflexes"]
        players_15 = players.filter(players.year == 2015)\
            .withColumn("skill_score_2015", reduce(column_add, (players[i] for i in skill_cols)))\
            .select("short_name", "long_name", "skill_score_2015")
        players_20 = players.filter(players.year == 2020) \
            .withColumn("skill_score_2020", reduce(column_add, (players[i] for i in skill_cols))) \
            .select("short_name", "long_name", "skill_score_2020")

        players_skill = players_15.join(players_20,
                                        [players_15.long_name == players_20.long_name,
                                         players_15.short_name == players_20.short_name], "inner")\
            .select(players_15.long_name, players_15.short_name, players_15.skill_score_2015, players_20.skill_score_2020)
        players_improve = players_skill.withColumn("improvement", col("skill_score_2020")-col("skill_score_2015"))\
            .sort(desc('improvement')).select("long_name").collect()
        playerls = [row[0] for row in players_improve]

    except:
        raise TypeError("Dataset not in correct form, use load_data() instead. ")

    if len(playerls) < x:
        print("x larger than player numbers, returning all the player names possible")

    return playerls[:x]


## 2

def largest_club_2021(players, y, year):
    '''
    List the y clubs that have largest number of players with contracts ending in 2021.
    :param players: the dataset players
    :param y: the number of clubs to output
    :param year: the year of data to be used (2015-2020) in integer
    :return: list of the required club names
    '''

    if not isinstance(y, int):
        raise TypeError("Input should be an integer value")

    if y <= 0:
        raise TypeError("Input should be positive")

    if not isinstance(year, int):
        raise TypeError("Year should be an integer value")

    if year not in [2015, 2016, 2017, 2018, 2019, 2020]:
        raise TypeError("Year should be between 2015 and 2020")

    try:
        dat = players.filter((players.year == year) & (players.contract_valid_until == 2021)).groupBy("club")\
            .agg(countDistinct("long_name"))\
            .orderBy(col("count(long_name)").desc()).select("club").collect()
        clubls = [row[0] for row in dat]
    except:
        raise TypeError("Dataset not in correct form, use load_data() instead. ")

    if len(clubls) < y:
        print("y larger than club numbers, returning all the club names possible")

    return clubls[:y]

## 3

def largest_club(players, z, year):
    '''
    List the z clubs with largest number of players in the dataset where z >= 5.
    :param players: the dataset players
    :param z: the number of clubs to output (z >= 5)
    :param year: the year of data to be used (2015-2020) in integer
    :return: list of the required club names
    '''

    if not isinstance(z, int):
        raise TypeError("Input should be an integer value")

    if z < 5:
        raise TypeError("The number should be greater or equal to 5")

    if not isinstance(year, int):
        raise TypeError("Year should be an integer value")

    if year not in [2015, 2016, 2017, 2018, 2019, 2020]:
        raise TypeError("Year should be between 2015 and 2020")

    try:
        dat = players.filter(players.year == year).groupBy("club").agg(countDistinct("long_name"))\
            .orderBy(col("count(long_name)").desc())
        clubls = [row[0] for row in dat.select("club").collect()]
        countls = [row[0] for row in dat.select("count(long_name)").collect()]
    except:
        raise TypeError("Dataset not in correct form, use load_data() instead. ")

    if all(m==countls[0] for m in countls):
        print("All teams have same number of players! ")

    if len(clubls) < z:
        print("z larger than club numbers, returning all the club names possible")

    return clubls[:z]

## 4

def popular_nation_team(players, year):
    '''
    Get the most popular nation_position and team_position in the dataset
    :param players: the dataset players
    :param year: the year of data to be used (2015-2020) in integer, or "all" for all the years
    :return: a dictionary with nation: most popular nation_position, and team: most popular team position
    '''

    if (not isinstance(year, int)) and year != "all":
        raise TypeError("Year should be an integer value")

    if year not in [2015, 2016, 2017, 2018, 2019, 2020, "all"]:
        raise TypeError("Year should be between 2015 and 2020")

    try:
        if year == "all":
            nation = players.groupBy("nation_position").agg(countDistinct("long_name"))\
                .orderBy(col("count(long_name)").desc()).select("nation_position").collect()
            team = players.groupBy("team_position").agg(countDistinct("long_name")) \
                .orderBy(col("count(long_name)").desc()).select("team_position").collect()
        else:
            nation = players.filter(players.year == year).groupBy("nation_position").agg(countDistinct("long_name")) \
                .orderBy(col("count(long_name)").desc()).select("nation_position").collect()
            team = players.filter(players.year == year).groupBy("team_position").agg(countDistinct("long_name")) \
                .orderBy(col("count(long_name)").desc()).select("team_position").collect()
        if nation[0][0] == None:
            nation_sol = nation[1][0]
        else:
            nation_sol = nation[0][0]
        if team[0][0] == None:
            team_sol = team[1][0]
        else:
            team_sol = team[0][0]
    except:
        raise TypeError("Dataset not in correct form, use load_data() instead. ")

    return {"nation": nation_sol, "team": team_sol}

## 5

def popular_nationality(players, year):
    '''
    Get the most popular nationality for the players in the dataset
    :param players: the dataset players
    :param year: the year of data to be used (2015-2020) in integer, or "all" for all the years
    :return: Name of the most popular nationality for the players in the dataset in string
    '''

    if (not isinstance(year, int)) and year != "all":
        raise TypeError("Year should be an integer value")

    if year not in [2015, 2016, 2017, 2018, 2019, 2020, "all"]:
        raise TypeError("Year should be between 2015 and 2020")

    try:
        if year == "all":
            nation = players.groupBy("nationality").agg(countDistinct("long_name")) \
                .orderBy(col("count(long_name)").desc()).select("nationality").collect()
        else:
            nation = players.filter(players.year == year).groupBy("nationality").agg(countDistinct("long_name")) \
                .orderBy(col("count(long_name)").desc()).select("nationality").collect()

        if nation[0][0] == None:
            nation_sol = nation[1][0]
        else:
            nation_sol = nation[0][0]
    except:
        raise TypeError("Dataset not in correct form, use load_data() instead. ")

    return nation_sol


