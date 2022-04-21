from src.main import *
import pandas as pd
import pyspark
from pyspark.sql import SparkSession, SQLContext, DataFrame
from pyspark.sql.functions import *
import unittest
from pytest_spark import *


def get_sorted_data_frame(data_frame, columns_list):
    return data_frame.sort_values(columns_list).reset_index(drop=True)


def test_load_data():
    players = load_data()
    col_full = ['sofifa_id', 'player_url', 'short_name', 'long_name', 'age', 'dob', 'height_cm', 'weight_kg',
                'nationality',
                'club', 'overall', 'potential', 'value_eur', 'wage_eur', 'player_positions', 'preferred_foot',
                'international_reputation', 'weak_foot', 'skill_moves', 'work_rate', 'body_type', 'real_face',
                'team_position',
                'joined', 'contract_valid_until', 'nation_position', 'year', 'attacking_crossing',
                'attacking_finishing',
                'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys', 'skill_curve',
                'skill_fk_accuracy',
                'skill_long_passing', 'skill_ball_control', 'movement_reactions', 'power_shot_power',
                'mentality_aggression',
                'mentality_interceptions', 'mentality_vision', 'mentality_penalties', 'defending_standing_tackle',
                'defending_sliding_tackle', 'team_jersey_number', 'pace', 'shooting', 'passing', 'dribbling',
                'defending',
                'physic', 'skill_dribbling', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
                'movement_balance', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
                'mentality_positioning', 'mentality_composure', 'defending_marking', 'goalkeeping_diving',
                'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes', 'ls',
                'st',
                'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm',
                'cdm',
                'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb']
    assert players is not None, "data not loaded"
    assert players.columns == col_full, "Columns wrong"
    assert players.count() == 100995, "Rows number wrong"

# 1
def test_top_players_happy(spark_session):
    skill_cols = ["skill_dribbling", "skill_curve", "skill_fk_accuracy", "skill_long_passing", "skill_ball_control",
                  "pace", "shooting", "passing", "defending", "physic", "dribbling", "attacking_crossing",
                  "power_shot_power",
                  "attacking_finishing", "attacking_heading_accuracy", "attacking_short_passing", "attacking_volleys",
                  "movement_acceleration", "movement_sprint_speed", "movement_agility", "movement_balance",
                  "movement_reactions",
                  "power_jumping", "power_stamina", "power_strength", "power_long_shots", "mentality_positioning",
                  "mentality_composure", "mentality_aggression", "mentality_interceptions", "mentality_vision",
                  "mentality_penalties", "defending_marking", "defending_standing_tackle", "defending_sliding_tackle",
                  "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking", "goalkeeping_positioning",
                  "goalkeeping_reflexes"]
    input = spark_session.createDataFrame(
        [(2020, "a", "ab")+(90,)*len(skill_cols),
         (2020, "b", 'bc')+(90,)*len(skill_cols),
         (2020, "c", "cd")+(90,)*len(skill_cols),
         (2015, "a", "ab")+(50,)*len(skill_cols),
         (2015, "b", "bc")+(60,)*len(skill_cols),
         (2015, "c", "cd")+(70,)*len(skill_cols),
         (2015, "q", "pq")+(10,)*len(skill_cols)],
        ["year", "short_name", "long_name"]+skill_cols,
    )
    expected_output_1 = ["ab"]
    expected_output_2 = ['ab', 'bc']
    expected_output_3 = ['ab', 'bc', 'cd']
    assert top_players(input, 1) is not None, "None returned for valid input"
    assert top_players(input, 1) == expected_output_1
    assert top_players(input, 2) == expected_output_2
    assert top_players(input, 3) == expected_output_3

def test_top_players_sad(spark_session):
    skill_cols = ["skill_dribbling", "skill_curve", "skill_fk_accuracy", "skill_long_passing", "skill_ball_control",
                  "pace", "shooting", "passing", "defending", "physic", "dribbling", "attacking_crossing",
                  "power_shot_power",
                  "attacking_finishing", "attacking_heading_accuracy", "attacking_short_passing", "attacking_volleys",
                  "movement_acceleration", "movement_sprint_speed", "movement_agility", "movement_balance",
                  "movement_reactions",
                  "power_jumping", "power_stamina", "power_strength", "power_long_shots", "mentality_positioning",
                  "mentality_composure", "mentality_aggression", "mentality_interceptions", "mentality_vision",
                  "mentality_penalties", "defending_marking", "defending_standing_tackle", "defending_sliding_tackle",
                  "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking", "goalkeeping_positioning",
                  "goalkeeping_reflexes"]
    input = spark_session.createDataFrame(
        [(2020, "a", "ab") + (90,) * len(skill_cols),
         (2020, "b", 'bc') + (90,) * len(skill_cols),
         (2020, "c", "cd") + (90,) * len(skill_cols),
         (2015, "a", "ab") + (50,) * len(skill_cols),
         (2015, "b", "bc") + (60,) * len(skill_cols),
         (2015, "c", "cd") + (70,) * len(skill_cols),
         (2015, "q", "pq") + (10,) * len(skill_cols)],
        ["year", "short_name", "long_name"] + skill_cols,
    )
    assert top_players(input, 5) == ['ab', 'bc', 'cd'], "Error when x large"

    try:
        top_players(input, None)
        assert False
    except TypeError:
        assert True

    try:
        top_players(input, "abc")
        assert False
    except TypeError:
        assert True

    try:
        top_players(input, 1.5)
        assert False
    except TypeError:
        assert True

    try:
        top_players(input, -2)
        assert False
    except TypeError:
        assert True

    try:
        top_players(input, 0)
        assert False
    except TypeError:
        assert True

    try:
        top_players(["abc"], 3)
        assert False
    except TypeError:
        assert True

    try:
        top_players("abc", 3)
        assert False
    except TypeError:
        assert True

    try:
        top_players(2, 3)
        assert False
    except TypeError:
        assert True

# 2
def test_largest_club_2021_happy(spark_session):
    input = spark_session.createDataFrame(
        [(2015, "q", 2021, "A"),
         (2015, 'w', 2021, "B"),
         (2015, "e", 2021, "C"),
         (2015, "r", 2020, "M"),
         (2015, "t", 2021, "A"),
         (2015, "y", 2010, "M"),
         (2015, "u", 2021, "B"),
         (2015, "i", 2021, "A"),
         (2015, "w", 2021, "B"),
         (2016, "q", 2021, "A"),
         (2016, 'w', 2021, "B"),
         (2016, "e", 2021, "C"),
         (2016, "r", 2020, "M"),
         (2016, "t", 2021, "A"),
         (2016, "y", 2010, "M"),
         (2016, "u", 2021, "B"),
         (2016, "i", 2021, "A"),
         (2016, "w", 2021, "B"),
         (2017, "q", 2021, "A"),
         (2017, 'w', 2021, "B"),
         (2017, "e", 2021, "C"),
         (2017, "r", 2020, "M"),
         (2017, "t", 2021, "A"),
         (2017, "y", 2010, "M"),
         (2017, "u", 2021, "B"),
         (2017, "i", 2021, "A"),
         (2017, "w", 2021, "B"),
         (2018, "q", 2021, "A"),
         (2018, 'w', 2021, "B"),
         (2018, "e", 2021, "C"),
         (2018, "r", 2020, "M"),
         (2018, "t", 2021, "A"),
         (2018, "y", 2010, "M"),
         (2018, "u", 2021, "B"),
         (2018, "i", 2021, "A"),
         (2018, "w", 2021, "B"),
         (2019, "q", 2021, "A"),
         (2019, 'w', 2021, "B"),
         (2019, "e", 2021, "C"),
         (2019, "r", 2020, "M"),
         (2019, "t", 2021, "A"),
         (2019, "y", 2010, "M"),
         (2019, "u", 2021, "B"),
         (2019, "i", 2021, "A"),
         (2019, "w", 2021, "B"),
         (2020, "q", 2021, "A"),
         (2020, 'w', 2021, "B"),
         (2020, "e", 2021, "C"),
         (2020, "r", 2020, "M"),
         (2020, "t", 2021, "A"),
         (2020, "y", 2010, "M"),
         (2020, "u", 2021, "B"),
         (2020, "i", 2021, "A"),
         (2020, "w", 2021, "B")
         ],
        ["year", "long_name", "contract_valid_until", "club"],
    )

    assert largest_club_2021(input, 1, 2015) is not None, "None returned for valid input"
    assert largest_club_2021(input, 1, 2015) == ['A'], "largest_club_2021(input, 1, 2015) wrong"
    assert largest_club_2021(input, 2, 2016) == ['A', 'B'], "largest_club_2021(input, 2, 2016) wrong"
    assert largest_club_2021(input, 3, 2017) == ['A', 'B', 'C'], "largest_club_2021(input, 3, 2017) wrong"
    assert largest_club_2021(input, 3, 2018) == ['A', 'B', 'C'], "largest_club_2021(input, 3, 2018) wrong"
    assert largest_club_2021(input, 2, 2019) == ['A', 'B'], "largest_club_2021(input, 2, 2019) wrong"
    assert largest_club_2021(input, 1, 2020) == ['A'], "largest_club_2021(input, 1, 2020) wrong"

def test_largest_club_2021_sad(spark_session):
    input = spark_session.createDataFrame(
        [(2015, "q", 2021, "A"),
         (2015, 'w', 2021, "B"),
         (2015, "e", 2021, "C"),
         (2015, "r", 2020, "M"),
         (2015, "t", 2021, "A"),
         (2015, "y", 2010, "M"),
         (2015, "u", 2021, "B"),
         (2015, "i", 2021, "A"),
         (2015, "w", 2021, "B"),
         (2016, "q", 2021, "A"),
         (2016, 'w', 2021, "B"),
         (2016, "e", 2021, "C"),
         (2016, "r", 2020, "M"),
         (2016, "t", 2021, "A"),
         (2016, "y", 2010, "M"),
         (2016, "u", 2021, "B"),
         (2016, "i", 2021, "A"),
         (2016, "w", 2021, "B"),
         (2017, "q", 2021, "A"),
         (2017, 'w', 2021, "B"),
         (2017, "e", 2021, "C"),
         (2017, "r", 2020, "M"),
         (2017, "t", 2021, "A"),
         (2017, "y", 2010, "M"),
         (2017, "u", 2021, "B"),
         (2017, "i", 2021, "A"),
         (2017, "w", 2021, "B"),
         (2018, "q", 2021, "A"),
         (2018, 'w', 2021, "B"),
         (2018, "e", 2021, "C"),
         (2018, "r", 2020, "M"),
         (2018, "t", 2021, "A"),
         (2018, "y", 2010, "M"),
         (2018, "u", 2021, "B"),
         (2018, "i", 2021, "A"),
         (2018, "w", 2021, "B"),
         (2019, "q", 2021, "A"),
         (2019, 'w', 2021, "B"),
         (2019, "e", 2021, "C"),
         (2019, "r", 2020, "M"),
         (2019, "t", 2021, "A"),
         (2019, "y", 2010, "M"),
         (2019, "u", 2021, "B"),
         (2019, "i", 2021, "A"),
         (2019, "w", 2021, "B"),
         (2020, "q", 2021, "A"),
         (2020, 'w', 2021, "B"),
         (2020, "e", 2021, "C"),
         (2020, "r", 2020, "M"),
         (2020, "t", 2021, "A"),
         (2020, "y", 2010, "M"),
         (2020, "u", 2021, "B"),
         (2020, "i", 2021, "A"),
         (2020, "w", 2021, "B")
         ],
        ["year", "long_name", "contract_valid_until", "club"],
    )

    assert largest_club_2021(input, 5, 2020) == ['A', 'B', 'C'], "Error when x large"

    try:
        largest_club_2021(input, None, 2020)
        assert False
    except TypeError:
        assert True

    try:
        largest_club_2021(input, "abc", 2019)
        assert False
    except TypeError:
        assert True

    try:
        largest_club_2021(input, 1.5, 2020)
        assert False
    except TypeError:
        assert True

    try:
        largest_club_2021(input, -2, 2018)
        assert False
    except TypeError:
        assert True

    try:
        largest_club_2021(input, 0, 2018)
        assert False
    except TypeError:
        assert True

    try:
        largest_club_2021(["abc"], 3, 2017)
        assert False
    except TypeError:
        assert True

    try:
        largest_club_2021("abc", 3, 2017)
        assert False
    except TypeError:
        assert True

    try:
        largest_club_2021(2, 3, 2019)
        assert False
    except TypeError:
        assert True

    try:
        largest_club_2021(input, 3, None)
        assert False
    except TypeError:
        assert True

    try:
        largest_club_2021(input, 3, -1)
        assert False
    except TypeError:
        assert True

    try:
        largest_club_2021(input, 3, 2023)
        assert False
    except TypeError:
        assert True

    try:
        largest_club_2021(input, 3, 1.5)
        assert False
    except TypeError:
        assert True

    try:
        largest_club_2021(input, 3, "abc")
        assert False
    except TypeError:
        assert True

# 3

def test_largest_club_happy(spark_session):
    input = spark_session.createDataFrame(
        [(2015, "A", "A"),
         (2015, "B", "A"),
         (2015, "C", "A"),
         (2015, "D", "A"),
         (2015, "E", "A"),
         (2015, "F", "A"),
         (2015, "G", "B"),
         (2015, "H", "B"),
         (2015, "I", "B"),
         (2015, "J", "B"),
         (2015, "K", "B"),
         (2015, "L", "C"),
         (2015, "M", "C"),
         (2015, "N", "C"),
         (2015, "O", "C"),
         (2015, "P", "D"),
         (2015, "Q", "D"),
         (2015, "R", "D"),
         (2015, "S", "E"),
         (2015, "T", "E"),
         (2015, "U", "F"),
         (2016, "A", "A"),
         (2016, "B", "A"),
         (2016, "C", "A"),
         (2016, "D", "A"),
         (2016, "E", "A"),
         (2016, "F", "A"),
         (2016, "G", "B"),
         (2016, "H", "B"),
         (2016, "I", "B"),
         (2016, "J", "B"),
         (2016, "K", "B"),
         (2016, "L", "C"),
         (2016, "M", "C"),
         (2016, "N", "C"),
         (2016, "O", "C"),
         (2016, "P", "D"),
         (2016, "Q", "D"),
         (2016, "R", "D"),
         (2016, "S", "E"),
         (2016, "T", "E"),
         (2016, "U", "F"),
         (2017, "A", "A"),
         (2017, "B", "A"),
         (2017, "C", "A"),
         (2017, "D", "A"),
         (2017, "E", "A"),
         (2017, "F", "A"),
         (2017, "G", "B"),
         (2017, "H", "B"),
         (2017, "I", "B"),
         (2017, "J", "B"),
         (2017, "K", "B"),
         (2017, "L", "C"),
         (2017, "M", "C"),
         (2017, "N", "C"),
         (2017, "O", "C"),
         (2017, "P", "D"),
         (2017, "Q", "D"),
         (2017, "R", "D"),
         (2017, "S", "E"),
         (2017, "T", "E"),
         (2017, "U", "F"),
         (2018, "A", "A"),
         (2018, "B", "A"),
         (2018, "C", "A"),
         (2018, "D", "A"),
         (2018, "E", "A"),
         (2018, "F", "A"),
         (2018, "G", "B"),
         (2018, "H", "B"),
         (2018, "I", "B"),
         (2018, "J", "B"),
         (2018, "K", "B"),
         (2018, "L", "C"),
         (2018, "M", "C"),
         (2018, "N", "C"),
         (2018, "O", "C"),
         (2018, "P", "D"),
         (2018, "Q", "D"),
         (2018, "R", "D"),
         (2018, "S", "E"),
         (2018, "T", "E"),
         (2018, "U", "F"),
         (2019, "A", "A"),
         (2019, "B", "A"),
         (2019, "C", "A"),
         (2019, "D", "A"),
         (2019, "E", "A"),
         (2019, "F", "A"),
         (2019, "G", "B"),
         (2019, "H", "B"),
         (2019, "I", "B"),
         (2019, "J", "B"),
         (2019, "K", "B"),
         (2019, "L", "C"),
         (2019, "M", "C"),
         (2019, "N", "C"),
         (2019, "O", "C"),
         (2019, "P", "D"),
         (2019, "Q", "D"),
         (2019, "R", "D"),
         (2019, "S", "E"),
         (2019, "T", "E"),
         (2019, "U", "F"),
         (2020, "A", "A"),
         (2020, "B", "A"),
         (2020, "C", "A"),
         (2020, "D", "A"),
         (2020, "E", "A"),
         (2020, "F", "A"),
         (2020, "G", "B"),
         (2020, "H", "B"),
         (2020, "I", "B"),
         (2020, "J", "B"),
         (2020, "K", "B"),
         (2020, "L", "C"),
         (2020, "M", "C"),
         (2020, "N", "C"),
         (2020, "O", "C"),
         (2020, "P", "D"),
         (2020, "Q", "D"),
         (2020, "R", "D"),
         (2020, "S", "E"),
         (2020, "T", "E"),
         (2020, "U", "F"),
         ],
        ["year", "long_name", "club"],
    )
    assert largest_club(input, 5, 2015) is not None, "None returned for valid input"
    assert largest_club(input, 5, 2015) == ['A', 'B', 'C', 'D', 'E'], "largest_club(input, 5, 2015) wrong"
    assert largest_club(input, 6, 2015) == ['A', 'B', 'C', 'D', 'E', 'F'], "largest_club(input, 6, 2015) wrong"
    assert largest_club(input, 5, 2016) == ['A', 'B', 'C', 'D', 'E'], "largest_club(input, 5, 2016) wrong"
    assert largest_club(input, 6, 2016) == ['A', 'B', 'C', 'D', 'E', 'F'], "largest_club(input, 6, 2016) wrong"
    assert largest_club(input, 5, 2017) == ['A', 'B', 'C', 'D', 'E'], "largest_club(input, 5, 2017) wrong"
    assert largest_club(input, 6, 2017) == ['A', 'B', 'C', 'D', 'E', 'F'], "largest_club(input, 6, 2017) wrong"
    assert largest_club(input, 5, 2018) == ['A', 'B', 'C', 'D', 'E'], "largest_club(input, 5, 2018) wrong"
    assert largest_club(input, 6, 2018) == ['A', 'B', 'C', 'D', 'E', 'F'], "largest_club(input, 6, 2018) wrong"
    assert largest_club(input, 5, 2019) == ['A', 'B', 'C', 'D', 'E'], "largest_club(input, 5, 2019) wrong"
    assert largest_club(input, 6, 2019) == ['A', 'B', 'C', 'D', 'E', 'F'], "largest_club(input, 6, 2019) wrong"
    assert largest_club(input, 5, 2020) == ['A', 'B', 'C', 'D', 'E'], "largest_club(input, 5, 2020) wrong"
    assert largest_club(input, 6, 2020) == ['A', 'B', 'C', 'D', 'E', 'F'], "largest_club(input, 6, 2020) wrong"


def test_largest_club_sad(spark_session):
    input = spark_session.createDataFrame(
        [(2015, "A", "A"),
         (2015, "B", "A"),
         (2015, "C", "A"),
         (2015, "D", "A"),
         (2015, "E", "A"),
         (2015, "F", "A"),
         (2015, "G", "B"),
         (2015, "H", "B"),
         (2015, "I", "B"),
         (2015, "J", "B"),
         (2015, "K", "B"),
         (2015, "L", "C"),
         (2015, "M", "C"),
         (2015, "N", "C"),
         (2015, "O", "C"),
         (2015, "P", "D"),
         (2015, "Q", "D"),
         (2015, "R", "D"),
         (2015, "S", "E"),
         (2015, "T", "E"),
         (2015, "U", "F"),
         (2016, "A", "A"),
         (2016, "B", "A"),
         (2016, "C", "A"),
         (2016, "D", "A"),
         (2016, "E", "A"),
         (2016, "F", "A"),
         (2016, "G", "B"),
         (2016, "H", "B"),
         (2016, "I", "B"),
         (2016, "J", "B"),
         (2016, "K", "B"),
         (2016, "L", "C"),
         (2016, "M", "C"),
         (2016, "N", "C"),
         (2016, "O", "C"),
         (2016, "P", "D"),
         (2016, "Q", "D"),
         (2016, "R", "D"),
         (2016, "S", "E"),
         (2016, "T", "E"),
         (2016, "U", "F"),
         (2017, "A", "A"),
         (2017, "B", "A"),
         (2017, "C", "A"),
         (2017, "D", "A"),
         (2017, "E", "A"),
         (2017, "F", "A"),
         (2017, "G", "B"),
         (2017, "H", "B"),
         (2017, "I", "B"),
         (2017, "J", "B"),
         (2017, "K", "B"),
         (2017, "L", "C"),
         (2017, "M", "C"),
         (2017, "N", "C"),
         (2017, "O", "C"),
         (2017, "P", "D"),
         (2017, "Q", "D"),
         (2017, "R", "D"),
         (2017, "S", "E"),
         (2017, "T", "E"),
         (2017, "U", "F"),
         (2018, "A", "A"),
         (2018, "B", "A"),
         (2018, "C", "A"),
         (2018, "D", "A"),
         (2018, "E", "A"),
         (2018, "F", "A"),
         (2018, "G", "B"),
         (2018, "H", "B"),
         (2018, "I", "B"),
         (2018, "J", "B"),
         (2018, "K", "B"),
         (2018, "L", "C"),
         (2018, "M", "C"),
         (2018, "N", "C"),
         (2018, "O", "C"),
         (2018, "P", "D"),
         (2018, "Q", "D"),
         (2018, "R", "D"),
         (2018, "S", "E"),
         (2018, "T", "E"),
         (2018, "U", "F"),
         (2019, "A", "A"),
         (2019, "B", "A"),
         (2019, "C", "A"),
         (2019, "D", "A"),
         (2019, "E", "A"),
         (2019, "F", "A"),
         (2019, "G", "B"),
         (2019, "H", "B"),
         (2019, "I", "B"),
         (2019, "J", "B"),
         (2019, "K", "B"),
         (2019, "L", "C"),
         (2019, "M", "C"),
         (2019, "N", "C"),
         (2019, "O", "C"),
         (2019, "P", "D"),
         (2019, "Q", "D"),
         (2019, "R", "D"),
         (2019, "S", "E"),
         (2019, "T", "E"),
         (2019, "U", "F"),
         (2020, "A", "A"),
         (2020, "B", "A"),
         (2020, "C", "A"),
         (2020, "D", "A"),
         (2020, "E", "A"),
         (2020, "F", "A"),
         (2020, "G", "B"),
         (2020, "H", "B"),
         (2020, "I", "B"),
         (2020, "J", "B"),
         (2020, "K", "B"),
         (2020, "L", "C"),
         (2020, "M", "C"),
         (2020, "N", "C"),
         (2020, "O", "C"),
         (2020, "P", "D"),
         (2020, "Q", "D"),
         (2020, "R", "D"),
         (2020, "S", "E"),
         (2020, "T", "E"),
         (2020, "U", "F"),
         ],
        ["year", "long_name", "club"],
    )
    input_same = spark_session.createDataFrame(
        [(2015, "q", "A"),
         (2015, 'w', "A"),
         (2015, "e", "B"),
         (2015, "t", "B"),
         (2015, "u", "C"),
         (2015, "i", "C"),
         (2015, "m", "D"),
         (2015, "a", "D"),
         (2015, "s", "E"),
         (2015, "d", "E")
         ],
        ["year", "long_name", "club"],
    )

    assert largest_club(input, 9, 2020) == ['A', 'B', 'C', 'D', 'E', 'F'], "Error when x large"
    assert sorted(largest_club(input_same, 5, 2015)) == ['A', 'B', 'C', 'D', 'E'], "Error when all clubs are same in size"

    try:
        largest_club(input, None, 2020)
        assert False
    except TypeError:
        assert True

    try:
        largest_club(input, 1, 2020)
        assert False
    except TypeError:
        assert True

    try:
        largest_club(input, "abc", 2019)
        assert False
    except TypeError:
        assert True

    try:
        largest_club_2021(input, 1.5, 2020)
        assert False
    except TypeError:
        assert True

    try:
        largest_club(input, -2, 2018)
        assert False
    except TypeError:
        assert True

    try:
        largest_club(input, 0, 2018)
        assert False
    except TypeError:
        assert True

    try:
        largest_club(["abc"], 6, 2017)
        assert False
    except TypeError:
        assert True

    try:
        largest_club("abc", 6, 2017)
        assert False
    except TypeError:
        assert True

    try:
        largest_club(2, 6, 2019)
        assert False
    except TypeError:
        assert True

    try:
        largest_club(input, 6, None)
        assert False
    except TypeError:
        assert True

    try:
        largest_club(input, 6, -1)
        assert False
    except TypeError:
        assert True

    try:
        largest_club(input, 6, 2023)
        assert False
    except TypeError:
        assert True

    try:
        largest_club(input, 6, 1.5)
        assert False
    except TypeError:
        assert True

    try:
        largest_club(input, 6, "abc")
        assert False
    except TypeError:
        assert True


# 4

def test_popular_nation_team_happy(spark_session):
    input = spark_session.createDataFrame(
        [(2015, "A", "A", "A"),
         (2015, "B", "A", "B"),
         (2015, "C", "A", "B"),
         (2015, "D", "B", "C"),
         (2015, "E", "B", "C"),
         (2015, "F", "C", "C"),
         (2016, "A", "A", "A"),
         (2016, "B", "A", "B"),
         (2016, "C", "A", "B"),
         (2016, "D", "B", "C"),
         (2016, "E", "B", "C"),
         (2016, "F", "C", "C"),
         (2017, "A", "A", "A"),
         (2017, "B", "A", "B"),
         (2017, "C", "A", "B"),
         (2017, "D", "B", "C"),
         (2017, "E", "B", "C"),
         (2017, "F", "C", "C"),
         (2018, "A", "A", "A"),
         (2018, "B", "A", "B"),
         (2018, "C", "A", "B"),
         (2018, "D", "B", "C"),
         (2018, "E", "B", "C"),
         (2018, "F", "C", "C"),
         (2019, "A", "A", "A"),
         (2019, "B", "A", "B"),
         (2019, "C", "A", "B"),
         (2019, "D", "B", "C"),
         (2019, "E", "B", "C"),
         (2019, "F", "C", "C"),
         (2020, "A", "A", "A"),
         (2020, "B", "A", "B"),
         (2020, "C", "A", "B"),
         (2020, "D", "B", "C"),
         (2020, "E", "B", "C"),
         (2020, "F", "C", "C"),
         ],
        ["year", "long_name", "nation_position", "team_position"],
    )
    assert popular_nation_team(input, 2015) is not None, "None returned for valid input"
    assert popular_nation_team(input, 2015) == {"nation": "A", "team": "C"}, "popular_nation_team(input, 2015) wrong"
    assert popular_nation_team(input, 2016) == {"nation": "A", "team": "C"}, "popular_nation_team(input, 2016) wrong"
    assert popular_nation_team(input, 2017) == {"nation": "A", "team": "C"}, "popular_nation_team(input, 2017) wrong"
    assert popular_nation_team(input, 2018) == {"nation": "A", "team": "C"}, "popular_nation_team(input, 2018) wrong"
    assert popular_nation_team(input, 2019) == {"nation": "A", "team": "C"}, "popular_nation_team(input, 2019) wrong"
    assert popular_nation_team(input, 2020) == {"nation": "A", "team": "C"}, "popular_nation_team(input, 2020) wrong"
    assert popular_nation_team(input, "all") == {"nation": "A", "team": "C"}, "popular_nation_team(input, 'all') wrong"


def test_popular_nation_team_sad(spark_session):
    input2 = spark_session.createDataFrame(
        [(2015, "A", "A", "A"),
         (2015, "B", "A", "B"),
         (2015, "C", "A", "B"),
         (2015, "D", "B", "C"),
         (2015, "E", "B", "C"),
         (2015, "F", "C", "C"),
         (2015, "A", None, None),
         (2015, "B", None, None),
         (2015, "C", None, None),
         (2015, "D", None, None),
         (2015, "E", None, None),
         (2015, "F", None, None),
         (2016, "A", "A", "A"),
         (2016, "B", "A", "B"),
         (2016, "C", "A", "B"),
         (2016, "D", "B", "C"),
         (2016, "E", "B", "C"),
         (2016, "F", "C", "C"),
         (2016, "A", None, None),
         (2016, "B", None, None),
         (2016, "C", None, None),
         (2016, "D", None, None),
         (2016, "E", None, None),
         (2016, "F", None, None),
         (2017, "A", "A", "A"),
         (2017, "B", "A", "B"),
         (2017, "C", "A", "B"),
         (2017, "D", "B", "C"),
         (2017, "E", "B", "C"),
         (2017, "F", "C", "C"),
         (2017, "A", None, None),
         (2017, "B", None, None),
         (2017, "C", None, None),
         (2017, "D", None, None),
         (2017, "E", None, None),
         (2017, "F", None, None),
         (2018, "A", "A", "A"),
         (2018, "B", "A", "B"),
         (2018, "C", "A", "B"),
         (2018, "D", "B", "C"),
         (2018, "E", "B", "C"),
         (2018, "F", "C", "C"),
         (2018, "A", None, None),
         (2018, "B", None, None),
         (2018, "C", None, None),
         (2018, "D", None, None),
         (2018, "E", None, None),
         (2018, "F", None, None),
         (2019, "A", "A", "A"),
         (2019, "B", "A", "B"),
         (2019, "C", "A", "B"),
         (2019, "D", "B", "C"),
         (2019, "E", "B", "C"),
         (2019, "F", "C", "C"),
         (2019, "A", None, None),
         (2019, "B", None, None),
         (2019, "C", None, None),
         (2019, "D", None, None),
         (2019, "E", None, None),
         (2019, "F", None, None),
         (2020, "A", "A", "A"),
         (2020, "B", "A", "B"),
         (2020, "C", "A", "B"),
         (2020, "D", "B", "C"),
         (2020, "E", "B", "C"),
         (2020, "F", "C", "C"),
         (2020, "A", None, None),
         (2020, "B", None, None),
         (2020, "C", None, None),
         (2020, "D", None, None),
         (2020, "E", None, None),
         (2020, "F", None, None),
         ],
        ["year", "long_name", "nation_position", "team_position"],
    )
    assert popular_nation_team(input2, 2015) == {"nation": "A", "team": "C"}, "Error when the most popular nation/team is None"
    assert popular_nation_team(input2, 2016) == {"nation": "A", "team": "C"}, "Error when the most popular nation/team is None"
    assert popular_nation_team(input2, 2017) == {"nation": "A", "team": "C"}, "Error when the most popular nation/team is None"
    assert popular_nation_team(input2, 2018) == {"nation": "A", "team": "C"}, "Error when the most popular nation/team is None"
    assert popular_nation_team(input2, 2019) == {"nation": "A", "team": "C"}, "Error when the most popular nation/team is None"
    assert popular_nation_team(input2, 2020) == {"nation": "A", "team": "C"}, "Error when the most popular nation/team is None"
    assert popular_nation_team(input2, "all") == {"nation": "A", "team": "C"}, "Error when the most popular nation/team is None"

    input = spark_session.createDataFrame(
        [(2015, "A", "A", "A"),
         (2015, "B", "A", "B"),
         (2015, "C", "A", "B"),
         (2015, "D", "B", "C"),
         (2015, "E", "B", "C"),
         (2015, "F", "C", "C"),
         (2015, "A", None, None),
         (2015, "B", None, None),
         (2015, "C", None, None),
         (2015, "D", None, None),
         (2015, "E", None, None),
         (2015, "F", None, None),
         (2016, "A", "A", "A"),
         (2016, "B", "A", "B"),
         (2016, "C", "A", "B"),
         (2016, "D", "B", "C"),
         (2016, "E", "B", "C"),
         (2016, "F", "C", "C"),
         (2016, "A", None, None),
         (2016, "B", None, None),
         (2016, "C", None, None),
         (2016, "D", None, None),
         (2016, "E", None, None),
         (2016, "F", None, None),
         (2017, "A", "A", "A"),
         (2017, "B", "A", "B"),
         (2017, "C", "A", "B"),
         (2017, "D", "B", "C"),
         (2017, "E", "B", "C"),
         (2017, "F", "C", "C"),
         (2017, "A", None, None),
         (2017, "B", None, None),
         (2017, "C", None, None),
         (2017, "D", None, None),
         (2017, "E", None, None),
         (2017, "F", None, None),
         (2018, "A", "A", "A"),
         (2018, "B", "A", "B"),
         (2018, "C", "A", "B"),
         (2018, "D", "B", "C"),
         (2018, "E", "B", "C"),
         (2018, "F", "C", "C"),
         (2018, "A", None, None),
         (2018, "B", None, None),
         (2018, "C", None, None),
         (2018, "D", None, None),
         (2018, "E", None, None),
         (2018, "F", None, None),
         (2019, "A", "A", "A"),
         (2019, "B", "A", "B"),
         (2019, "C", "A", "B"),
         (2019, "D", "B", "C"),
         (2019, "E", "B", "C"),
         (2019, "F", "C", "C"),
         (2019, "A", None, None),
         (2019, "B", None, None),
         (2019, "C", None, None),
         (2019, "D", None, None),
         (2019, "E", None, None),
         (2019, "F", None, None),
         (2020, "A", "A", "A"),
         (2020, "B", "A", "B"),
         (2020, "C", "A", "B"),
         (2020, "D", "B", "C"),
         (2020, "E", "B", "C"),
         (2020, "F", "C", "C"),
         (2020, "A", None, None),
         (2020, "B", None, None),
         (2020, "C", None, None),
         (2020, "D", None, None),
         (2020, "E", None, None),
         (2020, "F", None, None),
         ],
        ["year", "long_name", "nation_position", "team_position"],
    )

    try:
        popular_nation_team(input, None)
        assert False
    except TypeError:
        assert True

    try:
        popular_nation_team(input, -1)
        assert False
    except TypeError:
        assert True

    try:
        popular_nation_team(input, 2023)
        assert False
    except TypeError:
        assert True

    try:
        popular_nation_team(input, 1.5)
        assert False
    except TypeError:
        assert True

    try:
        popular_nation_team(input, "abc")
        assert False
    except TypeError:
        assert True

    try:
        popular_nation_team(["abc"], 2015)
        assert False
    except TypeError:
        assert True

    try:
        popular_nation_team("abc", 2015)
        assert False
    except TypeError:
        assert True

    try:
        popular_nation_team(2, 2015)
        assert False
    except TypeError:
        assert True

# 5

def test_popular_nationality_happy(spark_session):
    input = spark_session.createDataFrame(
        [(2015, "A", "A"),
         (2015, "B", "A"),
         (2015, "C", "A"),
         (2015, "D", "B"),
         (2015, "E", "B"),
         (2015, "F", "C"),
         (2016, "A", "A"),
         (2016, "B", "A"),
         (2016, "C", "A"),
         (2016, "D", "B"),
         (2016, "E", "B"),
         (2016, "F", "C"),
         (2017, "A", "A"),
         (2017, "B", "A"),
         (2017, "C", "A"),
         (2017, "D", "B"),
         (2017, "E", "B"),
         (2017, "F", "C"),
         (2018, "A", "A"),
         (2018, "B", "A"),
         (2018, "C", "A"),
         (2018, "D", "B"),
         (2018, "E", "B"),
         (2018, "F", "C"),
         (2019, "A", "A"),
         (2019, "B", "A"),
         (2019, "C", "A"),
         (2019, "D", "B"),
         (2019, "E", "B"),
         (2019, "F", "C"),
         (2020, "A", "A"),
         (2020, "B", "A"),
         (2020, "C", "A"),
         (2020, "D", "B"),
         (2020, "E", "B"),
         (2020, "F", "C"),
         ],
        ["year", "long_name", "nationality"],
    )
    assert popular_nationality(input, 2015) is not None, "None returned for valid input"
    assert popular_nationality(input, 2015) == "A", "popular_nationality(input, 2015) wrong"
    assert popular_nationality(input, 2016) == "A", "popular_nationality(input, 2016) wrong"
    assert popular_nationality(input, 2017) == "A", "popular_nationality(input, 2017) wrong"
    assert popular_nationality(input, 2018) == "A", "popular_nationality(input, 2018) wrong"
    assert popular_nationality(input, 2019) == "A", "popular_nationality(input, 2019) wrong"
    assert popular_nationality(input, 2020) == "A", "popular_nationality(input, 2020) wrong"
    assert popular_nationality(input, "all") == "A", "popular_nationality(input, 'all') wrong"

def test_popular_nationality_sad(spark_session):
    input2 = spark_session.createDataFrame(
        [(2015, "A", "A"),
         (2015, "B", "A"),
         (2015, "C", "A"),
         (2015, "D", "B"),
         (2015, "E", "B"),
         (2015, "F", "C"),
         (2015, "A", None),
         (2015, "B", None),
         (2015, "C", None),
         (2015, "D", None),
         (2015, "E", None),
         (2015, "F", None),
         (2016, "A", "A"),
         (2016, "B", "A"),
         (2016, "C", "A"),
         (2016, "D", "B"),
         (2016, "E", "B"),
         (2016, "F", "C"),
         (2016, "A", None),
         (2016, "B", None),
         (2016, "C", None),
         (2016, "D", None),
         (2016, "E", None),
         (2016, "F", None),
         (2017, "A", "A"),
         (2017, "B", "A"),
         (2017, "C", "A"),
         (2017, "D", "B"),
         (2017, "E", "B"),
         (2017, "F", "C"),
         (2017, "A", None),
         (2017, "B", None),
         (2017, "C", None),
         (2017, "D", None),
         (2017, "E", None),
         (2017, "F", None),
         (2018, "A", "A"),
         (2018, "B", "A"),
         (2018, "C", "A"),
         (2018, "D", "B"),
         (2018, "E", "B"),
         (2018, "F", "C"),
         (2018, "A", None),
         (2018, "B", None),
         (2018, "C", None),
         (2018, "D", None),
         (2018, "E", None),
         (2018, "F", None),
         (2019, "A", "A"),
         (2019, "B", "A"),
         (2019, "C", "A"),
         (2019, "D", "B"),
         (2019, "E", "B"),
         (2019, "F", "C"),
         (2019, "A", None),
         (2019, "B", None),
         (2019, "C", None),
         (2019, "D", None),
         (2019, "E", None),
         (2019, "F", None),
         (2020, "A", "A"),
         (2020, "B", "A"),
         (2020, "C", "A"),
         (2020, "D", "B"),
         (2020, "E", "B"),
         (2020, "F", "C"),
         (2020, "A", None),
         (2020, "B", None),
         (2020, "C", None),
         (2020, "D", None),
         (2020, "E", None),
         (2020, "F", None),
         ],
        ["year", "long_name", "nationality"],
    )
    assert popular_nationality(input2, 2015) == "A", "Error when the most popular nationality is None"
    assert popular_nationality(input2, 2016) == "A", "Error when the most popular nationality is None"
    assert popular_nationality(input2, 2017) == "A", "Error when the most popular nationality is None"
    assert popular_nationality(input2, 2018) == "A", "Error when the most popular nationality is None"
    assert popular_nationality(input2, 2019) == "A", "Error when the most popular nationality is None"
    assert popular_nationality(input2, 2020) == "A", "Error when the most popular nationality is None"
    assert popular_nationality(input2, "all") == "A", "Error when the most popular nationality is None"

    input = spark_session.createDataFrame(
        [(2015, "A", "A", "A"),
         (2015, "B", "A", "B"),
         (2015, "C", "A", "B"),
         (2015, "D", "B", "C"),
         (2015, "E", "B", "C"),
         (2015, "F", "C", "C"),
         (2015, "A", None, None),
         (2015, "B", None, None),
         (2015, "C", None, None),
         (2015, "D", None, None),
         (2015, "E", None, None),
         (2015, "F", None, None),
         (2016, "A", "A", "A"),
         (2016, "B", "A", "B"),
         (2016, "C", "A", "B"),
         (2016, "D", "B", "C"),
         (2016, "E", "B", "C"),
         (2016, "F", "C", "C"),
         (2016, "A", None, None),
         (2016, "B", None, None),
         (2016, "C", None, None),
         (2016, "D", None, None),
         (2016, "E", None, None),
         (2016, "F", None, None),
         (2017, "A", "A", "A"),
         (2017, "B", "A", "B"),
         (2017, "C", "A", "B"),
         (2017, "D", "B", "C"),
         (2017, "E", "B", "C"),
         (2017, "F", "C", "C"),
         (2017, "A", None, None),
         (2017, "B", None, None),
         (2017, "C", None, None),
         (2017, "D", None, None),
         (2017, "E", None, None),
         (2017, "F", None, None),
         (2018, "A", "A", "A"),
         (2018, "B", "A", "B"),
         (2018, "C", "A", "B"),
         (2018, "D", "B", "C"),
         (2018, "E", "B", "C"),
         (2018, "F", "C", "C"),
         (2018, "A", None, None),
         (2018, "B", None, None),
         (2018, "C", None, None),
         (2018, "D", None, None),
         (2018, "E", None, None),
         (2018, "F", None, None),
         (2019, "A", "A", "A"),
         (2019, "B", "A", "B"),
         (2019, "C", "A", "B"),
         (2019, "D", "B", "C"),
         (2019, "E", "B", "C"),
         (2019, "F", "C", "C"),
         (2019, "A", None, None),
         (2019, "B", None, None),
         (2019, "C", None, None),
         (2019, "D", None, None),
         (2019, "E", None, None),
         (2019, "F", None, None),
         (2020, "A", "A", "A"),
         (2020, "B", "A", "B"),
         (2020, "C", "A", "B"),
         (2020, "D", "B", "C"),
         (2020, "E", "B", "C"),
         (2020, "F", "C", "C"),
         (2020, "A", None, None),
         (2020, "B", None, None),
         (2020, "C", None, None),
         (2020, "D", None, None),
         (2020, "E", None, None),
         (2020, "F", None, None),
         ],
        ["year", "long_name", "nation_position", "team_position"],
    )

    try:
        popular_nationality(input, None)
        assert False
    except TypeError:
        assert True

    try:
        popular_nationality(input, -1)
        assert False
    except TypeError:
        assert True

    try:
        popular_nationality(input, 2023)
        assert False
    except TypeError:
        assert True

    try:
        popular_nationality(input, 1.5)
        assert False
    except TypeError:
        assert True

    try:
        popular_nationality(input, "abc")
        assert False
    except TypeError:
        assert True

    try:
        popular_nationality(["abc"], 2015)
        assert False
    except TypeError:
        assert True

    try:
        popular_nationality("abc", 2015)
        assert False
    except TypeError:
        assert True

    try:
        popular_nationality(2, 2015)
        assert False
    except TypeError:
        assert True








