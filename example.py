import pandas as pd
# /!\ The follwing two command lines only work if this file is in the same directory
# as utils.py and modelEoles.py !!!
from utils import get_config
from modelEoles import ModelEOLES

example = ModelEOLES(name="example_2006_norsv", config=get_config(path="config/config_2006.json"), output_path="outputs", include_reserve=False)

example.build_model()
example.solve(solver_name="gurobi")
example.extract_optimisation_results()

example.summary.to_csv("outputs/summary.csv")
example.results.to_csv("outputs/results.csv")
example.hourly_balance.to_csv("outputs/hourly_generation.csv")
