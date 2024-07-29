import pandas as pd
from utils import get_config
from modelEoles import ModelEOLES

example = ModelEOLES(name="example_2006_norsv", config=get_config(path="config/config_2006.json"), output_path="outputs", include_reserve=False)

eval.build_model()
eval.solve(solver_name="gurobi")
eval.extract_optimisation_results()

eval.summary.to_csv("outputs/example_2006_norsv/summary.csv")
eval.results.to_csv("outputs/example_2006_norsv/results.csv")
eval.hourly_generation.to_csv("outputs/example_2006_norsv/hourly_generation.csv")
