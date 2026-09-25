from shiny import App, reactive, render, ui
import pandas as pd
import pulp
import sys
import traceback

from data_processing import SUPPORTED_PARAMETERS
from optimizer import OptimizationConfig, solve_bmp_placement_from_coefficients

rows = []
for bmp, cost, value in [("Dry Pond", 1000, 100.0), ("Wet Pond", 100.0, 40.0)]:
    row = {
        "Subbasin": "1",
        "BMP": bmp,
        "Cost coefficient": cost,
        "Decision upper limit": 1.0,
        "Max BMP units": 1.0,
    }
    for p in SUPPORTED_PARAMETERS:
        row[p] = value
    rows.append(row)

coefficients = pd.DataFrame(rows)
targets = pd.DataFrame(
    [{"Parameter": p, "Internal Target": 20.0} for p in SUPPORTED_PARAMETERS]
)

#Shiny interaface
app_ui = ui.page_fluid(
    ui.h3("Test"),
    ui.input_action_button("run", "Run optimization"),
    ui.output_text_verbatim("result"),
)

def server(input, output, session):
    text = reactive.value("Ready")

    @reactive.effect
    @reactive.event(input.run)
    def _():
        try:
            print("1 App button clicked", flush=True)
            print("2 Python version executed:", sys.executable, flush=True)
            print("3 PuLP version:", pulp.__version__, flush=True)
            print("4 Solvers available:", pulp.listSolvers(onlyAvailable=True), flush=True)
            print("5 Calling optimizer", flush=True)

            solved = solve_bmp_placement_from_coefficients(
                coefficients,
                targets,
                OptimizationConfig(solver_time_limit_sec=120),
            )

            print("6 Optimizer returned", flush=True)
            print("7 Status:", solved["status"], flush=True)
            print("8 Objective:", solved["objective_value"], flush=True)

            text.set(
                f"Status: {solved['status']}\n"
                f"Objective: {solved['objective_value']}\n\n"
                f"{solved['placement'].to_string(index=False)}"
            )
        except Exception as e:
            print("ERROR:", repr(e), flush=True)
            traceback.print_exc()
            text.set(f"ERROR: {e}")

    @render.text
    def result():
        return text.get()

app = App(app_ui, server)
