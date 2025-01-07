import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import os
import time
from main import process_data, run_algorithm, plot_solution  # Assuming these functions are defined in main.py
import base64
import io
import matplotlib.pyplot as plt
from PIL import Image
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Delivery Optimization Dashboard"),
    
    dcc.Tabs([
        dcc.Tab(label='Run Entire Dataset', children=[
            html.Div([
                html.Label('Population Size'),
                dcc.Input(id='population-size-all', type='number', value=100),
                html.Label('Number of Generations'),
                dcc.Input(id='num-generations-all', type='number', value=2000),
                html.Label('Mutation Rate'),
                dcc.Input(id='mutation-rate-all', type='number', value=0.3, step=0.01),
                html.Label('Elite Size'),
                dcc.Input(id='elite-size-all', type='number', value=10),
                html.Button('Run All', id='run-all-button', n_clicks=0)
            ]),
            dcc.Loading(
                id="loading-1",
                type="default",
                children=[
                    dcc.Graph(id='all-dataset-graph'),
                    dcc.Graph(id='deviation-graph')
                ]
            ),
            html.Div(id='comparison-results')
        ]),
        dcc.Tab(label='Run Single File', children=[
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Drag and Drop or ', html.A('Select a File')]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False
            ),
            html.Div(id='output-file-upload'),
            html.Div([
                html.Label('Population Size'),
                dcc.Input(id='population-size', type='number', value=100),
                html.Label('Number of Generations'),
                dcc.Input(id='num-generations', type='number', value=2000),
                html.Label('Mutation Rate'),
                dcc.Input(id='mutation-rate', type='number', value=0.3, step=0.01),
                html.Label('Elite Size'),
                dcc.Input(id='elite-size', type='number', value=10),
                html.Button('Run Algorithm', id='run-algorithm-button', n_clicks=0)
            ]),
            dcc.Loading(
                id="loading-2",
                type="default",
                children=dcc.Graph(id='single-file-graph')
            ),
            html.Div(id='output-details')
        ])
    ])
])

def read_solution_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "Total cost" in line:
                total_cost = float(line.split(':')[-1].strip().replace('*/', ''))
                return total_cost
    return None

@app.callback(
    Output('all-dataset-graph', 'figure'),
    Output('deviation-graph', 'figure'),
    Output('run-all-button', 'disabled'),
    Output('comparison-results', 'children'),
    Input('run-all-button', 'n_clicks'),
    State('population-size-all', 'value'),
    State('num-generations-all', 'value'),
    State('mutation-rate-all', 'value'),
    State('elite-size-all', 'value')
)
def run_all_datasets(n_clicks, population_size, num_generations, mutation_rate, elite_size):
    if n_clicks > 0:
        datasets = [f for f in os.listdir('data') if f.endswith('.txt')]
        results = []
        deviations = []
        comparisons = []
        for dataset in datasets:
            with open(os.path.join('data', dataset), 'r') as file:
                data_str = file.read()
            data = process_data(data_str)
            best_solution, best_time, execution_time = run_algorithm(data, population_size, num_generations, mutation_rate, elite_size)
            results.append({'dataset': dataset, 'num_points': data['num_nodes'], 'runtime': execution_time})
            
            # Compare with provided solution
            solution_file = os.path.join('check_solution', dataset)
            if os.path.exists(solution_file):
                provided_cost = read_solution_file(solution_file)
                deviation = ((best_time - provided_cost) / provided_cost) * 100
                deviations.append({'num_points': data['num_nodes'], 'deviation': deviation})
                comparisons.append(html.P(f"{dataset}: Deviation = {deviation:.2f}%"))

        df = pd.DataFrame(results)
        avg_runtime = df.groupby('num_points')['runtime'].mean().reset_index()
        runtime_figure = {
            'data': [go.Bar(x=avg_runtime['num_points'], y=avg_runtime['runtime'])],
            'layout': go.Layout(title='Average Runtime vs Number of Delivery Points', xaxis={'title': 'Number of Points'}, yaxis={'title': 'Average Runtime (s)'})
        }

        df_deviation = pd.DataFrame(deviations)
        print("Deviations DataFrame:", df_deviation)  # Debug statement
        if not df_deviation.empty:
            avg_deviation = df_deviation.groupby('num_points')['deviation'].mean().reset_index()
            deviation_figure = {
                'data': [go.Bar(x=avg_deviation['num_points'], y=avg_deviation['deviation'])],
                'layout': go.Layout(title='Average Deviation vs Number of Delivery Points', xaxis={'title': 'Number of Points'}, yaxis={'title': 'Average Deviation (%)'})
            }
        else:
            deviation_figure = {}

        return runtime_figure, deviation_figure, False, comparisons
    return {}, {}, False, []

@app.callback(
    Output('output-file-upload', 'children'),
    Output('single-file-graph', 'figure'),
    Output('output-details', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('population-size', 'value'),
    State('num-generations', 'value'),
    State('mutation-rate', 'value'),
    State('elite-size', 'value'),
    Input('run-algorithm-button', 'n_clicks')
)
def run_single_file(contents, filename, population_size, num_generations, mutation_rate, elite_size, n_clicks):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        data_str = decoded.decode('utf-8')
        data = process_data(data_str)
        best_solution, best_time, execution_time = run_algorithm(data, population_size, num_generations, mutation_rate, elite_size)
        
        figure = plot_solution(data['coordinates'], best_solution)
        
        details = [
            html.P(f"Thời gian chạy: {execution_time:.2f} giây"),
            html.P(f"Best Solution: {best_solution}"),
            html.P(f"Best Delivery Time: {best_time:.2f}")
        ]
        
        return f'File {filename} processed in {execution_time:.2f} seconds', figure, details
    return '', {}, ''

if __name__ == '__main__':
    app.run_server(debug=True)