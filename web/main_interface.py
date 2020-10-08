import base64

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import dash_table
import Db
from dash.dependencies import Input, Output


db = Db.Db()






def generate_table(df,):
    return dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in df.columns],
    data=df.to_dict('records'),
    style_cell={'color': '#0a0806',
                'background': '#e6e6e6',
                'font-family': 'Work Sans",sans-serif',
                'minWidth': 95, 'maxWidth': 200, 'width': 95},
    style_header={
            'font-family': 'Vitesse,sans-serif',
            'fontWeight': 'bold',
            'color': '#0a0806',
            'font-size': 18},
    fixed_rows={'headers': True},
    style_table={'height': 2000,
                 'overflowX': 'auto'},

)

def generate_table_trade(df,):
    return dash_table.DataTable(
    id='table_trade',
    columns=[{"name": i, "id": i} for i in df.columns],
    data=df.to_dict('records'),
    style_cell={'color': '#0a0806',
                'background': '#e6e6e6',
                'font-family': 'Work Sans",sans-serif',
                'minWidth': 95, 'maxWidth': 200, 'width': 95},
    style_header={
            'font-family': 'Vitesse,sans-serif',
            'fontWeight': 'bold',
            'color': '#0a0806',
            'font-size': 18},
    style_data={
        'whiteSpace': 'normal',
        'height': 'auto',
    },
    # style_table={'height': 10000,
    #              'verflowX': 'auto'},
    # fixed_rows={'headers': True},
    #editable=True,

    style_data_conditional=[
        {
            'if': {
                'filter_query': '{profit} > 0',  # matching rows of a hidden column with the id, `id`
                'column_id': 'profit'
            },
            #'backgroundColor': '#3D9970',
            'color': '#3D9970',
            'fontWeight': 'bold'
        },
        {
            'if': {
                'filter_query': '{profit} < 0',  # matching rows of a hidden column with the id, `id`
                'column_id': 'profit'
            },
            #'backgroundColor': '#3D9970',
            'color': '#85144b',
            'fontWeight': 'bold'
        },]


    )


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Keep Predict'




def serve_layout():
    df = db.get_market_data()
    predict_df = db.get_predict()
    trade_df = db.get_trades()
    trade_df_buy = trade_df[trade_df['action'] == 'BUY']
    trade_df_sell = trade_df[trade_df['action'] == 'SELL']
    trade_df['current_balance'] = trade_df['net_worth']
    trade_df = trade_df.drop(['net_worth'], axis=1)
    encoded_image = base64.b64encode(open('picture.png', 'rb').read()).decode('ascii')
    return html.Div(style={'backgroundColor': '#e6e6e6'}, children=[
        html.Img(src='data:image/png;base64,{}'.format(encoded_image), style={'height':'4%', 'width':'4%'}),
    html.H1(
        children='Keep Predict',
        style={
            'textAlign': 'center',
            'color': '#0a0806',
            'font-family': 'Vitesse,sans-serif',
            'font-weight': '900'
        }
        ),
        html.H1(
            children='Disclaimer: for entertainment only, do not use for trading decisions.',
            style={
                'textAlign': 'center',
                'color': '#0a0806',
                'font-family': 'Vitesse,sans-serif',
                'font-weight': '20',
                'font-size': 15,
            }
        ),
        dcc.Link('Link to Github', href='https://github.com/airocket/keep_rl', target="_blank", style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'color': '#0a0806'}),


    dcc.Graph(
        id='example-graph-2',
        figure={
            'data': [
                {'x': df['index'].tolist(), 'y': df['close_keep'].tolist(), 'type': 'line', 'name': 'Keep Price', 'line':dict(color='#48dbb4')},
                {'x': predict_df['time'].tolist(), 'y': predict_df['lstm_predict'].tolist(), 'type': 'line', 'name': 'LSTM Predict', 'line':dict(color='#FF34CE'), 'mode': 'line'},
            ],
            'layout': {
                'plot_bgcolor': '#e6e6e6',
                'paper_bgcolor': '#e6e6e6',
                'yaxis': dict(title='Keep Price'),

            }

        }
    ), html.H1(
        children='Keep Predict Trade',
        style={
            'textAlign': 'center',
            'color': '#0a0806',
            'font-family': 'Vitesse,sans-serif',
            'font-weight': '900'
        }
    ),html.H1(
            children='Disclaimer: for entertainment only, do not use for trading decisions.',
            style={
                'textAlign': 'center',
                'color': '#0a0806',
                'font-family': 'Vitesse,sans-serif',
                'font-weight': '20',
                'font-size': 15,
            }
        ), html.Div([dcc.Link('Learning history', href='http://www.keeppredict.com:6006/', target="_blank", style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'color': '#0a0806'}),
            dcc.Link('Learning graphs', href='http://www.keeppredict.com:6006/#graphs', target="_blank", style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'color': '#0a0806'}),
        dcc.Link('Link to Github', href='https://github.com/airocket/keep_rl', target="_blank",
                 style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center',
                        'color': '#0a0806'})]),

        dcc.Graph(
            id='example-graph-3',
            figure={
                'data': [
                    {'x': df['index'].tolist(), 'y': df['close_keep'].tolist(), 'type': 'line', 'name': 'Keep Price',
                     'line': dict(color='#48dbb4'), 'yaxis': 'y1'},
                    {'x': trade_df_sell['time'].tolist(), 'y': trade_df_sell['net_worth'].tolist(),
                     'type': 'line', 'name': 'Current Balance', 'yaxis': 'y2', 'line': dict(color='#ffa500')},
                    {'x': trade_df_buy['time'].tolist(), 'y': trade_df_buy['keep_price'].tolist(), 'type': 'scattergl',
                     'name': 'Buy', 'mode': "markers", 'marker': dict(color='#002FFF'), 'yaxis': 'y1'},
                    {'x': trade_df_sell['time'].tolist(), 'y': trade_df_sell['keep_price'].tolist(),
                     'type': 'scattergl', 'name': 'Sell', 'mode': "markers", 'marker': dict(color='#FF0004'), 'yaxis': 'y1'},
                ],
                'layout': {
                    'plot_bgcolor': '#e6e6e6',
                    'paper_bgcolor': '#e6e6e6',
                    'yaxis': dict(title='Keep Price'),
                    'yaxis2': dict(title='Net Worth',
                                        overlaying='y',
                                        side='right')

                }

            }
        ), html.H1(
            children='Trade History',
            style={
                'textAlign': 'center',
                'color': '#0a0806',
                'font-family': 'Vitesse,sans-serif',
                'font-weight': '900',
                #'font-size': 25,
            }
        ),html.H1(
            children='(Model start balance 100000)',
            style={
                'textAlign': 'center',
                'color': '#0a0806',
                'font-family': 'Vitesse,sans-serif',
                'font-weight': '20',
                'font-size': 15,
            }),
        generate_table_trade(trade_df)
    , html.H1(
            children='Training data',
            style={
                'textAlign': 'center',
                'color': '#0a0806',
                'font-family': 'Vitesse,sans-serif',
                'font-weight': '900',
                #'font-size': 25,
            }
        ),
        generate_table(df)
])

app.layout = serve_layout




if __name__ == '__main__':
    import logging
    file_log = logging.FileHandler('web_log.log')
    console_out = logging.StreamHandler()
    logging.basicConfig(handlers=(file_log, console_out),level=logging.INFO)
    app.run_server(debug=False, port=800, host='localhost')