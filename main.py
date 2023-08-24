import base64
import dash
from dash import html
import pandas as pd
from dash import dcc
import plotly.express as px
from flask import Flask
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# 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
# load data
df = pd.read_csv('/Users/vantagecircle/Downloads/cleaned_steps.csv')
#df_temp= pd.read_csv('/Users/vantagecircle/Downloads/POWER_Point_Daily_20160101_20201231_026d1365N_091d7991E_LST - POWER_Point_Daily_20160101_20201231_026d1365N_091d7991E_LST.csv')

# pandas operation
df = df[df['steps_count']>=0]
df = df[df['user_name'].str.len() <= 20]
df['steps_at'] = pd.to_datetime(df['steps_at'])
df['year'] = df['steps_at'].dt.year
df['month'] = df['steps_at'].dt.month
df['day'] = df['steps_at'].dt.day
season_mapping = {1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring', 6: 'summer', 7: 'summer', 8: 'summer',
                  9: 'autumn', 10: 'autumn', 11: 'autumn', 12: 'winter'}
df['season'] = df['month'].map(season_mapping)
user_join= pd.DataFrame(columns=['Year', 'Users Joined'])
years = df['year'].unique()
for year in years:
    users_joined = df[df['year'] == year]['user_name'].nunique()
    user_join = pd.concat([user_join, pd.DataFrame({'Year': [year], 'Users Joined': [users_joined]})], ignore_index=True)

# Sort the data by user_id and steps_at
df.sort_values(['user_id', 'steps_at'], inplace=True)
# Group the data by user_id and count the number of unique days using the dt accessor
unique_days_per_user = df.groupby(['user_id', 'user_name']).apply(lambda x: x['steps_at'].dt.date.nunique())
# Reset the index to obtain a DataFrame
unique_days_per_user = unique_days_per_user.reset_index()
# Rename the column
unique_days_per_user.columns = ['user_id', 'user_name', 'unique_days']
unique_days_per_user=unique_days_per_user.sort_values(by='unique_days',ascending=False).head(10)
# Calculate churn rate
# Calculate yearly steps count for each user
user_yearly_steps = df.groupby(['user_id', 'year'])['steps_count'].sum().reset_index()
# Define the thresholds
threshold = 1000000
threshold_16 = 400000

# Create a custom function to set the threshold based on the year
def get_threshold(year):
    return threshold_16 if year == 2016 else threshold

# Apply the custom function to create a new 'threshold_value' column
user_yearly_steps['threshold_value'] = user_yearly_steps['year'].apply(get_threshold)

# Filter the churned users based on the threshold_value column
churned_users = user_yearly_steps[user_yearly_steps['steps_count'] < user_yearly_steps['threshold_value']]

# Calculate the total number of users in each year
total_users = user_yearly_steps.groupby('year')['user_id'].nunique().reset_index()

# Calculate the number of churned users in each year
churned_user_count = churned_users.groupby('year')['user_id'].nunique().reset_index()

# Merge total users and churned users count DataFrames
churn_data = pd.merge(total_users, churned_user_count, on='year', how='left')

# Calculate churn rate for each year
churn_data['churn_rate'] = (churn_data['user_id_y'] / churn_data['user_id_x']) * 100
churn_data['churn_rate'] = churn_data['churn_rate'].round(2)

# Drop the 'threshold_value' column as it is no longer needed
user_yearly_steps.drop(columns=['threshold_value'], inplace=True)

churn_data = churn_data.head(4)

# temperature and precipitation analysis
# drop columns
#df_temp.drop(columns='TS',inplace=True)
# Merge the datasets based on date columns
#merged_df = pd.merge(df, df_temp, left_on=['year', 'month', 'day'], right_on=['YEAR', 'MO', 'DY'], how='left')
# Add the 'temperature' column from Dataset 1 to Dataset 2
#merged_df['temperature'] = merged_df['T2M_MAX']
# Drop the unnecessary columns from Dataset 1
#merged_df.drop(['YEAR', 'MO', 'DY', 'T2M_MAX'], axis=1, inplace=True)

sorted_years = sorted(df['year'].unique())
options = [{'label': str(year), 'value': year} for year in sorted_years]
default_value = sorted_years[0]
year_options=[{'label': str(year), 'value': year} for year in sorted_years]
monthly_averages = df.groupby(['year', 'month'])['steps_count'].mean().reset_index()

season_steps = df.groupby('season')['steps_count'].sum().reset_index()
season_order = ['winter', 'spring', 'summer', 'autumn']
season_steps['season'] = pd.Categorical(season_steps['season'], categories=season_order, ordered=True)
season_steps = season_steps.sort_values('season')

# Calculate the total steps count across all seasons
total_steps_count = season_steps['steps_count'].sum()

# Calculate the percentage of each season rounded to two decimal points
season_steps['percentage'] = ((season_steps['steps_count'] / total_steps_count) * 100).round(2)

# Create a subplot with 2 rows and 1 column
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Total steps across seasons.", "Percentage distribution across seasons."))

# Add the line chart to the first row of the subplot
fig.add_trace(
    go.Scatter(x=season_steps['season'], y=season_steps['steps_count'], mode='lines+markers', name='Steps Count'),
    row=1, col=1)

# Add the bar chart to the second row of the subplot
fig.add_trace(go.Bar(x=season_steps['season'], y=season_steps['percentage'],
                     text=season_steps['percentage'].apply(lambda x: f"{x:.2f}%"),
                     textposition='auto', name='Percentage'),
              row=2, col=1)

# Update layout properties
fig.update_layout(title_text="Seasons Steps Count with Percentage",
                  showlegend=False, height=600)

# Update axis labels
fig.update_yaxes(title_text="Steps Count", row=1, col=1)
fig.update_yaxes(title_text="Percentage", row=2, col=1)

# Custom CSS style for rounded cards
rounded_card_style = {
    'borderRadius': '15px',
    'boxShadow': '0 4px 6px 0 rgba(0, 0, 0, 0.5)',
    'marginBottom': '20px',
    'padding': '20px'
}
info_button='https://res.cloudinary.com/vantagecircle/image/upload/v1661750161/dashboard/icon4.png'
# create the dash app
server = Flask(__name__)
app = dash.Dash(__name__, server = server, external_stylesheets=[dbc.themes.UNITED, dbc.icons.BOOTSTRAP])

image_path = '/Users/vantagecircle/Downloads/logo.png'
#check if the image exists
if not os.path.isfile(image_path):
    raise FileNotFoundError('Image file not found.')
# read the image file and encode it to base64
with open(image_path,"rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()

# App Layout
app.layout=html.Div(
    style={'padding':'50px'},
    children=[
        dbc.Row(
            [
                html.Img(
                        src='data:image/jpeg;base64,{}'.format(encoded_image),
                        style={'height': '100px','width': '500px','margin':'0 auto'})
            ]
        ),
        html.Br(),
        dbc.Card(
            dbc.CardBody([
                dbc.Row([dbc.Col([
                    html.H1(
                        children='Steps Count Dashboard',
                        style={'color': 'black', 'textAlign': 'center', 'fontSize': '60px', 'padding': '10px',
                               'fontWeight': 'bold'}
                    )],width=11,align='center'),
                    dbc.Col([
                        html.A(
                            dbc.Button(
                                html.Img(src=info_button),
                                style={'backgroundColor': 'white', 'border': 'none', 'verticalAlign': 'middle',
                                       'marginLeft': '10px'}
                            ),
                            href="#",
                            id="info-button",
                        ),
                        dbc.Tooltip(
                            "This dashboard is just for practice",
                            target="info-button",
                            placement="right"
                        )
                    ])
                ])
            ]),
            style=rounded_card_style
        ),
        html.Br(),

dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([html.H2(children='Step count distribution across seasons.',style={'padding':'10px','color':'black','textAlign':'left','fontWeight':'bold'})],width=3,align='left'),
            dbc.Card(dbc.CardBody([
            dbc.Col([dcc.Graph(id='combined-plot',figure=fig, config={'displayModeBar': False})])]),style=rounded_card_style)
        ])
        ]),style=rounded_card_style),
        html.Br(),
        dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col(
                [
                dbc.Row(html.H2(children='User steps count.',style={'padding':'10px','color':'black','textAlign':'left','fontWeight':'bold'})),
                dbc.Row(html.P(children='Select User:',style={'textAlign':'left'})),
                dcc.Dropdown(id='user',options=[{'label': i, 'value': i} for i in df['user_name'].unique()],value='Somir Saikia',style={'backgroundColor':'lightgray'}),
                html.Br(),
                dbc.Row(html.P(children='Select Year:',style={'textAlign':'left'})),
                dcc.Dropdown(id='year',options=options,value=default_value,style={'backgroundColor':'lightgray'}),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                dbc.Row(html.P(id='user-month-stats', style={'textAlign': 'left'}))
                ],width=3,align='left'
            ),
            dbc.Col(
                [
                    dbc.Row([dbc.Card(dbc.CardBody([dcc.Graph(id='bar-plot', config={'displayModeBar': False})]))],style=rounded_card_style),
                    html.Br(),
                    dbc.Row([dbc.Card(dbc.CardBody([dcc.Graph(id='step-counts', config={'displayModeBar': False})]))],style=rounded_card_style),
                    html.Br(),
                    dbc.Row(
                        [
                            dbc.Card(dbc.CardBody([
                            dbc.Col(dcc.Graph(id='line-chart', config={'displayModeBar': False}))]),style=rounded_card_style),
                            dbc.Card(dbc.CardBody([
                            dbc.Col(dcc.Graph(id='pie-chart', config={'displayModeBar': False}))]),style=rounded_card_style)
                        ]
                    )
                ]
            )
        ])
            ]),style=rounded_card_style),
                    html.Br(),
                    html.Br(),
        dbc.Card(dbc.CardBody([
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row(html.H2(children='Active users per year using Vantage Fit app.',style={'padding':'10px','color':'black','textAlign':'left','fontWeight':'bold'})),
                        dcc.Dropdown(id='year-dropdown',options=year_options,value=default_value,style={'backgroundColor':'lightgray'}),
                        html.Br(),
                        dbc.Row(html.P(id='unique-users-per-year', style={'textAlign': 'left'}))
                    ],width=3,align='left'
                ),
                dbc.Col(
                    [
                        dbc.Card(dbc.CardBody([
                        dcc.Graph(id='user-join-trend-chart', config={'displayModeBar': False})]),style=rounded_card_style)
                    ]
                )
            ]
        )]),style=rounded_card_style)

    ]
)

# Callbacks
@app.callback(
    Output('step-counts', 'figure'),
    [Input('user', 'value'),
     Input('year', 'value')]
)
def update_step_counts(selected_user, selected_year):
    filtered_df = df[(df['user_name'] == selected_user) & (df['year'] == selected_year)]
    monthly_steps = filtered_df.groupby('month')['steps_count'].sum().reset_index()
    monthly_steps.rename(columns={'month':'Month','steps_count':'Steps count'},inplace=True)
    fig = px.bar(monthly_steps, x='Month', y='Steps count',
                 title=f'Monthly step counts of user: {selected_user}, Year: {selected_year}')
    return fig

@app.callback(
    Output('bar-plot', 'figure'),
    Input('year', 'value')
)
def update_bar_plot(selected_year):
    # Filter the DataFrame based on the selected year
    filtered_df = df[df['year'] == selected_year]

    # Calculate the average steps for each month
    monthly_average_steps = filtered_df.groupby('month')['steps_count'].sum().reset_index()

    # Create the bar plot using Plotly Express
    fig = px.bar(
        monthly_average_steps, x='month', y='steps_count',
        title=f"Overall steps count for the year: {selected_year}",
        labels={'month': 'Month', 'steps_count': 'Average Steps Count'}
    )

    return fig

@app.callback(
    Output('user-join-trend-chart','figure'),
    Input('year-dropdown','value')
)
def update_user_join_trend(year_selected):
    # Filter the DataFrame for the selected year
    filtered_df = df[df['year'] == year_selected]
    # Get the first appearance date of each unique user
    first_appearance_dates = filtered_df.groupby('user_name')['steps_at'].min()
    # Extract month and year from the first appearance dates and create a new 'month_year' column
    first_appearance_dates = first_appearance_dates.reset_index()
    first_appearance_dates['month_year'] = first_appearance_dates['steps_at'].dt.strftime('%B %Y')
    # Group by 'month_year' and count the number of unique users in each month
    user_join_trend = first_appearance_dates.groupby('month_year')['user_name'].nunique().reset_index()
    user_join_trend.rename(columns={'user_name': 'Unique Users'}, inplace=True)
    # Convert the 'month_year' column to a pandas Categorical data type with the correct sorting order
    user_join_trend['month_year'] = pd.to_datetime(user_join_trend['month_year'], format='%B %Y')
    user_join_trend.sort_values('month_year', inplace=True)
    # Create the line chart
    fig = px.line(user_join_trend, x='month_year', y='Unique Users', title=f'User Trend in {year_selected}', text='Unique Users')
    fig.update_layout(xaxis_title='Month', yaxis_title='Number of Unique Users',paper_bgcolor='white', plot_bgcolor='white')
    fig.update_traces(
        line=dict(color='orange'),  # Set line color to orange
        fill='tonexty',  # Add gradient below the line
        fillcolor='rgba(255, 165, 0, 0.2)'  # Set the gradient color
    )

    return fig

@app.callback(
    Output('unique-users-per-year', 'children'),
    Input('year-dropdown', 'value')
)
def update_unique_users_per_year(year_selected):
    # Filter the DataFrame for the selected year
    filtered_df = df[df['year'] == year_selected]

    # Count the number of unique users for the selected year
    unique_users_count = filtered_df['user_id'].nunique()

    return f"Number of Unique Users using the app in {year_selected}: {unique_users_count}"

@app.callback(
    [Output('line-chart', 'figure'),
     Output('pie-chart', 'figure')],
    [Input('user', 'value')]
)
def update_line_pie(selected_user):
    selected_data = df[df['user_name'] == selected_user]
    season_steps = selected_data.groupby('season')['steps_count'].sum().reset_index()
    season_order = ['winter', 'spring', 'summer', 'autumn']
    season_steps['season'] = pd.Categorical(season_steps['season'], categories=season_order, ordered=True)
    season_steps = season_steps.sort_values('season')
    fig_line = px.line(season_steps, x='season', y='steps_count',
                       title=f"Steps count of {selected_user} across seasons.")

    fig_pie = px.pie(season_steps, values='steps_count', names='season',
                     title=f"Steps count distribution for {selected_user}",
                     color_discrete_sequence=px.colors.qualitative.Set3
                     )

    # Customize 3D pie chart appearance
    fig_pie.update_traces(
        textinfo='percent+label',  # Display both percentage and label
        hoverinfo='label+percent+value',
        marker=dict(line=dict(color='white', width=2))  # Add a border to slices
    )


    return fig_line, fig_pie

@app.callback(
    Output('user-month-stats', 'children'),
    [Input('user', 'value'),
     Input('year', 'value')]
)
def update_user_month_stats(selected_user, selected_year):
    if selected_user is not None and selected_year is not None:
        selected_data = df[(df['user_name'] == selected_user) & (df['year'] == selected_year)]

        if not selected_data.empty:
            monthly_steps = selected_data.groupby('month')['steps_count'].sum().reset_index()
            month_with_max_steps = monthly_steps.loc[monthly_steps['steps_count'].idxmax(), 'month']
            return f"{selected_user} has walked the most in the {month_with_max_steps} month of {selected_year}."
        else:
            return f"No data available for {selected_user} in the year {selected_year}."
    return ""

def calculate_top_user_per_month(data):
    top_users = data.groupby(['year', 'month', 'user_name'])['steps_count'].sum().reset_index()
    top_users = top_users.loc[top_users.groupby(['year', 'month'])['steps_count'].idxmax()]
    return top_users

@app.callback(
    Output('bar-chart', 'figure'),
    Input('year1', 'value')
)
def update_table(selected_year):
    filtered_df = df[df['year'] == selected_year]
    top_users_per_month = calculate_top_user_per_month(filtered_df)
    top_users_per_month.rename(columns={'month':'Month','steps_count':'Steps count','user_name':'User name'},inplace=True)
    bar_chart = px.bar(top_users_per_month, x='Month', y='Steps count', color='User name',
                       title=f'Monthly step count of top user for the year {selected_year}')

    return bar_chart



@app.callback(
    Output('user-average', 'children'),
    Input('year2', 'value')
)
def update_user_with_highest_average_steps(year_selected):
    # Filter the DataFrame for the selected year
    filtered_df = df[df['year'] == year_selected]

    if filtered_df.empty:
        return f"No data available for the year {year_selected}"

    # Calculate the average steps for each user
    average_steps = filtered_df.groupby('user_name')['steps_count'].mean().reset_index()

    if average_steps.empty:
        return f"No user data available for the year {year_selected}"

    # Get the user with the highest average steps
    highest_user = average_steps.loc[average_steps['steps_count'].idxmax(), 'user_name']

    return f"User with Highest Average Steps in {year_selected}: {highest_user}"



# Run server
if __name__ == '__main__':
    app.run_server(debug=True,host='0.0.0.0',port=3000,threaded=False)