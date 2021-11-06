import plotly.graph_objects as go # or plotly.express as px
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans, DBSCAN
from plotly.subplots import make_subplots
from sklearn.neighbors import NearestNeighbors
import numpy as np

df = pd.read_csv('nrel_df.csv')

def scale_data(df,features=None):
    if features is None:
        features = [c for c in df.columns if c != 'latitude' and 
                    c != 'longitude' and 'cluster' not in c]
    x = df[features].values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_scaled = pd.DataFrame(x_scaled)
    return df_scaled

def cluster_data_kmeans(df,n_clusters=8,features=None):
    df_scaled = scale_data(df,features)
    clusters = KMeans(n_clusters=n_clusters, random_state=0).fit(df_scaled)
    df['kmeans_cluster'] = clusters.labels_
    return df,clusters.inertia_

def cluster_data_dbscan(df_scaled,eps=0.1,min_pts=15,features=None):
    df_scaled = scale_data(df,features)
    clusters = DBSCAN(eps=eps,min_samples=min_pts).fit(df_scaled)
    df['dbscan_cluster'] = clusters.labels_
    return df

def curvature(a):
    b = [0]*len(a)
    for i in range(len(b)):
        if 0<i<len(a)-1:
            b[i] = (a[i+1]-2*a[i]+a[i-1])
        elif i==0:
            b[i] = (a[i+1]-2*a[i])
        elif i==len(a)-1:
            b[i] = (-2*a[i]+a[i-1])
    return b  

def get_optimal_k(df,features=None):
    distortions = []
    K = range(1,10)
    for k in K:
        df,dist = cluster_data_kmeans(df,n_clusters=k,features=features)
        distortions.append(dist)  
    return K[np.argmax(curvature(distortions))]

def get_optimal_eps(df,features=None):
    df_scaled = scale_data(df,features)
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(df_scaled)
    distances, indices = nbrs.kneighbors(df_scaled)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    return distances[np.argmax(curvature(distances))]

def add_subplot(fig,field,row,col,x,y,clen=0.35,cbar_title=''):
    fig.add_trace(go.Scattergeo(
        lon = df['longitude'],
        lat = df['latitude'],
        text = df[field],
        mode = 'markers',
        marker_color = df[field],
        showlegend=False,
        marker = dict(colorscale='inferno',
                      colorbar=dict(x=x,y=y,len=clen,thickness=7,
                                    title=cbar_title,titleside='right',
                                    titlefont=dict(family='Courier New',size=12),
                                    tickfont=dict(family='Courier New',size=12))),
        ),row=row,col=col)
    return fig

def closest_lat_lon(df,lat=40,lon=-105):
    lats = df['latitude'].values.astype(float)
    lons = df['longitude'].values.astype(float)
    diff = np.inf
    index = 0
    for i in range(len(lats)):
        dist = (float(lat)-lats[i])**2+(float(lon)-lons[i])**2
        if dist < diff:
            index = i
            diff = dist
    return index        

def get_corrs(df,lat=40,lon=-105,features=None):
    min_max_scaler = preprocessing.MinMaxScaler()
    if features is None:
        features = [c for c in df.columns if c != 'latitude' and 
                c != 'longitude' and c != 'cluster']# and 
                #c != 'wind_speed' and c != 'clearsky_dhi' and
                #c != 'clearsky_dni' and c != 'clearsky_ghi' and
                #c != 'elevation' and c != 'landcover']
    x_scaled = min_max_scaler.fit_transform(df[features])
    df_scaled = pd.DataFrame(x_scaled)
    df_corrs = df_scaled.T.corr()
    return df_corrs.iloc[closest_lat_lon(df,lat=lat,lon=lon)]


def generate_figure(df,features=['ghi'],n_clusters=None,eps=None,min_pts=15,lat=40,lon=-105):
    feature_units = ['','','','','','','','','','','','','','','','',
                     'm','m2','hPa','W/m2','Celsius','Celsius',
                     'unitless',"type",'m/s','%','W/m2','degrees',
                     'cm','W/m2','W/m2','W/m2','W/m2','unitless','hPa',
                     '','','degrees','atm-cm','']
    if n_clusters is None:
        n_clusters = get_optimal_k(df,features)
    if eps is None:
        eps = get_optimal_eps(df,features)
    df = cluster_data_dbscan(df,
                             eps=eps,
                             min_pts=min_pts,
                             features=features)
    df,dist = cluster_data_kmeans(df,
                                  n_clusters=n_clusters,
                                  features=features)
    
    base_features = ['dbscan_cluster',None,None,'kmeans_cluster',None,None,'Correlations',None,
                     None,None,None,None,None,None,None,None,
                    'elevation', 'landcover', 'surface_pressure',
                    'ghi', 'air_temperature', 'dew_point', 'surface_albedo', 'cloud_type',
                    'wind_speed', 'relative_humidity', 'dni', 'solar_zenith_angle',
                    'total_precipitable_water', 'dhi', 'clearsky_dhi', 'clearsky_ghi',
                    'clearsky_dni', 'aod', 'cloud_press_acha','cld_opd_dcomp',
                    'cld_reff_dcomp', 'wind_direction','ozone','ssa']
    
    titles = (f'*{f}*' if f in features else f'{f}' for f in base_features if f is not None)

    fig = make_subplots(
    rows=5, cols=8,
    column_widths=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2],
    row_heights=[1,1,1,1,1,],
    specs=[
           [{"type":"scattergeo","rowspan":2,"colspan":2},None,None,{"type":"scattergeo","rowspan":2,"colspan":2},None,None,{"type":"scattergeo","rowspan":2,"colspan":2},None],
           [None,None,None,None,None,None,None,None],
           [{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"}],
           [{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"}],
           [{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"}]],
    subplot_titles=list(titles))

    colorbar_x = [0.085+i*(1.0/8+0.003) for i in range(8)]
    colorbar_y = [0.92-i*(1.0/5+0.0135) for i in range(5)]
    
    corrs = get_corrs(df,features=features,lat=lat,lon=lon)

    for i,field in enumerate(base_features):
        if field is None:
            pass
        else:
            cbar_title = feature_units[i]
            row_idx = i//8
            col_idx = i%8
            #print(row_idx,col_idx)
            if row_idx==0:
                if col_idx==0:
                    fig = add_subplot(fig,field,row_idx+1,col_idx+1,0.21,0.85,clen=0.25,cbar_title=cbar_title)
                if col_idx==3:
                    fig = add_subplot(fig,field,row_idx+1,col_idx+1,0.59,0.85,clen=0.25,cbar_title=cbar_title)
                if col_idx==6:
                    fig.add_trace(go.Scattergeo(lon = df['longitude'],
                              lat = df['latitude'],
                              text = corrs,
                              mode = 'markers',
                              marker_color = corrs,
                              marker = dict(colorscale='inferno',colorbar=dict(x=0.98,y=0.85,len=0.25,thickness=7)),
                              showlegend=False),row=1,col=7)    
            else:        
                fig = add_subplot(fig,field,row_idx+1,col_idx+1,colorbar_x[col_idx],colorbar_y[row_idx],clen=0.14,cbar_title=cbar_title)

    fig.update_geos(scope='usa')
    fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="LightSteelBlue",
    height=700, width=1800,
    )
    for n,i in enumerate(fig['layout']['annotations']):
        if '*' in i['text']:# in features:
            i['font'] = dict(size=14,color='red')
        else:
            i['font'] = dict(size=14)       
    return fig

base_features = [
                 'elevation', 'landcover', 'surface_pressure', 'ghi', 
                 'air_temperature', 'dew_point', 'surface_albedo', 'cloud_type',
                 'wind_speed', 'relative_humidity', 'dni', 'solar_zenith_angle',
                 'total_precipitable_water', 'dhi', 'clearsky_dhi', 'clearsky_ghi',
                 'clearsky_dni', 'aod', 'cloud_press_acha','cld_opd_dcomp',
                 'cld_reff_dcomp', 'wind_direction','ozone','ssa']

dropdown_options = []
for f in base_features:
    dropdown_options.append({'label':f,'value':f})

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([html.H6('NSRDB Clustering and Correlations',style={'width':'100%', 'textAlign': 'center','font-size': '25px',"padding-top": "1px"}),
    html.Div([
        "DBSCAN eps: ",
        dcc.Input(id='eps', value=0.01, type='text')
    ],style={'width': '20%', 'display': 'inline-block','textAlign':'center','height':'10px'}),
    html.Div([
        "DBSCAN min_pts: ",
        dcc.Input(id='min_pts', value=15, type='text')
    ],style={'width': '20%', 'display': 'inline-block','textAlign':'center','height':'10px'}),
    html.Div([
        "KMeans n_clusters: ",
        dcc.Input(id='n_clusters', value=5, type='text')
    ],style={'width': '20%', 'display': 'inline-block','textAlign':'center'}),
     html.Div([
        "Corr lat: ",
        dcc.Input(id='lat', value=40, type='text')
    ],style={'width': '20%', 'display': 'inline-block','textAlign':'center','height':'10px'}),
    html.Div([
        "Corr lon: ",
        dcc.Input(id='lon', value=-105, type='text')
    ],style={'width': '20%', 'display': 'inline-block','textAlign':'center','height':'10px'}),
    #html.H6('Feature Selection:',style={'width':'100%', 'textAlign': 'center','font-size': '26px'}),
    dcc.Dropdown(id='features',
    options=dropdown_options,
    value=['ghi','dni','dhi'],
    multi=True,
    style={'width':'1800px', 'textAlign': 'center'}),
    html.Div([dcc.Graph(id='my-output')],style={'width':'90%','textAlign': 'center'}),
    dcc.Interval(
            id = 'graph-update',
            interval = 1000,
            n_intervals = 0
        )

],style={"width": "1800px",
         "height": "950px",
         "display": "inline-block",
         "border": "3px #5c5c5c solid",
         "padding-top": "1px",
         "padding-left": "1px",
         "overflow": "hidden"})


@app.callback(
    Output(component_id='my-output', component_property='figure'),
    [Input(component_id='eps', component_property='value'),
     Input(component_id='min_pts', component_property='value'),
     Input(component_id='n_clusters', component_property='value'),
     Input(component_id='features', component_property='value'),
     Input(component_id='lat', component_property='value'),
     Input(component_id='lon', component_property='value')]
)
def update_output_div(eps,min_pts,n_clusters,features,lat,lon):
    if eps is not None:
        try:
            eps = float(eps)
            if eps <= 0.0:
                eps = None
        except:
            eps = None
    if min_pts is not None:
        try:
            min_pts = int(min_pts)
            if min_pts <= 0:
                min_pts = 15
        except:
            min_pts = 15
    if n_clusters is not None:
        try:
            n_clusters = int(n_clusters)
            if n_clusters <= 1:
                n_clusters = None
        except:
            n_clusters = None
    if lat is not None:
        try:
            lat = float(lat)
        except:
            lat = 40
    if lon is not None:
        try:
            lon = float(lon)
        except:
            lon = -105
    if not features:
        features = ['ghi','dni','dhi']
    fig = generate_figure(df,features,
                          n_clusters=n_clusters,
                          eps=eps,min_pts=min_pts,
                          lat=lat,lon=lon)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
