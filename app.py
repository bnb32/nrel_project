import plotly.graph_objects as go # or plotly.express as px
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
#from sklearn import preprocessing
from dask_ml import preprocessing
from sklearn.cluster import DBSCAN #KMeans, DBSCAN
from dask_ml.cluster import KMeans
from dask.array import corrcoef
import dask.dataframe as dd
from plotly.subplots import make_subplots
from sklearn.neighbors import NearestNeighbors
import numpy as np

pdf = pd.read_csv('nrel_df.csv')
df = dd.from_pandas(pdf,npartitions=4)
lats = df['latitude'].values.astype(float).compute()
lons = df['longitude'].values.astype(float).compute()

def scale_data(df,features=None):
    if features is None:
        features = [c for c in df.columns if c != 'latitude' and 
                    c != 'longitude' and 'cluster' not in c]
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.feature_names_in_=features
    return min_max_scaler.fit_transform(df)

def cluster_data_kmeans(df,n_clusters=8):
    clusters = KMeans(n_clusters=n_clusters, random_state=0).fit(df)
    return clusters

def cluster_data_dbscan(df,eps=0.1,min_pts=15):
    clusters = DBSCAN(eps=eps,min_samples=min_pts).fit(df.compute())
    return clusters

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

def get_optimal_k(df):
    distortions = []
    K = range(1,10)
    for k in K:
        clusters = cluster_data_kmeans(df,n_clusters=k)
        distortions.append(clusters.inertia_)
    return K[np.argmax(curvature(distortions))]

def get_optimal_eps(df):
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(df)
    distances, indices = nbrs.kneighbors(df)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    return distances[np.argmax(curvature(distances))]

def add_subplot(df,fig,field,row,col,x,y,clen=0.35,cbar_title='',initialize=False):
    if initialize:
        fig.add_trace(go.Scattergeo(
        lon = lons,
        lat = lats,
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
    else:
        fig.update_traces(go.Scattergeo(
        lon = lons,
        lat = lats,
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

def closest_lat_lon(lat=40,lon=-105):
    diff = np.inf
    index = 0
    for i in range(len(lats)):
        dist = (float(lat)-lats[i])**2+(float(lon)-lons[i])**2
        if dist < diff:
            index = i
            diff = dist
    #print(f'closest idx: {index}')        
    return index        

def get_corrs(df,lat=40,lon=-105):
    df_corrs = df.compute().T.corr() 
    return df_corrs.iloc[closest_lat_lon(lat=lat,lon=lon)]

def generate_figure(df,fig,features=['ghi'],n_clusters=None,eps=None,lat=40,lon=-105,min_pts=3,initialize=False):
    feature_units = ['','','','','','','','','','','','','','','','',
                     'm','m2','hPa','W/m2','Celsius','Celsius',
                     'unitless',"type",'m/s','%','W/m2','degrees',
                     'cm','W/m2','W/m2','W/m2','W/m2','unitless','hPa',
                     '','','degrees','atm-cm','']
    base_features = ['dbscan_cluster',None,None,'kmeans_cluster',None,None,'Correlations',None,
                     None,None,None,None,None,None,None,None,
                    'elevation', 'landcover', 'surface_pressure',
                    'ghi', 'air_temperature', 'dew_point', 'surface_albedo', 'cloud_type',
                    'wind_speed', 'relative_humidity', 'dni', 'solar_zenith_angle',
                    'total_precipitable_water', 'dhi', 'clearsky_dhi', 'clearsky_ghi',
                    'clearsky_dni', 'aod', 'cloud_press_acha','cld_opd_dcomp',
                    'cld_reff_dcomp', 'wind_direction','ozone','ssa']
    
#titles = (f'*{f}*' if f in features else f'{f}' for f in base_features if f is not None)
    titles = (f'{f}' for f in base_features if f is not None)
    
    df_scaled = scale_data(df,features)

    if n_clusters is None:
        n_clusters = get_optimal_k(df_scaled)
    if eps is None:
        eps = get_optimal_eps(df_scaled)
    
    df['dbscan_cluster'] = dd.from_array(cluster_data_dbscan(df_scaled,
        eps=eps,min_pts=min_pts).labels_)
    df['kmeans_cluster'] = dd.from_array(cluster_data_kmeans(df_scaled,
        n_clusters=n_clusters).labels_)
    
    if initialize:

        fig = make_subplots(rows=5, cols=8,
                column_widths=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2],
                row_heights=[1,1,1,1,1,],
                specs=[[{"type":"scattergeo","rowspan":2,"colspan":2},None,None,{"type":"scattergeo","rowspan":2,"colspan":2},None,None,{"type":"scattergeo","rowspan":2,"colspan":2},None],
                    [None,None,None,None,None,None,None,None],
                    [{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"}],
                    [{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"}],
                    [{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"},{"type":"scattergeo"}]],
                subplot_titles=list(titles))

    colorbar_x = [0.085+i*(1.0/8+0.003) for i in range(8)]
    colorbar_y = [0.92-i*(1.0/5+0.0135) for i in range(5)]
    
    corrs = get_corrs(df_scaled,lat=lat,lon=lon)

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
                    fig = add_subplot(df,fig,field,row_idx+1,col_idx+1,0.21,0.85,clen=0.25,cbar_title=cbar_title,initialize=initialize)
                if col_idx==3:
                    fig = add_subplot(df,fig,field,row_idx+1,col_idx+1,0.59,0.85,clen=0.25,cbar_title=cbar_title,initialize=initialize)
                if col_idx==6:
                    if initialize:
                        fig.add_trace(go.Scattergeo(lon = lons,
                              lat = lats,
                              text = corrs,
                              mode = 'markers',
                              marker_color = corrs,
                              marker = dict(colorscale='inferno',colorbar=dict(x=0.98,y=0.85,len=0.25,thickness=7)),
                              showlegend=False),row=1,col=7) 
                    else:
                        fig.update_traces(go.Scattergeo(lon = lons,
                              lat = lats,
                              text = corrs,
                              mode = 'markers',
                              marker_color = corrs,
                              marker = dict(colorscale='inferno',colorbar=dict(x=0.98,y=0.85,len=0.25,thickness=7)),
                              showlegend=False),row=1,col=7) 
                        
            else:        
                if initialize:
                    fig = add_subplot(df,fig,field,row_idx+1,col_idx+1,colorbar_x[col_idx],colorbar_y[row_idx],clen=0.14,cbar_title=cbar_title,initialize=initialize)

    fig.update_geos(scope='usa')
    fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="LightSteelBlue",
    height=700, width=1800,
    )
    for n,i in enumerate(fig['layout']['annotations']):
        if i['text'] in features:
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

default_features = ['ghi','dhi','dni']
default_eps = 0.4
default_n_clusters = 5
default_min_pts = 4

global fig
fig = None
fig = generate_figure(df,fig,features=default_features,n_clusters=default_n_clusters,eps=default_eps,min_pts=default_min_pts,initialize=True)

app.layout = html.Div([html.H6('NSRDB Clustering and Correlations',style={'width':'100%', 'textAlign': 'center','font-size': '25px',"padding-top": "1px"}),
    html.Div([
        "DBSCAN eps: ",
        dcc.Input(id='eps', value=default_eps, type='text')
    ],style={'width': '20%', 'display': 'inline-block','textAlign':'center','height':'10px'}),
    html.Div([
        "DBSCAN min_pts: ",
        dcc.Input(id='min_pts', value=default_min_pts, type='text')
    ],style={'width': '20%', 'display': 'inline-block','textAlign':'center','height':'10px'}),
    html.Div([
        "KMeans n_clusters: ",
        dcc.Input(id='n_clusters', value=default_n_clusters, type='text')
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
    value=default_features,
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
                min_pts = default_min_pts
        except:
            min_pts = default_min_pts
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
        features = default_features
    global fig 
    fig = generate_figure(df,fig,features,
                          n_clusters=n_clusters,
                          eps=eps,min_pts=min_pts,
                          lat=lat,lon=lon,initialize=False)
    return fig


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)
    #app.run_server(debug=True, use_reloader=False)
