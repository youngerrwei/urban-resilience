import json, warnings
import geopandas as gpd
import pandas as pd
import numpy as np
import colorsys
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity

# ============================== 配置路径 ==============================
SHP_PATH = "乡镇街道边界\\广佛街道级.shp"
CSV_PATH = "scored4.xlsx"

# ============================== 工具函数 ==============================
def safe_read_csv(path):
    return pd.read_excel(path)

def safe_read_shp(path):
    for enc in [None,"GBK","gb18030","utf-8"]:
        try:
            return gpd.read_file(path) if enc is None else gpd.read_file(path, encoding=enc)
        except:
            continue
    raise RuntimeError("Shapefile 读取失败")

def normalize_columns_inplace(df):
    df.rename(columns={c: str(c).strip() for c in df.columns}, inplace=True)

def pick_best_key(cols):
    candidates = ["district","name","street","xzqhmc_4","街道","街道名","街道名称","乡镇","乡镇街道","行政区","区县","名称"]
    for k in candidates:
        for c in cols:
            if str(c).strip().lower() == k.lower():
                return c
    return None

def normalized_values_for_dimensions(df, dims):
    arr = np.zeros((len(df), len(dims)), dtype=float)
    for i,d in enumerate(dims):
        col = pd.to_numeric(df[d], errors="coerce").astype(float)
        if col.isna().all():
            arr[:,i]=0
        else:
            mn,mx = np.nanmin(col),np.nanmax(col)
            arr[:,i] = np.where(np.isclose(mx,mn),0,(col-mn)/(mx-mn))
            arr[:,i][np.isnan(arr[:,i])] = 0.5
    return arr

def rgb_from_xyz_cube(x, y, z, brighten=0.35, gamma=1.8):
    def smoothstep(t):
        t = np.clip(t, 0, 1)
        return t*t*(3-2*t)
    x = smoothstep(x ** gamma); y = smoothstep(y ** gamma); z = smoothstep(z ** gamma)
    def mix_with_white(color, brighten):
        return (1 - brighten) * np.array(color) + brighten * np.array([255,255,255])
    C000 = mix_with_white([30,30,30],brighten)
    C100 = mix_with_white([220,50,50],brighten)
    C010 = mix_with_white([50,220,50],brighten)
    C001 = mix_with_white([50,50,220],brighten)
    C110 = mix_with_white([220,220,50],brighten)
    C011 = mix_with_white([50,220,220],brighten)
    C101 = mix_with_white([220,50,220],brighten)
    C111 = mix_with_white([245,245,245],brighten)
    weights = np.array([
        (1-x)*(1-y)*(1-z),
        x*(1-y)*(1-z),
        (1-x)*y*(1-z),
        (1-x)*(1-y)*z,
        x*y*(1-z),
        (1-x)*y*z,
        x*(1-y)*z,
        x*y*z
    ]) ** 1.5
    weights /= weights.sum()
    C = (weights[0]*C000 + weights[1]*C100 + weights[2]*C010 + weights[3]*C001 +
         weights[4]*C110 + weights[5]*C011 + weights[6]*C101 + weights[7]*C111)
    r,g,b = C/255.0
    h,s,v = colorsys.rgb_to_hsv(r,g,b)
    v = v**0.9; s = s**1.1
    r,g,b = colorsys.hsv_to_rgb(h,s,v)
    r,g,b = (r*255,g*255,b*255)
    return f"rgb({int(r)},{int(g)},{int(b)})"

# ============================== 读取数据 ==============================
gdf = safe_read_shp(SHP_PATH)
df  = safe_read_csv(CSV_PATH)
normalize_columns_inplace(gdf)
normalize_columns_inplace(df)
gdf_key = pick_best_key(gdf.columns)
df_key  = pick_best_key(df.columns)
if not gdf_key or not df_key:
    raise KeyError("无法找到连接键")
gdf["district"] = gdf[gdf_key].astype(str).str.strip()
df["district"]  = df[df_key].astype(str).str.strip()

PREFERRED_DIMENSIONS = ["social_score","economic_score","environmental_score","infrastructure_score","composite_score","composite_zscore"]
dimensions = [c for c in PREFERRED_DIMENSIONS if c in df.columns]
if len(dimensions) < 2:
    raise RuntimeError("数据中没有足够的维度列，请检查 CSV（需要至少2个优先维度列）")

for c in dimensions:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.sort_index().drop_duplicates(subset=["district"], keep="last")
gdf = gdf.merge(df[["district"]+dimensions], on="district", how="left")

if gdf.crs is not None:
    try:
        gdf = gdf.to_crs(epsg=4326)
    except Exception:
        pass
else:
    warnings.warn("Shapefile 未包含 CRS")

minx,miny,maxx,maxy = gdf.total_bounds
center_lat = (miny+maxy)/2
center_lon = (minx+maxx)/2
zoom_val = 10
geojson = json.loads(gdf.to_json())
chinese_labels = {
    "social_score":"社会得分",
    "economic_score":"经济得分",
    "environmental_score":"环境得分",
    "infrastructure_score":"基础设施得分",
    "composite_score":"总得分",
    "composite_zscore":"标准化总得分"
}

# ============================== Dash布局（保留原有并新增控件和两图） ==============================
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="广州城市韧性可视化")
server = app.server

map_dropdown_options = [{"label": chinese_labels.get(d,d), "value": d} for d in dimensions]
axis_dropdown_options = [{"label": chinese_labels.get(d,d), "value": d} for d in dimensions]

# 新增选择控件选项（用于新增两图）
knn_axis_options = axis_dropdown_options.copy()
density_axis_options = axis_dropdown_options.copy()
cluster_count_options = [{"label": str(k), "value": k} for k in [3,4,5,6,8,10,12,16,20]]
bin_options = [{"label": str(b), "value": b} for b in [10,20,30,40,60,80]]

app.layout = dbc.Container([
    dbc.NavbarSimple(brand="广州城市韧性可视化平台", color="primary", dark=True, fluid=True),
    dbc.Row([
        dbc.Col([  # 左上单维度地图
            dbc.Card([
                dbc.CardHeader(html.H5("单维度韧性地图")),
                dbc.CardBody([
                    html.Label("选择地图维度（绿色得分高、红色得分低）"),
                    dcc.Dropdown(id="map-single-dim", options=map_dropdown_options, value=dimensions[0], clearable=False),
                    dcc.Graph(id="map-single", style={"height":"320px"})
                ])
            ])
        ], md=6),
        dbc.Col([  # 右上三维散点
            dbc.Card([
                dbc.CardHeader(html.H5("三维散点图（红 = X维度高、绿 = Y维度高、蓝 = Z维度高）")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(id="x-dim", options=axis_dropdown_options, value=dimensions[0], clearable=False)),
                        dbc.Col(dcc.Dropdown(id="y-dim", options=axis_dropdown_options, value=dimensions[1], clearable=False)),
                        dbc.Col(dcc.Dropdown(id="z-dim", options=axis_dropdown_options, value=dimensions[2] if len(dimensions)>2 else dimensions[0], clearable=False))
                    ], className="g-2 mb-2"),
                    dcc.Graph(id="scatter3d", style={"height":"320px"})
                ])
            ])
        ], md=6)
    ], className="mt-3"),
    dbc.Row([  # 原有下方两列（RGB 地图 + 六边形密度）
        dbc.Col([  # 左下 RGB渐变地图
            dbc.Card([
                dbc.CardHeader(html.H5("RGB渐变色街道韧性地图")),
                dbc.CardBody([dcc.Graph(id="map-rgb", style={"height":"320px"})])
            ])
        ], md=6),
        dbc.Col([  # 右下 六边形密度图（原有）
            dbc.Card([
                dbc.CardHeader(html.H5("六边形密度（蜂窝）图")),
                dbc.CardBody([dcc.Graph(id="hex-density", style={"height":"320px"})])
            ])
        ], md=6)
    ], className="mt-3"),
    dbc.Row([  # 新增两图行（左：KDE聚类；右：密度分布）
        dbc.Col([  # 左下下 KNN / KDE 聚类图（用户可选 X/Y & 簇数）
            dbc.Card([
                dbc.CardHeader(html.H5("KMeans + KDE（核密度）聚类图（可选维度 & 簇数）")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Label("X 维度："), width=2),
                        dbc.Col(dcc.Dropdown(id="knn-x", options=knn_axis_options, value=dimensions[0], clearable=False), width=4),
                        dbc.Col(html.Label("簇数 K："), width=2),
                        dbc.Col(dcc.Dropdown(id="knn-k", options=cluster_count_options, value=8, clearable=False), width=4),
                    ], className="g-2 mb-2"),
                    dbc.Row([
                        dbc.Col(html.Label("Y 维度："), width=2),
                        dbc.Col(dcc.Dropdown(id="knn-y", options=knn_axis_options, value=dimensions[1], clearable=False), width=4),
                        dbc.Col(html.Div("（显示 KMeans 聚类 & KDE 等高线）"), width=6)
                    ], className="g-2 mb-2"),
                    dcc.Graph(id="knn-kde-fig", style={"height":"360px"})
                ])
            ])
        ], md=6),

        dbc.Col([  # 右下下 密度分布图（用户可选 X/Y & bins）
            dbc.Card([
                dbc.CardHeader(html.H5("密度分布图（可选 X & Y 维度）")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Label("X 维度："), width=2),
                        dbc.Col(dcc.Dropdown(id="den-x", options=density_axis_options, value=dimensions[0], clearable=False), width=4),
                        dbc.Col(html.Label("bins："), width=2),
                        dbc.Col(dcc.Dropdown(id="den-bins", options=bin_options, value=30, clearable=False), width=4)
                    ], className="g-2 mb-2"),
                    dbc.Row([
                        dbc.Col(html.Label("Y 维度："), width=2),
                        dbc.Col(dcc.Dropdown(id="den-y", options=density_axis_options, value=dimensions[1], clearable=False), width=4),
                        dbc.Col(html.Div("（使用热力 / 密度图展示二元分布）"), width=6)
                    ], className="g-2 mb-2"),
                    dcc.Graph(id="density-fig", style={"height":"360px"})
                ])
            ])
        ], md=6)
    ], className="mt-3")
], fluid=True)
# ============================== 新增控件 & 第三行图 ==============================
dist_axis_options = axis_dropdown_options.copy()

app.layout.children.append(
    dbc.Row([  # 第三行（分布曲线 + 成对散点热力）
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("单维度分布曲线（直方图 + KDE）")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Label("选择维度："), width=3),
                        dbc.Col(dcc.Dropdown(id="dist-dim", options=dist_axis_options, value=dimensions[0], clearable=False), width=9),
                    ], className="g-2 mb-2"),
                    dcc.Graph(id="dist-fig", style={"height":"320px"})
                ])
            ])
        ], md=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("成对维度散点 + KDE 热力")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Label("X 维度："), width=2),
                        dbc.Col(dcc.Dropdown(id="pair-x", options=dist_axis_options, value=dimensions[0], clearable=False), width=4),
                        dbc.Col(html.Label("Y 维度："), width=2),
                        dbc.Col(dcc.Dropdown(id="pair-y", options=dist_axis_options, value=dimensions[1], clearable=False), width=4),
                    ], className="g-2 mb-2"),
                    dcc.Graph(id="pair-kde-fig", style={"height":"320px"})
                ])
            ])
        ], md=6)
    ], className="mt-3")
)

# ============================== Callbacks（保留原有） ==============================
@app.callback(Output("map-single","figure"), Input("map-single-dim","value"))
def update_single_map(map_dim):
    hover_texts = []
    for _, row in gdf.iterrows():
        text = f"<b>{row['district']}</b><br>"
        text += "<br>".join([f"{chinese_labels.get(d,d)}: {row[d]:.3f}" for d in dimensions])
        hover_texts.append(text)
    gdf["_hover"] = hover_texts

    # ✅ 红–绿渐变（红 = 得分低，绿 = 得分高）
    red_green_scale = [
        [0.0, "#d73027"],
        [0.25, "#fc8d59"],
        [0.5, "#fee08b"],
        [0.75, "#d9ef8b"],
        [1.0, "#1a9850"]
    ]

    fig = px.choropleth_mapbox(
        gdf, geojson=geojson, locations="district", color=map_dim,
        featureidkey="properties.district",
        mapbox_style="open-street-map",
        center={"lat": center_lat, "lon": center_lon}, zoom=zoom_val,
        color_continuous_scale=red_green_scale,
        hover_data={"_hover": True, "district": False, map_dim: False}
    )
    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
    fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0}, height=320, coloraxis_showscale=False)
    return fig


@app.callback(Output("scatter3d","figure"), [Input("x-dim","value"), Input("y-dim","value"), Input("z-dim","value")])
def update_scatter(xdim, ydim, zdim):
    POINT_SIZE = 2  # ✅ 这里改即可控制大小
    plot_df = gdf.copy()
    arr_norm = normalized_values_for_dimensions(plot_df,[xdim,ydim,zdim])

    # ✅ 使用更鲜艳的 SOM 风格色彩空间（更接近你图中的效果）
    def som_rgb(x, y, z):
        # 高饱和度 + 中等亮度
        r = np.clip(255 * x, 0, 255)
        g = np.clip(255 * y, 0, 255)
        b = np.clip(255 * z, 0, 255)
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        r, g, b = colorsys.hsv_to_rgb(h, min(1.0, s*1.3), min(1.0, v*1.1))
        return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

    plot_df["_hex"] = [som_rgb(a[0], a[1], a[2]) for a in arr_norm]

    fig = go.Figure(data=[go.Scatter3d(
        x=plot_df[xdim], y=plot_df[ydim], z=plot_df[zdim],
        mode='markers',
        marker=dict(size=POINT_SIZE,color=plot_df["_hex"],opacity=0.9,line=dict(width=0)),
        text=plot_df["district"],
        hovertemplate="<b>%{text}</b><br>"+f"{chinese_labels.get(xdim,xdim)}: "+"%{x:.3f}<br>"+f"{chinese_labels.get(ydim,ydim)}: "+"%{y:.3f}<br>"+f"{chinese_labels.get(zdim,zdim)}: "+"%{z:.3f}<br><extra></extra>"
    )])
    fig.update_layout(scene=dict(
        xaxis_title=chinese_labels.get(xdim,xdim),
        yaxis_title=chinese_labels.get(ydim,ydim),
        zaxis_title=chinese_labels.get(zdim,zdim),
        aspectmode="cube"
    ), margin=dict(l=0,r=0,t=30,b=0))
    return fig


@app.callback(Output("map-rgb","figure"), [Input("x-dim","value"), Input("y-dim","value"), Input("z-dim","value")])
def update_rgb_map(xdim, ydim, zdim):
    plot_df = gdf.copy()
    arr_norm = normalized_values_for_dimensions(plot_df,[xdim,ydim,zdim])

    def som_rgb(x, y, z):
        r = np.clip(255 * x, 0, 255)
        g = np.clip(255 * y, 0, 255)
        b = np.clip(255 * z, 0, 255)
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        r, g, b = colorsys.hsv_to_rgb(h, min(1.0, s*1.3), min(1.0, v*1.1))
        return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

    plot_df["_hex"] = [som_rgb(a[0],a[1],a[2]) for a in arr_norm]

    fig = go.Figure()
    for _, row in plot_df.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        if geom.geom_type == "Polygon":
            coords = [list(geom.exterior.coords)]
        elif geom.geom_type == "MultiPolygon":
            coords = [list(poly.exterior.coords) for poly in geom.geoms]
        else:
            continue
        for poly_coords in coords:
            lons,lats = zip(*poly_coords)
            hover_text = f"<b>{row['district']}</b><br>" + "<br>".join([f"{chinese_labels.get(d,d)}: {row[d]:.3f}" for d in dimensions])
            fig.add_trace(go.Scattermapbox(
                lon=lons, lat=lats, mode='lines', fill='toself',
                fillcolor=row["_hex"], line=dict(width=0.4,color="#222222"),
                hoverinfo='text', text=hover_text
            ))

    fig.update_layout(mapbox_style="open-street-map",
                      mapbox_center={"lat":center_lat,"lon":center_lon},
                      mapbox_zoom=zoom_val,
                      margin={"r":0,"t":30,"l":0,"b":0}, height=320,
                      showlegend=False)
    return fig


# 右下原有六边形密度图（保持简洁，仅显示计数）
@app.callback(Output("hex-density", "figure"), [Input("x-dim", "value"), Input("y-dim", "value"), Input("z-dim", "value")])
def update_hex_density(xdim, ydim, zdim):
    plot_df = gdf.copy()
    arr_norm = normalized_values_for_dimensions(plot_df, [xdim, ydim, zdim])
    plot_df["_hex"] = [rgb_from_xyz_cube(a[0], a[1], a[2]) for a in arr_norm]

    nx, ny = 8, 8
    n_hex = nx * ny
    kmeans = KMeans(n_clusters=n_hex, random_state=42)
    # 用三维归一化向量做聚类
    arr_for_cluster = normalized_values_for_dimensions(plot_df, [xdim, ydim, zdim])
    plot_df["_cluster"] = kmeans.fit_predict(arr_for_cluster)

    # 统计每簇个数与平均 RGB
    cluster_stats = plot_df.groupby("_cluster").agg(
        count=("district","size"),
        rgb_first=("_hex", lambda s: s.iloc[0])
    ).reset_index()

    # 布局六边形中心（错行紧密排列）
    hex_centers = []
    hex_radius = 1
    for i in range(nx):
        for j in range(ny):
            if len(hex_centers) >= n_hex:
                break
            cx = i * 1.5 * hex_radius
            cy = j * np.sqrt(3) * hex_radius + (i % 2) * np.sqrt(3) / 2 * hex_radius
            hex_centers.append((cx, cy))
    cluster_stats["cx"] = [c[0] for c in hex_centers][:len(cluster_stats)]
    cluster_stats["cy"] = [c[1] for c in hex_centers][:len(cluster_stats)]

    # 绘制：只显示每簇街道数量（数字）与六边形色块（统一或簇色）
    fig = go.Figure()
    for _, row in cluster_stats.iterrows():
        color = row["rgb_first"] if pd.notna(row["rgb_first"]) else "rgb(200,200,200)"

        fig.add_trace(go.Scatter(
            x=[row["cx"]], y=[row["cy"]],
            mode='markers+text',
            marker=dict(size=36, symbol='hexagon', color=color, line=dict(width=1,color='black')),
            text=[str(int(row["count"]))],
            textposition="middle center",
            hoverinfo='skip',
            showlegend=False
        ))

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x"),
        margin=dict(l=6, r=6, t=8, b=6),
        height=320,
        plot_bgcolor='white'
    )
    return fig

# ============================== 新增：KNN/KDE 聚类图（左下下） ==============================
@app.callback(
    Output("knn-kde-fig", "figure"),
    [Input("knn-x","value"), Input("knn-y","value"), Input("knn-k","value")]
)
def update_knn_kde(xdim, ydim, k_clusters):
    plot_df = gdf.copy()
    # 取出用户选的两个维度（若有缺失则 drop）
    data = plot_df[[xdim, ydim, "district"]].dropna().copy()
    X = normalized_values_for_dimensions(data, [xdim, ydim])

    # KMeans 聚类
    k = int(k_clusters) if k_clusters is not None else 8
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    data["cluster"] = labels
    data["_x"] = X[:,0]; data["_y"] = X[:,1]

    # KDE（核密度）估计 —— 在归一化坐标上画等高线
    kde = KernelDensity(bandwidth=0.08, kernel='gaussian')
    kde.fit(X)
    # 网格
    xi = np.linspace(0,1,200)
    yi = np.linspace(0,1,200)
    xx, yy = np.meshgrid(xi, yi)
    grid = np.vstack([xx.ravel(), yy.ravel()]).T
    log_dens = kde.score_samples(grid)
    zz = np.exp(log_dens).reshape(xx.shape)

    # 画图：先等高线（作为背景），再绘制簇点，点颜色用簇号调色
    fig = go.Figure()

    # Contour 等高线（核密度）
    fig.add_trace(go.Contour(
        x=xi, y=yi, z=zz,
        showscale=False,
        contours=dict(start=zz.min(), end=zz.max(), size=(zz.max()-zz.min())/8),
        opacity=0.6, hoverinfo='skip', colorscale='Blues'
    ))

    # 每个样本点（按簇着色），只画小点，不显示过多文本
    palette = px.colors.qualitative.Safe
    for c in np.unique(labels):
        sub = data[data["cluster"]==c]
        color = palette[int(c) % len(palette)]
        fig.add_trace(go.Scatter(
            x=sub["_x"], y=sub["_y"],
            mode='markers',
            marker=dict(size=6, color=color, opacity=0.9, line=dict(width=0)),
            name=f"簇 {int(c)}",
            hovertemplate="<b>%{text}</b><br>簇: "+str(int(c))+"<extra></extra>",
            text=sub["district"]
        ))

    # 绘制簇心
    fig.add_trace(go.Scatter(
        x=centers[:,0], y=centers[:,1],
        mode='markers+text',
        marker=dict(size=14, symbol='x', color='black'),
        text=[str(i) for i in range(len(centers))],
        textposition="top center",
        hoverinfo='skip',
        showlegend=False
    ))

    fig.update_layout(
        title=f"KMeans ({k}) + KDE（核密度）—— 维度：{chinese_labels.get(xdim,xdim)} vs {chinese_labels.get(ydim,ydim)}",
        xaxis=dict(title=chinese_labels.get(xdim,xdim), range=[0,1], showgrid=False, zeroline=False),
        yaxis=dict(title=chinese_labels.get(ydim,ydim), range=[0,1], showgrid=False, zeroline=False, scaleanchor="x"),
        margin=dict(l=6,r=6,t=28,b=6),
        height=360
    )
    return fig

# ============================== 新增：密度分布图（右下下） ==============================
@app.callback(
    Output("density-fig", "figure"),
    [Input("den-x","value"), Input("den-y","value"), Input("den-bins","value")]
)
def update_density_plot(xdim, ydim, n_bins):
    plot_df = gdf.copy()
    data = plot_df[[xdim, ydim, "district"]].dropna().copy()

    # 使用归一化值以保证 bin 划分相对统一
    X = normalized_values_for_dimensions(data, [xdim, ydim])
    data["_x"] = X[:,0]; data["_y"] = X[:,1]

    # 使用 Plotly 的 density_heatmap（矩形 bin）来近似展示密度分布；
    # nbinsx / nbinsy 可以近似控制 hex 的粒度（用户可调整）
    fig = px.density_heatmap(
        data_frame=data, x="_x", y="_y",
        nbinsx=int(n_bins), nbinsy=int(n_bins),
        color_continuous_scale="Viridis",
        labels={"_x": chinese_labels.get(xdim,xdim), "_y": chinese_labels.get(ydim,ydim)},
        hover_data={"district": True}
    )

    fig.update_traces(hovertemplate="样本数: %{z}<extra></extra>")
    fig.update_layout(
        title=f"密度分布：{chinese_labels.get(xdim,xdim)} vs {chinese_labels.get(ydim,ydim)}",
        xaxis=dict(range=[0,1], showgrid=False, zeroline=False, title=chinese_labels.get(xdim,xdim)),
        yaxis=dict(range=[0,1], showgrid=False, zeroline=False, scaleanchor="x", title=chinese_labels.get(ydim,ydim)),
        margin=dict(l=6,r=6,t=28,b=6),
        height=360,
        coloraxis_colorbar=dict(title="计数")
    )
    return fig

# 单维度分布曲线
@app.callback(Output("dist-fig","figure"), Input("dist-dim","value"))
def update_distribution(dim):
    data = gdf[dim].dropna()
    fig = go.Figure()
    # 直方图
    fig.add_trace(go.Histogram(x=data, nbinsx=30, histnorm='probability density',
                               marker=dict(color='lightblue'), opacity=0.6, name="直方图"))
    # KDE 曲线
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    xi = np.linspace(data.min(), data.max(), 200)
    fig.add_trace(go.Scatter(x=xi, y=kde(xi), mode='lines',
                             line=dict(color='darkblue', width=2), name="KDE"))
    fig.update_layout(
        title=f"{chinese_labels.get(dim,dim)} 分布曲线",
        xaxis_title=chinese_labels.get(dim,dim),
        yaxis_title="密度",
        bargap=0.05
    )
    return fig

# 成对散点 + KDE 热力
@app.callback(Output("pair-kde-fig","figure"), [Input("pair-x","value"), Input("pair-y","value")])
def update_pair_kde(xdim, ydim):
    data = gdf[[xdim, ydim]].dropna()
    X = normalized_values_for_dimensions(data, [xdim, ydim])

    # KDE 网格
    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(bandwidth=0.08).fit(X)
    xi = np.linspace(0,1,100); yi = np.linspace(0,1,100)
    xx, yy = np.meshgrid(xi, yi)
    grid = np.vstack([xx.ravel(), yy.ravel()]).T
    zz = np.exp(kde.score_samples(grid)).reshape(xx.shape)

    fig = go.Figure()
    # 背景 KDE 等高线
    fig.add_trace(go.Contour(
        x=xi, y=yi, z=zz,
        colorscale="Viridis", opacity=0.6, showscale=False,
        contours=dict(showlines=False)
    ))
    # 散点
    fig.add_trace(go.Scatter(
        x=X[:,0], y=X[:,1],
        mode="markers", marker=dict(size=5, color="black", opacity=0.6),
        name="样本点"
    ))
    fig.update_layout(
        title=f"{chinese_labels.get(xdim,xdim)} vs {chinese_labels.get(ydim,ydim)} 散点 + KDE 热力",
        xaxis_title=chinese_labels.get(xdim,xdim),
        yaxis_title=chinese_labels.get(ydim,ydim)
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)
