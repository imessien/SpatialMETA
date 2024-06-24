import scanpy as sc
import anndata
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import dash
from dash import dcc
from dash import html
import dash
from dash import callback_context, no_update
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import numpy as np
import matplotlib 
import scipy
import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import pandas as pd 
import json

from ..utils.utils import (
    get_spatial_image,
    get_spatial_scalefactors_dict,
    rgb2hex,
    points_within_distance_along_path,
    points_within_distance_outside_path,
    ResizeLongestSide
)
from ..plotly.layout import (
    layout, header, image, right_sidebar,
    left_sidebar
)

class Wrapper:
    """
    Plotly wrapper for spatial data.
    
    :param adata: Anndata object containing spatial data.
    :param log: Log of the current state of the plot, defaults to None.
    :param n_clicks: Dictionary of the number of clicks for each action, defaults to dict(annotate=0).
    
    """
    def __init__(self,
        adata: sc.AnnData
    ):
        self.adata = adata
        self.log = None
        self.n_clicks = dict(
            annotate = 0
        )

    def get_feature_color(self, feature_name, cmap_name='viridis'):
        gene_exp = self.get_feature_value(feature_name)
        vmax = np.max(gene_exp)
        cmap = getattr(matplotlib.cm, cmap_name)
        colors = list(map(lambda x: 
            rgb2hex(cmap(x/vmax))[:-2], 
            gene_exp
        ))
        return colors 

    def get_categorical_color(self, feature_type):
        features = pd.Categorical(self.adata.obs[feature_type])
        palette = sc.plotting.palettes.godsnot_102
        colormap = dict(zip(
            features.categories,
            palette
        ))
        colormap['undefined'] = '#F7F7F7'
        colors = list(map(lambda x: colormap[x], features))
        self.log = colors
        return colors

    def get_feature_value(self, feature_name):
        if feature_name is None:
            gene_exp = np.zeros(self.adata.shape[0])
        gene_exp = self.adata.layers['normalized'][
            :,list(self.adata.var.index).index(feature_name)
        ]
        if scipy.sparse.issparse(gene_exp):
            gene_exp = gene_exp.toarray().flatten()
        return gene_exp 

    def to_general_scatter(self,
        feature_type,
        scale,
        feature_name=None,
        x=0,
        y=0
    ):
        if feature_type == 'expression':
            return self.to_feature_scatter(
                feature_name, scale, x, y
            )
        else: 
            if is_numeric_dtype(self.adata.obs[feature_type]):
                vmin = np.min(self.adata.obs[feature_type])
                vmax = np.max(self.adata.obs[feature_type])
                colors = list(map(lambda x:
                    rgb2hex(plt.cm.viridis((x - vmin) / (vmax - vmin)))[:-2],
                    self.adata.obs[feature_type]
                ))
                scatter = go.Scatter(
                    x = self.adata.obsm["spatial"][:,0] * scale,
                    y = self.adata.obsm["spatial"][:,1] * scale,
                    marker = {
                        "color": colors,
                        "size": 4
                    },
                    mode = 'markers',
                    hoverinfo='none',
                )
            elif is_string_dtype(self.adata.obs[feature_type]):
                colors = self.get_categorical_color(feature_name)
                scatter = go.Scatter(
                    x = self.adata.obsm["spatial"][:,0] * scale,
                    y = self.adata.obsm["spatial"][:,1] * scale,
                    marker = {
                        "color": colors,
                        "size": 4
                    },
                    mode = 'markers',
                    hoverinfo='none',
                )
                return scatter, None

    def to_feature_scatter(self, 
        feature_name, 
        scale,
        x=0,
        y=0
    ):
        colors = self.get_feature_color(feature_name)
        scatter = go.Scatter(
            x = self.adata.obsm["spatial"][:,0] * scale,
            y = self.adata.obsm["spatial"][:,1] * scale,
            marker = {
                "color": colors,
                "size": 4
            },
            mode = 'markers',
            hoverinfo='none',
        )

        colorscale = self.to_continuous_colorscale(
            'viridis', x, y, scale, feature_name
        )
        return scatter, colorscale

    def to_continuous_colorscale(
        self, 
        name, 
        x, 
        y, 
        scale,
        feature_name
    ):
        name = name[0].upper() + name[1:]
        n = 128
        return go.Bar(
            orientation = "h",
            y=[y * scale] * n,
            x=[x * scale] * n,
            customdata=[(x + 1) / n for x in range(n)],
            marker=dict(
                color=list(range(n)), 
                colorscale=name, 
                line_width=0
            ),
            hovertemplate="%{customdata}",
            name=feature_name,
            width=10,
            
        )

    def to_feature_summary_fig(self, feature_name, groups=None):
        if groups is None:
            groups = {"Selected": list(range(len(self.adata)))}
        gene_exp = self.get_feature_value(feature_name)
        x = []
        y = []
        for k,v in groups.items():
            x.append(gene_exp[v])
            y.append(k)

        summary_fig = ff.create_distplot(
            x,
            y,
            show_rug=False,
            bin_size=0.25
        )
        self.summary_fig = summary_fig
        summary_fig.update_layout(
            autosize=True,
            width=200,
            height=120,
            margin=go.layout.Margin(
                l=0, #left margin
                r=0, #right margin
                b=0, #bottom margin
                t=0, #top margin
            ),
            xaxis=go.layout.XAxis(
                showline=True,
                color='#000000',
                linewidth=1,
                linecolor='#000000'
            ),
            yaxis=go.layout.YAxis(
                showline=True,
                color='#000000',
                linewidth=1,
                linecolor='#000000'
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode=False,
            legend=dict(
                x=0,
                y=1.5,
                orientation='h'
            )
        )
        return summary_fig 

    def _to_obs_dropdown(self):
        return [
            {"label": "Feature expression (string)", "value": 'expression'},
        ] + [{"label": f"{x} ({str(self.adata.obs[x].dtype)})", "value": x} for x in list(filter(lambda x: 
            is_string_dtype(self.adata.obs[x]) or is_numeric_dtype(self.adata.obs[x]), self.adata.obs.columns))]

    def to_plotly(
        self,
        init_feature='COL3A1'
    ):
        """
        Use Dash to create a plotly figure for spatial data.
        
        :param init_feature: Initial feature to display, defaults to 'COL3A1'.
        
        :return: Dash app.
        """
        img_, scale = get_spatial_image(self.adata)
        scale_factors = get_spatial_scalefactors_dict(self.adata)
        img = ResizeLongestSide(1024).apply_image(img_)
        scale = scale * (img.shape[0] / img_.shape[0])

        spatial_key = list(self.adata.uns["spatial"].keys())[0]
        self.adata.uns['spatial'][spatial_key]['images']['scaledres'] = img
        self.adata.uns['spatial'][spatial_key]['scalefactors']['tissue_scaledres_scalef'] = scale


        diameter = (65/scale_factors['spot_diameter_fullres']) * (1/scale) # 1 pixel = N μm

        default_image_height = 1024
        scalefactor = default_image_height / img.shape[1]
        image_width = img.shape[0] * scalefactor
        fig = go.Figure(
            data = [
                go.Image(
                    z = img,
                    hoverinfo='none'
                )
            ],
            layout = layout,
            
        )
        fig.update_layout(
            autosize=True,
            width=700,
            height=400,
            dragmode = 'select',
            newselection = {
                "line": {
                    "color": 'white',
                    "width": 3
                },
            },
            activeselection = {
                "fillcolor": "#F7F7F7"
            }
        )

        scatter, colorscale = self.to_feature_scatter(
            init_feature, scale, 10, 15
        )
        scatter.update(
            unselected=dict(marker=dict(
                opacity=0.5
            ))
        )
        fig.add_trace(
            scatter,
        )
        feature_scatter_trace_id = len(fig.data)-1
        if colorscale is not None:
            fig.add_trace(colorscale)
            feature_colorscale_trace_id = len(fig.data)-1

        summary_fig = self.to_feature_summary_fig(
            init_feature
        )

        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP]
        app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        app.layout = html.Div([
            header,
            dbc.Container(
                [
                    dbc.Row(
                        id="app-content",
                        children=[
                            dbc.Col(left_sidebar(self.adata, init_feature), md=3),
                            dbc.Col(image(fig), md=6), 
                            dbc.Col(right_sidebar(summary_fig, init_feature=init_feature), md=3)
                        ],
                    ),
                ],
                fluid=True,
            ),
        ])

        # Callbacks
        @app.callback(
            dash.dependencies.Output(
                "main_graph", 
                "figure", allow_duplicate=True
            ),
            [
                dash.dependencies.Input("feature_name_select","value"),
                dash.dependencies.Input("display-dropdown", "value")
            ],
            prevent_initial_call=True
        )
        def update_scatter_feature(feature_name, display_type):
            self.log = display_type
            trace = fig.data[feature_scatter_trace_id]
            if display_type == 'expression':
                trace['marker']['color'] = self.get_feature_color(
                    feature_name
                )
            else:
                if is_numeric_dtype(self.adata.obs[display_type]):
                    vmin = np.min(self.adata.obs[display_type])
                    vmax = np.max(self.adata.obs[display_type])
                    colors = list(map(lambda x:
                        rgb2hex(plt.cm.viridis((x - vmin) / (vmax - vmin)))[:-2],
                        self.adata.obs[display_type]
                    ))
                    trace['marker']['color'] = colors
                elif is_string_dtype(self.adata.obs[display_type]):
                    trace['marker']['color'] = self.get_categorical_color(
                        display_type
                    )

            return fig

        @app.callback(
            dash.dependencies.Output(
                "main_graph", 
                "figure", allow_duplicate=True
            ),
            [
                dash.dependencies.Input("feature_opacity_slider","value")
            ],
            prevent_initial_call=True
        )
        def update_scatter_opacity(feature_opacity):
            trace = fig.data[feature_scatter_trace_id]
            trace['opacity'] = feature_opacity / 100
            return fig

        @app.callback(
            dash.dependencies.Output(
                "main_graph", 
                "figure", allow_duplicate=True
            ),
            [
                dash.dependencies.Input("image_opacity_slider","value")
            ],
            prevent_initial_call=True
        )
        def update_scatter_opacity(feature_opacity):
            trace = fig.data[0]
            trace['opacity'] = feature_opacity / 100
            return fig

        @app.callback(
            dash.dependencies.Output(
                "main_graph", 
                "figure", allow_duplicate=True
            ),
            [
                dash.dependencies.Input("marker_size_slider","value")
            ],
            prevent_initial_call=True
        )
        def update_scatter_opacity(size):
            trace = fig.data[feature_scatter_trace_id]
            trace['marker']['size'] = int(size)
            return fig

        @app.callback(
            dash.dependencies.Output(
                "feature-summary-header", "children"
            ),
            [
                dash.dependencies.Input("feature_name_select","value")
            ]
        )
        def update_feature_header(value):
            return "Expression of " + value

        @app.callback(
            dash.dependencies.Output(
                "expression_histogram",
                "figure"
            ),
            [
                dash.dependencies.Input("main_graph", "selectedData"),
                dash.dependencies.Input("feature_name_select","value")
            ]
        )
        def update_on_selection(value, feature_name):
            if feature_name is None:
                return None
            if value is None or len(value['points']) == 0:
                summary_fig = self.to_feature_summary_fig(
                feature_name
            )
                return summary_fig

            indices = [x['pointIndex'] for x in value['points']]
            summary_fig = self.to_feature_summary_fig(
                feature_name, {
                    "Selected": indices
                }
            )
            return summary_fig

        @app.callback(
            [
                dash.dependencies.Output("display-dropdown", "options"),
                dash.dependencies.Output("alert-auto", "is_open", allow_duplicate=True),
                dash.dependencies.Output("alert-auto", "children", allow_duplicate=True),
                dash.dependencies.Output("alert-auto", "color", allow_duplicate=True),
                dash.dependencies.Output(
                    "main_graph", 
                    "figure", allow_duplicate=True
                ),
            ],
            [
                dash.dependencies.Input('annotation-group-name', 'value'),
                dash.dependencies.Input('annotation-class-name', 'value'),
                dash.dependencies.Input('annotation-submit','n_clicks'),
                dash.dependencies.Input("main_graph", "selectedData"),
                dash.dependencies.Input("display-dropdown", "value")
            ],
            [
                State("alert-auto", "is_open")
            ],
            prevent_initial_call=True
        )
        def annotate_adata(group_name, class_name, n_click, value, display_type, is_open):
            self.log = value
            # primary secondary success warning danger info light dark
            if group_name is None and class_name is None and value is None:
                return "", self._to_obs_dropdown(), is_open, "", "info", no_update
            if group_name is None:
                return self._to_obs_dropdown(), not is_open, "Please provide group name", "warning", no_update
            if class_name is None:
                return self._to_obs_dropdown(), not is_open, "Please provide class name", "warning", no_update
            if value is None or len(value['points']) == 0: 
                return self._to_obs_dropdown(), not is_open, "No points were selected", "warning", no_update
            if group_name not in self.adata.obs.columns:
                self.adata.obs[group_name] = 'undefined'

            if n_click is None or n_click == self.n_clicks['annotate']:
                return self._to_obs_dropdown(), is_open, "", "info", no_update

            self.n_clicks['annotate'] = n_click
            cdex = list(self.adata.obs.columns).index(group_name)
            indices = [x['pointIndex'] for x in value['points']]
            indices = list(map(lambda z: z[0], filter(lambda x: 
                x[1] == 'undefined',
                zip(indices, self.adata.obs.iloc[indices, cdex])
            )))

            self.adata.obs.iloc[
                indices, 
                cdex
            ] = class_name
            if group_name == display_type:
                trace = fig.data[feature_scatter_trace_id]
                trace['marker']['color'] = self.get_categorical_color(
                    group_name
                )

            return (
                self._to_obs_dropdown(),
                not is_open,
                f"{len(value['points'])} of points was annotated!",
                "success",
                fig,
            )

        @app.callback(
            Output("selected-points", "children"),
            [
                Input("main_graph", "selectedData")
            ],
            prevent_initial_call=True
        )
        def log_selected_data(selectedData):
            if selectedData is None:
                return "No points selected, showing all data"
            return f"{len(selectedData['points'])} points selected"

        @app.callback(
            [
                dash.dependencies.Output("trajectory-pre", "children"),
                dash.dependencies.Output("alert-auto", "is_open", allow_duplicate=True),
                dash.dependencies.Output("alert-auto", "children", allow_duplicate=True),
                dash.dependencies.Output("alert-auto", "color", allow_duplicate=True),
            ],
            [
                Input("main_graph", "relayoutData"),
                dash.dependencies.Input("distance_slider","value")
            ],
            prevent_initial_call=True,
        )
        def on_new_trajectory(relayout_data, distance):
            for key in relayout_data:
                if "shapes" in key and len(relayout_data[key]) > 0:
                    path = relayout_data[key][-1]['path']

                    spatial_coord = self.adata.obsm['spatial'] * scale
                    distance_in_pixel = distance / diameter 

                    if path.endswith("Z"):
                        # Close path as region
                        points, indices, distance, locations = points_within_distance_outside_path(
                            path,
                            spatial_coord,
                            0
                        )
                        if 'region' not in self.adata.uns:
                            self.adata.uns['region'] = {}
                        last = len(self.adata.uns['region'])
                        self.adata.uns['region'][f"region_{last+1}"] = {
                            "path": path,
                            "points": np.array(list(map(lambda x: [x.x, x.y], points))),
                            "indices": indices,
                            "distances": distance,
                            "locations": locations
                        }
                        return (
                            json.dumps(f"{key}: {relayout_data[key]}", indent=2),
                            True,
                            f"Region was added in adata.uns['region']['region_{last+1}']",
                            "success",
                        )
                    else:
                        # Open path as trajectory
                        points, indices, distance, locations = points_within_distance_along_path(
                            path,
                            spatial_coord,
                            distance_in_pixel
                        )
                        if 'trajectory' not in self.adata.uns:
                            self.adata.uns['trajectory'] = {}
                        last = len(self.adata.uns['trajectory'])
                        self.adata.uns['trajectory'][f"trajectory_{last+1}"] = {
                            "path": path,
                            "points": np.array(list(map(lambda x: [x.x, x.y], points))),
                            "indices": indices,
                            "distance": distance,
                            "locations": locations
                        }
                        return (
                            json.dumps(f"{key}: {relayout_data[key]}", indent=2),
                            True,
                            f'Trajectory was added in adata.uns["trajectory"]["trajectory_{last+1}"]',
                            "success",
                        )

            return no_update, False, "", "info"

        return app
