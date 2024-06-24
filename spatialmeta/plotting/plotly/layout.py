import plotly.graph_objects as go 
import dash
from dash import dcc
from dash import html
import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

from .defaults import *

layout = go.Layout(
    margin=go.layout.Margin(
        l=0, #left margin
        r=0, #right margin
        b=0, #bottom margin
        t=0, #top margin
    ),
    # paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis = {
        'showgrid': False,  # thin lines in the background
        'showgrid': False,  # thin lines in the background
        'zeroline': False,
        # thick line at x=0
        'tickfont': {'color': 'rgb(255,255,255)'}
    },
    showlegend=False,
    yaxis = {
        'showgrid': False,  # thin lines in the background
        'showgrid': False,  # thin lines in the background
        'zeroline': False,  # thick line at x=0)
        'tickfont': {'color': 'rgb(255,255,255)'}
    },
    modebar=go.layout.Modebar(
        orientation='v',
    ),
    newshape_line_color='cyan'
)

# Header
header = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Img(
                            id="logo",
                            src="https://dash.gallery/self-driving/assets/logo.png",
                            height="30px",
                        ),
                        md="auto",
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H3("Spatial Multiomics Visualization"),
                                ],
                                id="app-title",
                            )
                        ],
                        md="auto",
                        align="center",
                    ),
                    dbc.Col(
                        [
                            dbc.Alert(
                                "",
                                id="alert-auto",
                                is_open=True,
                                duration=4000,
                                color="info",
                            ),
                        ],
                        md="auto",
                        align="center",
                    ),
                ],
                align="center",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.NavbarToggler(id="navbar-toggler"),
                            dbc.Collapse(
                                dbc.Nav(
                                    [],
                                    navbar=True,
                                ),
                                id="navbar-collapse",
                                navbar=True,
                            ),
                        ],
                        md=2,
                    ),
                ],
                align="center",
            ),
        ],
        fluid=True,
    ),
    dark=False,
    color="light",
    sticky="top",
)

# left_sidebar
left_sidebar = lambda adata, init_feature: [
    dbc.Card(
        id="left_sidebar-card",
        children=[
            dbc.CardHeader("Display options"),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.InputGroupText("Display"),
                            dcc.Dropdown(
                                id="display-dropdown",
                                options=[
                                    {
                                        "label": "Feature expression (string)",
                                        "value": "expression",
                                    },
                                ]
                                + [
                                    {
                                        "label": f"{x} ({str(adata.obs[x].dtype)})",
                                        "value": x,
                                    }
                                    for x in list(
                                        filter(
                                            lambda x: is_string_dtype(adata.obs[x])
                                            or is_numeric_dtype(adata.obs[x]),
                                            adata.obs.columns,
                                        )
                                    )
                                ],
                                value="expression",
                            ),
                        ]
                    ),
                    html.Br(),
                    dbc.Row(
                        [
                            dbc.InputGroupText("Feature Name"),
                            html.Br(),
                            dcc.Dropdown(
                                options=[
                                    {"label": x, "value": x}
                                    for x in sorted(adata.var.index)
                                ],
                                id="feature_name_select",
                                value=(
                                    init_feature
                                    if init_feature is not None
                                    else "COL3A1"
                                ),
                            ),
                        ]
                    ),
                    html.Br(),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.InputGroupText("Image Opacity"),
                                    html.Br(),
                                    dcc.Slider(
                                        id="image_opacity_slider",
                                        min=0,
                                        max=100,
                                        step=1,
                                        value=100,
                                        marks={
                                            v: {
                                                "label": f"{v}%",
                                                "style": {"color": "#77b0b1"},
                                            }
                                            for v in [0, 100]
                                        },
                                        updatemode="drag",
                                    ),
                                ],
                            ),
                            dbc.Col(
                                [
                                    dbc.InputGroupText("Feature Opacity"),
                                    html.Br(),
                                    dcc.Slider(
                                        id="feature_opacity_slider",
                                        min=0,
                                        max=100,
                                        step=1,
                                        value=100,
                                        # tooltip={"placement": "bottom", "always_visible": True},
                                        marks={
                                            v: {
                                                "label": f"{v}%",
                                                "style": {"color": "#77b0b1"},
                                            }
                                            for v in [0, 100]
                                        },
                                        updatemode="drag",
                                    ),
                                ]
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.InputGroupText("Marker size"),
                            html.Br(),
                            dcc.Slider(
                                id="marker_size_slider",
                                min=0,
                                max=10,
                                step=1,
                                value=4,
                                # tooltip={"placement": "bottom", "always_visible": True},
                                marks={
                                    v: {"label": f"{v}%", "style": {"color": "#77b0b1"}}
                                    for v in [0, 10]
                                },
                                updatemode="drag",
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.InputGroupText("Maximum distance to line)"),
                            html.Br(),
                            dcc.Slider(
                                id="distance_slider",
                                min=0,
                                max=2000,
                                step=10,
                                value=1000,
                                marks={
                                    v: {"label": f"{v}μm", "style": {"color": "#77b0b1"}}
                                    for v in [0, 2000]
                                },
                                updatemode="drag",
                            ),
                        ],
                    ),
                ]
            ),
        ],
    ),
    html.Br(),
    dbc.Card(
        id="left-sidebar-tool-card",
        children=[
            dbc.CardHeader("Annotation Tools"),
            dbc.CardBody(
                [
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText("GroupName-"),
                            dbc.Input(id="annotation-group-name"),
                        ],
                        className="mb-3",
                    ),
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText("ClassName-"),
                            dbc.Input(id="annotation-class-name"),
                        ],
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Button(
                                "Annotate",
                                color="primary",
                                className="me-1",
                                id="annotation-submit",
                                n_clicks=0,
                            ),
                            html.P(id="annotation-msg"),
                        ]
                    ),
                ]
            ),
        ],
    ),
]
# right_sidebar
right_sidebar = lambda summary_fig, init_feature: [
    dbc.Card(
        id="right-sidebar-card-2",
        children=[
            dbc.CardHeader(f"Expression of {init_feature if init_feature is not None else 'COL3A1'}", id='feature-summary-header'),
            dbc.CardBody(
                [
                    html.Div(
                        id="selected-points",
                        children=[
                            html.P("No points selected, showing all data"),
                        ],
                    ),
                    # Wrap dcc.Loading in a div to force transparency when loading
                    html.Div(
                        id="transparent-loader-wrapper-2",
                        children=[
                            dcc.Loading(
                                id="expression-histogram-loading",
                                type="circle",
                                children=[
                                    # Graph
                                    dcc.Graph(
                                        id="expression_histogram",
                                        figure=summary_fig,
                                        config={
                                            "displayModeBar": False,
                                        },
                                    ),
                                ],
                            )
                        ],
                    ),
                    html.Div(
                        id="transparent-loader-wrapper-2",
                        children=[
                            dcc.Loading(
                                id="trajectpry-graph-loading",
                                type="circle",
                                children=[
                                    # Graph
                                    dcc.Graph(
                                        id="trajectory_graph",
                                        figure=summary_fig,
                                        config={
                                            "displayModeBar": False,
                                        },
                                    ),
                                ],
                            )
                        ],
                    ),
                ]
            ),
        ],
    ),
    dbc.Card(
        id="right-sidebar-card-3",
        children=[
            dbc.CardHeader("Selected Path"),
            dbc.CardBody(
                [
                    html.Pre(id="trajectory-pre")
                ]
            ),
        ],
    ),
]


image = lambda fig: [
    dbc.Card(
        id="image-card",
        children=[
            dbc.CardHeader("Viewer"),
            dbc.CardBody(
                [
                    # Wrap dcc.Loading in a div to force transparency when loading
                    html.Div(
                        id="transparent-loader-wrapper",
                        children=[
                            dcc.Loading(
                                id="image-loading",
                                type="circle",
                                children=[
                                    # Graph
                                    dcc.Graph(
                                        id="main_graph",
                                        figure=fig,
                                        config={
                                            "modeBarButtonsToAdd": [
                                                "drawopenpath",
                                                "drawclosedpath",
                                                "eraseshape",
                                            ],
                                            "modeBarButtonsToRemove": [
                                                "drawrect",
                                                "drawcircle",
                                            ]
                                        },
                                    ),
                                ],
                            )
                        ],
                    ),
                ]
            ),
        ],
    )
]
