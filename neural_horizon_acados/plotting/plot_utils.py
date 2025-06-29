import numpy as np

from typing import Optional, Literal
from types import MappingProxyType
from bokeh.plotting import figure
from bokeh.models import (
    Legend, GridPlot, Column, Row, ColumnDataSource, HoverTool, LegendItem, 
    GlyphRenderer, ColorBar, ButtonLike
)

from ..mpc.mpc_dataclass import AMPC_data, MPC_data



latex_labels = MappingProxyType({
    'N_MPC': r'$$M$$',
    'N_NN': r'$$N_{NN}$$',
    'N_hidden': r'$$n_{hidden}$$',
    'N_hidden_end': r'$$n_{hidden}$$',
    'Mean_Time': r'$$\bar{t}_{sol} \, [\mathrm{s}]$$',
    'Median_Time': r'$$\tilde{t}_{sol} \, [\mathrm{s}]$$',
    'Cost': r'Cost',
    'R2_score': r'$$R^2$$'
})


def latex_label_with_unit(label: str, unit: Optional[str] = None):
    """
    Generates a LaTeX-formatted label with an optional unit.

    Parameters
    ----------
    ``label`` : str
        The label to be formatted in LaTeX.
    ``unit`` : Optional[str], optional
        The unit to be appended to the label. If None, no unit is added. Default is None.

    Returns
    -------
    ``latex_label`` : str
        The LaTeX-formatted label with the unit if provided.
    """
    latex_label = latex_labels[label]
    left_bracket_index = latex_label.find('[')
    right_bracket_index = latex_label.find(']')
    if unit is None:
        return latex_label
    elif left_bracket_index != -1 and right_bracket_index != -1:
        return latex_label.replace(latex_label[left_bracket_index:right_bracket_index+1], fr'[\mathrm{{{unit}}}]')
    else:
        return fr'{latex_label[:-2]} \, [\mathrm{{{unit}}}]$$'



def exclude_buttons_figure(p: figure | Column | Row | GridPlot):
    """
    Removes button-like renderers and toolbars from a Bokeh figure or layout.

    Parameters
    ----------
    ``p`` : figure | Column | Row | GridPlot
        The Bokeh figure or layout to clean.

    Returns
    -------
    ``clean_p`` : figure | Column | Row | GridPlot
        The cleaned Bokeh figure or layout without button-like renderers and toolbars.
    """
    def clean_item(item):
        if isinstance(item, (Column, Row)):
            new_childs = []
            for it in item.children:
                new_item = clean_item(it)
                if new_item is not None:
                    new_childs.append(new_item)
            item.children = new_childs
            return item

        elif isinstance(item, GridPlot):
            new_childs = []
            for it in item.children:
                new_item = clean_item(it[0])
                if new_item is not None:
                    new_childs.append((new_item, *it[1:]))
            item.children = new_childs
            item.toolbar_location = None
            item.toolbar.logo = None
            return item
        
        elif isinstance(item, figure):
            new_renderers = [r for r in item.renderers if not isinstance(r, ButtonLike)]
            item.renderers = new_renderers
            item.toolbar_location = None
            item.toolbar.logo = None
            return item
        
        elif isinstance(item, (list, tuple)):
            new_items = [] 
            for it in item:
                new_it = clean_item(it)
                if new_it is not None:
                    new_items.append(new_it)
            return new_items 
            
    p = clean_item(p)

    return p


def pt2px_height(pt: int):
    """
    Converts a height value from points to pixels.

    Parameters
    ----------
    ``pt`` : int
        The height in points to be converted.

    Returns
    -------
    ``px_height`` : int
        The height in pixels.
    """
    return int(np.round(pt) * 1.52)


def get_legend_height(legend: Legend):
    """
    Calculates the height of a Bokeh legend in pixels.

    Parameters
    ----------
    ``legend`` : Legend
        The Bokeh legend for which the height is calculated.

    Returns
    -------
    ``height`` : int
        The height of the legend in pixels.
    """
    label_font_size = pt2px_height(int(legend.label_text_font_size[:-2]))
    title_font_size = pt2px_height(int(legend.title_text_font_size[:-2]))
    num_items = len(legend.items)
    if legend.ncols is None or legend.ncols == 1:
        # If ncols is not set or 1, calculate height for a single column
        height = num_items * max(label_font_size, legend.label_height, legend.glyph_height) + (num_items - 1) * legend.spacing
    else:
        # If ncols is set, calculate height based on the number of columns
        num_rows = (num_items + legend.ncols - 1) // legend.ncols
        height = num_rows * max(label_font_size, legend.label_height, legend.glyph_height) + (num_rows - 1) * legend.spacing

    height += 2 * (legend.padding + legend.margin)

    if legend.title:
        height += title_font_size + legend.spacing 

    return height



def get_legend_item_widths(legend: Legend):
    """
    Calculates the widths of items in a Bokeh legend.

    Parameters
    ----------
    ``legend`` : Legend
        The Bokeh legend for which the item widths are calculated.

    Returns
    -------
    ``widths`` : list of int
        A list of widths for each legend item in pixels.
    """
    widths = []
    legend_text_size = pt2px_height(int(legend.label_text_font_size[:-2]))
    for item in legend.items:
        label = item.label['value']
        widths.append(len(label) * legend_text_size)  # Estimate width by character length, adjust multiplier as needed
    return widths



def calculate_ncols(plot_width, legend: Legend, max_width_factor: float = 0.85):
    """
    Calculates the number of columns for a Bokeh legend to fit within the plot width.

    Parameters
    ----------
    ``plot_width`` : int
        The width of the plot in pixels.
    ``legend`` : Legend
        The Bokeh legend for which the number of columns is calculated.
    ``max_width_factor`` : float, optional
        The maximum width factor of the plot to be occupied by the legend.
         Default = 0.85.

    Returns
    -------
    ``ncols`` : int
        The number of columns for the legend.
    """
    item_widths = get_legend_item_widths(legend)
    available_width = int(plot_width * max_width_factor - 2 * (legend.padding + legend.margin))# Subtract a margin from the plot width for padding
    ncols = 1
    current_width = 0
    max_width = max(*item_widths, legend.label_width)
    for _ in range(len(item_widths)):
        add_width = max_width + legend.glyph_width + legend.spacing + legend.label_standoff
        if (current_width + add_width) < available_width:
            ncols += 1
            current_width += add_width
        else:
            break
    
    ncols = min(len(item_widths), ncols)
    return ncols



def get_figure_size(latex_text_width_pt = 497.92325, fraction: float = 1., ratio: Optional[float]=None):
    """
    Calculate the width and height of a figure based on a given fraction of the LaTeX text width.
    The height is determined using the golden ratio.

    Parameters:
    ``fraction``  : float, optional
        Fraction of the LaTeX text width (e.g., 0.5 for half the width)

    Returns:
    ``width, height`` : tuple[int, int] 
        Width and height of the figure in pixels
    """
    # Convert points to pixels (1 pt = 1.333 pixels)
    latex_text_width_px = latex_text_width_pt * 1.333

    # Calculate width and height based on the golden ratio
    width = latex_text_width_px * fraction

    if ratio is None:
        golden_ratio = (1 + 5 ** 0.5) / 2  # Approx 1.618
        height = width / golden_ratio
    else:
        height = width / ratio

    return int(width), int(height)



def style_single_figure(
        p: figure,
        line_color = 'black',
        text_color = 'black',
        font = 'serif',
        tick_font_size = '10pt',
        label_font_size = '12pt',
        line_width = 2,
        dot_size = 4,
        output_backend: Optional[Literal['svg', 'html']] = None,
        move_legend_top: bool = True,
    ):
    """
    Styles a Bokeh figure with specified aesthetic parameters.

    Parameters
    ----------
    ``p`` : figure
        The Bokeh figure to style.
    ``line_color`` : str, optional
        Color of the axis lines, ticks, and legend border. Default is 'black'.
    ``text_color`` : str, optional
        Color of the axis labels, tick labels, and legend text. Default is 'black'.
    ``font`` : str, optional
        Font style for axis labels, tick labels, and legend text. Default is 'serif'.
    ``tick_font_size`` : str, optional
        Font size for the axis tick labels. Default is '10pt'.
    ``label_font_size`` : str, optional
        Font size for the axis labels. Default is '12pt'.
    ``line_width`` : int, optional
        Width of the lines in the figure. 
         Default = 2.
    ``dot_size`` : int, optional
        Size of the dots in the figure. Default is 4.
    ``output_backend`` : Optional[Literal['svg', 'html']], optional
        Output backend for the figure rendering. Default is None.
    ``move_legend_top`` : bool, optional
        If True, moves the legend to the top of the figure. Default is True.
    """
    if output_backend is not None:
        p.output_backend = output_backend

    # Set background and grid style
    p.background_fill_color = "#fafafa"
    p.xgrid.grid_line_color = "#e0e0e0"
    p.ygrid.grid_line_color = "#e0e0e0"
    p.xgrid.grid_line_dash = [6, 4]
    p.ygrid.grid_line_dash = [6, 4]

    # Set font styles
    p.title.text_font = font
    p.xaxis.axis_label_text_font = font
    p.yaxis.axis_label_text_font = font
    p.xaxis.major_label_text_font = font
    p.yaxis.major_label_text_font = font
    p.axis.major_label_text_font_size = tick_font_size
    p.axis.axis_label_text_font_size = label_font_size
    p.axis.axis_line_color = line_color
    p.axis.major_tick_line_color = line_color
    p.axis.minor_tick_line_color = line_color
    p.axis.major_label_text_color = text_color
    p.axis.axis_label_text_color = text_color
    p.outline_line_color = None

    if hasattr(p.xaxis, 'group_text_font_size'):
        p.xaxis.group_text_font_style = 'normal' 
        p.xaxis.group_text_color = 'black'
        p.xaxis.group_text_font = font
        p.xaxis.group_text_font_size = tick_font_size

        p.xaxis.major_label_text_font_size = '0pt'

    # Set Line styles
    for renderer in p.renderers:
        if hasattr(renderer, 'glyph'):
            if hasattr(renderer.glyph, 'line_width'):
                renderer.glyph.line_width = line_width # Adjust the thickness as needed
            if hasattr(renderer.glyph, 'size'):
                renderer.glyph.size = dot_size

    # Customize legend
    if len(p.legend) > 0:
        legend_items = [LegendItem(label=item.label, renderers=item.renderers) for item in p.legend.items]
        new_legend = Legend(items=legend_items, location="top_left", orientation="horizontal")

        ## Customize legend
        new_legend.title = p.legend.title
        new_legend.title_text_font = font
        new_legend.title_text_font_style = "bold"
        new_legend.label_text_font = font
        new_legend.label_text_font_size = tick_font_size
        new_legend.label_text_color = text_color
        new_legend.border_line_color = line_color
        new_legend.border_line_width = 1
        new_legend.border_line_alpha = 0.8
        new_legend.background_fill_color = "#f0f0f0"
        new_legend.background_fill_alpha = 0.8

        new_legend.spacing = 5
        # new_legend.label_width = 100
        
        new_legend.ncols = calculate_ncols(p.width, new_legend)

        # Remove the original legend
        p.legend.visible = False

        p.height += get_legend_height(new_legend) - get_legend_height(p.legend)

        # Add the new legend above the plot
        p.add_layout(new_legend, 'above' if move_legend_top else 'below')

    cbar: ColorBar = p.select_one(ColorBar)
    if cbar is not None:
        cbar.major_label_text_font = font
        cbar.title_text_font = font
        cbar.major_label_text_color = text_color
        cbar.title_text_color = text_color
        cbar.major_label_text_font_size = tick_font_size
        cbar.title_text_font_size = label_font_size
        cbar.bar_line_color = line_color
        cbar.major_tick_line_color = line_color

        


def set_figure_to_default_latex(p: figure | Column | Row | GridPlot | list, **kwargs) -> None:
    """
    Applies default LaTeX styling to a Bokeh figure or layout, including nested layouts.

    Parameters
    ----------
    ``p`` : figure | Column | Row | GridPlot | list
        The Bokeh figure or layout to style. Can also be a list of figures or layouts.
    ``**kwargs`` : optional
        Additional keyword arguments passed to the `style_single_figure` function.
    """
    # Check the type of p and apply styling appropriately
    if isinstance(p, figure):
        style_single_figure(p, **kwargs)
    elif isinstance(p, (Column, Row)):
        for p_child in p.children:
            set_figure_to_default_latex(p_child, **kwargs)
    elif isinstance(p, GridPlot):
        for i, item in enumerate(p.children):
            p_child = item[0]
            set_figure_to_default_latex(p_child, **kwargs)

            if i < len(p.children)-1 and hasattr(p_child, 'xaxis') and hasattr(p_child, 'yaxis') and type(p_child) != Legend and type(p_child) != ColorBar:
                p_child.height -= pt2px_height(int(p_child.xaxis.major_label_text_font_size[:-2]))
                p_child.xaxis.major_tick_line_color = None
                p_child.xaxis.minor_tick_line_color = None
                p_child.xaxis.major_label_text_font_size = '0pt'

    elif isinstance(p, list):  # Handle lists of figures
        for item in p:
            set_figure_to_default_latex(item, **kwargs)




def plot_line(
        fig: figure, 
        x_data: list, 
        y_data: list, 
        line_alpha: float, 
        line_width: int, 
        line_dash: str, 
        color: str, 
        hover_tool: bool
    ) -> GlyphRenderer:
    """
    Plots a line on a Bokeh figure with optional hover tool.

    Parameters
    ----------
    ``fig`` : figure
        The Bokeh figure to plot the line on.
    ``x_data`` : list or array-like
        The x-coordinates of the line.
    ``y_data`` : list or array-like
        The y-coordinates of the line.
    ``line_alpha`` : float
        The transparency level of the line.
    ``line_width`` : int
        The width of the line.
    ``line_dash`` : str
        The dash pattern of the line.
    ``color`` : str
        The color of the line.
    ``hover_tool`` : bool
        If True, adds a hover tool to display the line's coordinates.

    Returns
    -------
    ``handle`` : GlyphRenderer
        The renderer for the plotted line.
    """
    source = ColumnDataSource(data=dict(x=x_data, y=y_data))
    handle = fig.line(
        x='x', y='y', source=source, line_alpha=line_alpha, line_width=line_width,
        line_dash=line_dash, color=color
    )
    if hover_tool:
        hover = HoverTool(tooltips=[("x, y", "@x, @y")], mode='mouse', renderers=[handle])
        fig.add_tools(hover)
    return handle






def find_Ts(MPC_result: AMPC_data | MPC_data, threshold=None) -> int | None:
    """
    Finds the time step index corresponding to the settling time of the closed-loop simulation results.

    Parameters
    ----------
    ``MPC_result`` : AMPC_data | MPC_data
        The MPC result object containing simulation data. This should include:
        - `X`: a 2D numpy array of predicted states.
        - `P`: an instance of the MPC_Param dataclass with problem parameters.
    ``threshold`` : float, optional
        The threshold value used to determine if the control input is considered close enough to the reference. If None, a default threshold of 5% of the state bounds is used. Default = None.

    Returns
    -------
    ``Ts_ind`` :  int | None
        The index of the last time step where the state is within the threshold, 
         or None if the state does not settle within the simulation time.
    """
    X,P = MPC_result.X, MPC_result.mpc_param
    if threshold is None:
        threshold = 0.05
        # use 5% of the xbnd as default
    threshold_values = ((P.ubx - P.lbx)*threshold).reshape((-1, 1))
    mask = (np.abs(X) < threshold_values).all(axis=0)
    Ts_ind = np.where(np.invert(mask))[0][-1]
    if Ts_ind < X.shape[1]-1:
        return Ts_ind
    else:
        return None