import numpy as np

from itertools import chain
from typing import Callable, Optional, Literal
from collections import defaultdict
from bokeh.plotting import figure
from bokeh.models import (
    BoxAnnotation, Legend, AdaptiveTicker, PrintfTickFormatter, Range1d, CustomJS, RadioButtonGroup
)
from bokeh.layouts import gridplot
from bokeh.palettes import Category10_10

from ..mpc.mpc_dataclass import AMPC_data, MPC_data, dataclass_group_by
from .plot_utils import set_figure_to_default_latex, find_Ts, plot_line, calculate_ncols, pt2px_height



def plot_MPC_results(
    MPC_results: list[AMPC_data | MPC_data],
    time_type: Optional[Literal['acados', 'python']] = 'acados', 
    time_scale: str = 'log',
    additional_plots: Optional[list[Literal['Iterations', 'Prep_Time', 'Fb_Time', 'Prep_Iterations', 'Fb_Iterations']]] = None,
    additional_plots_options: Optional[dict[Literal['Iterations', 'Prep_Time', 'Fb_Time', 'Prep_Iterations', 'Fb_Iterations'], dict]] = None,  
    group_by: Callable[[AMPC_data | MPC_data], str] = lambda x: x.name,
    background_fill_color: str = "#FAFAFA",
    bnd_color: str = '#D55E00',
    width: int = 1200,
    height: int = 200,
    plot_mpc_trajectories = False,
    plot_Ts: bool = True,
    activate_hover: bool = False,
    threshold: float = None,
    xbnd: Optional[float] = None,
    ybnds: Optional[list[list[float]]] = None,
    legend_title: Optional[str] = None,
    legend_cols: Optional[int] = None,
    # parameters of the plots
    cols = Category10_10,
    dash: list = ['solid','solid','solid','dashed','dotdash'],
    thickness: list = [8,4,4,3,3],
    alpha: list = [.75,.5,1,1,1],
    latex_style: bool = False,
):
    """
    Generates a Bokeh plot of state and input trajectories for multiple MPC results.

    Parameters
    ----------
    ``MPC_results`` : list[AMPC_data | MPC_data]
        List of MPC result data objects containing simulation and trajectory data.
    ``time_type`` : Literal['acados', 'python'], optional
        The time data source for the solver iteration time plot. 
         Default = 'acados'.
    ``additional_plots`` : list[Literal['Iterations', 'Prep_Time', 'Fb_Time', 'Prep_Iterations', 'Fb_Iterations']], optional
        List of additional plots to include. 
         Default = None.
    ``additional_plots_options`` : dict[Literal['Iterations', 'Prep_Time', 'Fb_Time', 'Prep_Iterations', 'Fb_Iterations'], dict], optional
        Dictionary of additional plot options. 
         Default = None.
    ``group_by`` : Callable[[AMPC_data | MPC_data], str], optional
        Function to group MPC results by a specific attribute. 
         Default = lambda x: x.name.
    ``plot_mpc_trajectories`` : bool, optional
        Whether to plot MPC predicted trajectories. 
         Default = False.
    ``plot_Ts`` : bool, optional
        Whether to plot solver convergence times. 
         Default = True.
    ``activate_hover`` : bool, optional
        Whether to activate hover tools. 
         Default = False.
    ``threshold`` : float, optional
        Threshold for solver iteration time plot. 
         Default = None.
    ``legend_title`` : str, optional
        Title for the legend. 
         Default = None.
    ``width`` : int, optional
        Width of each plot. 
         Default = 1200.
    ``height`` : int, optional
        Height of each plot. 
         Default = 200.
    ``theta_bnd`` : float | Iterable[float], optional
        Boundary for the angle theta in radians. 
         Default = 1.5*np.pi.
    ``xbnd`` : float, optional
        Boundary for the x-axis. 
         Default = None.
    ``time_scale`` : str, optional
        The scaling of the y-axis for the solver iteration time plot. Can be 'log' or 'linear'. 
         Default = 'log'.
    ``background_fill_color`` : str, optional
        Background color of the plots. 
         Default = "#FAFAFA".
    ``bnd_color`` : str, optional
        Color for shading areas outside state boundaries. 
         Default = '#D55E00'.
    ``legend_cols`` : int, optional
        Number of columns in the legend. 
         Default = None.
    ``cols`` : list[str], optional
        List of colors for each MPC result. 
         Default = Category10_10.
    ``dash`` : list[str], optional
        List of line dash styles for each MPC result. 
         Default = ['solid', 'solid', 'solid', 'dashed', 'dotdash'].
    ``thickness`` : list[int], optional
        List of line widths for each MPC result. 
         Default = [8, 4, 4, 3, 3].
    ``alpha`` : list[float], optional
        List of line transparencies for each MPC result. 
         Default = [.75, .5, 1, 1, 1].
    ``latex_style`` : bool, optional
        Whether to apply LaTeX styling to the plots. 
         Default = False.

    Returns
    -------
    ``layout`` : bokeh.layouts.gridplot
        A vertical grid of Bokeh figures containing the MPC result plots.
    """
    # groups the results
    cMPC_groups: dict[str, list[AMPC_data | MPC_data]] = {key: list(group) for key, group in dataclass_group_by(MPC_results, group_by)}

    # styling attributes - in case there aren't enough, we will use default values
    c_cols = dict(zip(cMPC_groups.keys(), cols))
    l_dash = dict(zip(cMPC_groups.keys(), dash))
    l_thk = dict(zip(cMPC_groups.keys(), thickness))
    l_a = dict(zip(cMPC_groups.keys(), alpha))
    legend_it = defaultdict(list)
    
    # set of figures
    temp_MPC = MPC_results[0]
    P = temp_MPC.mpc_param
    nx, nu, x_rng, nt, dt = P.model_param.nx, P.model_param.nu, P.T_sim, P.N_sim, P.Ts
    ylabels_px, ylabel_pu = P.model_param.xlabel_latex, P.model_param.ulabel_latex
    xbnd = x_rng if xbnd is None else xbnd

    P.make_arrays_1d()

    if ybnds is None:
        bx_diffs = [np.abs(P.lbx[j] - P.ubx[j]) for j in range(nx)]
        ybnds = [[P.lbx[j] - bx_diff*0.1, P.ubx[j] + bx_diff*0.1] for j, bx_diff in enumerate(bx_diffs)]
        bu_diffs = [np.abs(P.lbu[j] - P.ubu[j]) for j in range(nu)]
        ybnds += [[P.lbu[j] - bu_diff*0.1, P.ubu[j] + bu_diff*0.1] for j, bu_diff in enumerate(bu_diffs)]

    
    activate_hover = activate_hover and len(MPC_results) < 20
    x_comb_range = Range1d(start=0, end=xbnd)
        
    px = [figure(background_fill_color=background_fill_color, width=width, height=height, x_range=x_comb_range, y_range=ybnds[i], y_axis_label=ylabels_px[i]) for i in range(nx)]
    pu = [figure(background_fill_color=background_fill_color, width=width, height=height, x_range=x_comb_range, y_range=ybnds[i+nx], y_axis_label=ylabel_pu[i]) for i in range(nu)]
    pt = figure(background_fill_color=background_fill_color, width=width, height=height, x_range=x_comb_range, y_axis_label=r'$$t_{sol} \, [s]$$', y_axis_type=time_scale)
    pt.yaxis.ticker = AdaptiveTicker(desired_num_ticks=3)
    pt.yaxis.formatter = PrintfTickFormatter(format='%e')

    if additional_plots is not None and additional_plots:
        possible_add_plot_names = ['Iterations', 'Prep_Time', 'Fb_Time', 'Prep_Iterations', 'Fb_Iterations']
        if all(add_plot_name not in possible_add_plot_names for add_plot_name in additional_plots):
            raise AttributeError(f'Given \"additional_plots\" contain a name which is not an attribute of AMPC_data or MPC_data classes! \
                             \nValid options are: {possible_add_plot_names} \nYou provided: {additional_plots}')  
        
        if additional_plots_options is None:
            additional_plots_options = {}
        
        pi = {add_plot_name: figure(
            background_fill_color=background_fill_color, 
            width=width, 
            height=height, 
            x_range=x_comb_range, 
            **additional_plots_options.get(add_plot_name, {})
        ) for add_plot_name in additional_plots}
    

    is_first_plot = True
    for group_name, c_list in cMPC_groups.items(): 
        for cMPC in c_list:
            x = [i*dt for i in range(nt)]
            xi = [[(j+i)*dt for i in range(cMPC.X_traj.shape[2])] for j in range(nt)]
            Ts_ind = find_Ts(cMPC ,threshold) if plot_Ts else None

            legend_handles = []


            ## STATE PLOTS
            for state_idx in range(nx):
                temp_l_alpha = l_a.get(group_name,1)
                temp_l_width = l_thk.get(group_name,2)
                temp_l_dash = l_dash.get(group_name,'solid')
                temp_color = c_cols.get(group_name,'#D55E00')

                if is_first_plot:
                    # plot reference and state bounds
                    px[state_idx].line(x=x, y=0, line_width=2, color='black', line_dash= 'dotted')
                    bnd_top_x = BoxAnnotation(bottom=P.ubx[state_idx], fill_alpha=0.4, fill_color=bnd_color)
                    bnd_bot_x = BoxAnnotation(top=P.lbx[state_idx], fill_alpha=0.4, fill_color=bnd_color)
                    px[state_idx].add_layout(bnd_top_x)
                    px[state_idx].add_layout(bnd_bot_x)
                    

                # open loop state trajectories
                if plot_mpc_trajectories:
                    state_traj_handle = px[state_idx].multi_line(
                        xs=xi,
                        ys=[cMPC.X_traj[i,state_idx,:] for i in range(nt)],
                        line_alpha=temp_l_alpha,
                        line_width=temp_l_width//2,
                        color=temp_color
                    )
                    legend_handles.append(state_traj_handle)
                
                # closed loop states plot
                state_handle = plot_line( 
                    px[state_idx], x, cMPC.X[state_idx,:],
                    line_alpha=temp_l_alpha,
                    line_width=temp_l_width,
                    line_dash=temp_l_dash,
                    color=temp_color,
                    hover_tool=activate_hover,
                )
                legend_handles.append(state_handle)
                
                # broken state cap point
                if np.isnan(cMPC.X[state_idx,-1]): 
                    last_pos = (~np.isnan(cMPC.X)).sum(axis = 1) - 1
                    broken_state_handle = px[state_idx].scatter(
                        x = x[last_pos[state_idx]],
                        y = cMPC.X[state_idx,last_pos[state_idx]],
                        size = 10, 
                        color = temp_color,
                        marker='star',
                    )
                    legend_handles.append(broken_state_handle)

                # settling point
                if Ts_ind is not None:
                    ts_handle = px[state_idx].scatter(
                        x = x[Ts_ind], 
                        y = cMPC.X[state_idx,Ts_ind],
                        size = 10,
                        color = temp_color
                    )
                    legend_handles.append(ts_handle)
            
            
            ## INPUT PLOT
            for input_idx in range(nu):
                if is_first_plot:
                    bnd_top_u = BoxAnnotation(bottom=P.ubu[input_idx], fill_alpha=0.4, fill_color=bnd_color)
                    bnd_bot_u = BoxAnnotation(top=P.lbu[input_idx], fill_alpha=0.4, fill_color=bnd_color)
                    pu[input_idx].add_layout(bnd_top_u)
                    pu[input_idx].add_layout(bnd_bot_u)

                # open loop input trajectories
                if plot_mpc_trajectories:
                    ui = [[(j+i)*dt for i in range(cMPC.U_traj.shape[2])] for j in range(nt)]
                    input_traj_handle = pu[input_idx].multi_line(
                        xs=ui,
                        ys=[cMPC.U_traj[i,input_idx,:] for i in range(nt)],
                        line_alpha=temp_l_alpha,
                        line_width=temp_l_width//2,
                        color=temp_color
                    )
                    legend_handles.append(input_traj_handle)
                
                # closed loop inputs plots
                input_handle = plot_line(
                    pu[input_idx], x, cMPC.U[input_idx], 
                    line_alpha=temp_l_alpha,
                    line_width=temp_l_width,
                    line_dash=temp_l_dash,
                    color=temp_color,
                    hover_tool=activate_hover,
                )
                legend_handles.append(input_handle)
                
                # broken input cap point
                if np.isnan(cMPC.U[input_idx,-1]): 
                    last_pos = (~np.isnan(cMPC.U)).sum(axis = 1) - 1
                    broken_input_handle = pu[input_idx].scatter(x=x[last_pos[0]], y=cMPC.U[0,last_pos[0]], size=10, color=temp_color, marker='star')
                    legend_handles.append(broken_input_handle)


            # SOLVER TIMES
            soltime_handle = plot_line(
                pt, x, cMPC.Acados_Time if isinstance(cMPC, AMPC_data) and time_type == 'acados' else cMPC.Time,
                line_alpha=temp_l_alpha,
                line_width=temp_l_width,
                line_dash=temp_l_dash,
                color=temp_color,
                hover_tool=activate_hover,
            )
            legend_handles.append(soltime_handle)


            # ADDITIONAL PLOTS
            if additional_plots is not None:
                for add_plot_name in additional_plots:
                    if hasattr(cMPC, add_plot_name):
                        add_plot_handle = plot_line(
                            pi[add_plot_name], x, getattr(cMPC, add_plot_name),
                            line_alpha=temp_l_alpha,
                            line_width=temp_l_width,
                            line_dash=temp_l_dash,
                            color=temp_color,
                            hover_tool=activate_hover,
                        )
                        legend_handles.append(add_plot_handle)

                        if plot_mpc_trajectories and hasattr(cMPC, f'{add_plot_name}_traj'):
                            value = getattr(cMPC, f'{add_plot_name}_traj')
                            ii = [[(j+i)*dt for i in range(value.shape[1])] for j in range(nt)]
                            add_plot_traj_handle = pi[add_plot_name].multi_line(
                                xs=ii,
                                ys=[value[i,:] for i in range(nt)],
                                line_alpha=temp_l_alpha,
                                line_width=temp_l_width//2,
                                color=temp_color
                            )
                            legend_handles.append(add_plot_traj_handle)

            legend_it[group_name].extend(legend_handles)

            is_first_plot = False
    

    # LEGEND
    legend_it = list(legend_it.items())
    legend = Legend(items=legend_it, orientation="horizontal", click_policy="hide", title=legend_title)
    legend.ncols = calculate_ncols(width, legend) if legend_cols is None else legend_cols
    nrows = (len(legend.items) + legend.ncols - 1) // legend.ncols
    estim_legend_height = legend.spacing * (nrows - 1) + max(legend.label_height, legend.glyph_height) * nrows + 2 * legend.padding + 2 * legend.margin # get_legend_height(legend)

    p_add = [*px, *pu, pt] if time_type is not None else [*px, *pu]
    if additional_plots is not None:
        p_add.extend([pi[add_plot_name] for add_plot_name in additional_plots])

    # change the height of last figure and add legend
    p_add[-1].xaxis.axis_label = r'$$t_{sim}$$'
    p_add[-1].height += estim_legend_height + p_add[-1].xaxis.axis_label_standoff + pt2px_height(int(p_add[-1].xaxis.axis_label_text_font_size[:-2]))
    p_add[-1].add_layout(legend, 'below')
    

    for ii in range(len(p_add)-1):
        p_add[ii].xaxis.major_tick_line_color = None
        p_add[ii].xaxis.minor_tick_line_color = None
        p_add[ii].xaxis.major_label_text_font_size = '0pt'


    # SELECTION BUTTON
    select = RadioButtonGroup(labels=["Show all", "Hide all"], active=0)
    all_renderers = list(chain.from_iterable((r[1] for r in legend_it)))
    callback = CustomJS(args=dict(renderers=all_renderers), code="""
        var visibility = cb_obj.active == 0;
        for (var i = 0; i < renderers.length; i++) {
            renderers[i].visible = visibility;
        }
    """)
    select.js_on_change('active', callback)
    p = [select] + p_add


    layout = gridplot([[x] for x in p])
    if latex_style:
        set_figure_to_default_latex(layout, move_legend_top=False, output_backend='svg')

    return layout
