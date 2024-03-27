import typing

import holoviews as hv
import hvplot.xarray  # noqa: F401
import panel as pn
import param

hv.extension("bokeh")


class XarrayGUI(param.Parameterized):
    add_var_selector = param.Action(default=lambda x: x.param.trigger("add_var_selector"), label="Add Plot")
    ALL_SIMS = "all_simulations"

    def __init__(self, ds, time_dim: str, rho_dim: str, simulation_dim: typing.Optional[str] = None, **params):
        super().__init__(**params)
        self.ds = ds
        self.time_dim = time_dim
        self.rho_dim = rho_dim
        self.simulation_dim = simulation_dim
        self.time = pn.widgets.DiscreteSlider(name="Time (s)", options=list(self.ds.time.values))
        self.variable_selectors = pn.Column()
        self.plots = pn.GridBox(ncols=2)

        if self.simulation_dim:
            self.simulation_selector = pn.widgets.MultiSelect(
                name="Simulation Case",
                options=[XarrayGUI.ALL_SIMS, *list(self.ds[self.simulation_dim].values)],
                value=[XarrayGUI.ALL_SIMS],
            )
        else:
            self.simulation_selector = pn.widgets.StaticText(name="Simulation Case", value="")

    @param.depends("add_var_selector", watch=True)
    def add_var_selector_callback(self):
        var_selector = pn.widgets.MultiChoice(name="Choose Variables", options=list(self.ds.data_vars))
        remove_button = pn.widgets.Button(name="Remove", button_type="danger")
        remove_button.on_click(lambda event, widget=var_selector: self.remove_widget_callback(event, id(widget)))

        def make_plot(variables_list, time_value, selected_simulations):
            if not variables_list:  # If variables_list is empty, do nothing
                return

            data = (
                self.ds
                if (selected_simulations == [XarrayGUI.ALL_SIMS] or self.simulation_dim is None)
                else self.ds.sel({self.simulation_dim: selected_simulations})
            )

            # Select the data for the selected variables
            data = data[variables_list]

            # Define a list of colors, one for each variable
            colors = ["blue", "green", "red", "purple", "orange", "brown", "pink", "gray", "olive", "cyan"]
            color_map = dict(zip(variables_list, colors[: len(variables_list)]))  # Map each variable to a color

            plots = []

            def plot_variable(var, sim):
                if sim:
                    data_plot = data[var].sel({self.simulation_dim: sim})
                else:
                    data_plot = data[var]
                if self.rho_dim in data_plot.dims:  # Check if 'rho' is a dimension for the variable
                    # Select the slice for the specified time_value and plot with 'rho' on the x-axis
                    slice_data = data_plot.sel({self.time_dim: time_value})
                    plot = slice_data.hvplot.line(x=self.rho_dim, label=var, color=color_map[var])
                else:
                    # Plot with time on the x-axis for variables without 'rho' dimension
                    plot = data_plot.hvplot.line(x=self.time_dim, label=var, color=color_map[var])
                return plot

            for var in variables_list:
                if self.simulation_dim:
                    sim_plots_for_var = [plot_variable(var, sim) for sim in data[self.simulation_dim].values]
                else:
                    sim_plots_for_var = [plot_variable(var, None)]
                plots.extend(sim_plots_for_var)

            overlay = hv.Overlay(plots).opts(title=",".join(variables_list), xlabel="Variable", ylabel="Value", legend_position="right")

            # Add a vertical line at the time corresponding to the time_slider value
            vline = hv.VLine(time_value).opts(color="red", line_width=1.5)
            overlay = overlay * vline

            return overlay

        self.variable_selectors.append(pn.Row(var_selector, remove_button))
        self.plots.append(pn.bind(make_plot, var_selector, self.time, self.simulation_selector))

    def remove_widget_callback(self, event, widget_id):
        for i, row in enumerate(self.variable_selectors):
            if widget_id == id(row[0]):
                self.variable_selectors.pop(i)
                self.plots.pop(i)
                return

    def build_view(self):
        template = pn.template.BootstrapTemplate(title="POPSIM GUI")
        sidebar = pn.Column(
            self.simulation_selector,
            self.time,
            pn.WidgetBox(
                "## Plot Variables",
                self.param.add_var_selector,
                self.variable_selectors,
                sizing_mode="stretch_width",
            ),
        )
        main = pn.Column(self.plots)
        template.sidebar.append(sidebar)
        template.main.append(main)
        return template
