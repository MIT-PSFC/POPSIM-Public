import holoviews as hv
import panel as pn
import param

hv.extension("bokeh")


class CometMirrorGUI(param.Parameterized):
    add_var_selector = param.Action(default=lambda x: x.param.trigger("add_var_selector"), label="Add Plot")
    TIME_DIM = "time"
    SIMULATION_DIM = "simulation"
    ALL_SIMS = "all_simulations"

    def __init__(self, ds, **params):
        super().__init__(**params)
        self.ds = ds
        self.time = pn.widgets.DiscreteSlider(name="Time (s)", options=list(self.ds.time.values))
        self.simulation_selector = pn.widgets.MultiSelect(
            name="Simulation Case",
            options=[CometMirrorGUI.ALL_SIMS, *list(self.ds[self.SIMULATION_DIM].values)],
            value=[CometMirrorGUI.ALL_SIMS],
        )
        self.variable_selectors = pn.Column()
        self.plots = pn.GridBox(ncols=2)

    @param.depends("add_var_selector", watch=True)
    def add_var_selector_callback(self):
        var_selector = pn.widgets.MultiChoice(name="Choose Variables", options=list(self.ds.data_vars))
        remove_button = pn.widgets.Button(name="Remove", button_type="danger")
        remove_button.on_click(lambda event, widget=var_selector: self.remove_widget_callback(event, id(widget)))

        def make_plot(variables_list, time_value, selected_simulations):
            if not variables_list:  # If variables_list is empty, do nothing
                return
            data = (
                self.ds if selected_simulations == [CometMirrorGUI.ALL_SIMS] else self.ds.sel({self.SIMULATION_DIM: selected_simulations})
            )

            # Select the data for the selected variables
            data = data[variables_list]

            plots = []
            for var in variables_list:
                if "rho" in data[var].dims:  # Check if 'rho' is a dimension for the variable
                    # Select the slice for the specified time_value and plot with 'rho' on the x-axis
                    slice_data = data[var].sel({self.TIME_DIM: time_value})
                    plot = slice_data.hvplot.line(x="rho", label=var, by=CometMirrorGUI.SIMULATION_DIM)
                else:
                    # Plot with time on the x-axis for variables without 'rho' dimension
                    plot = data[var].hvplot.line(x=self.TIME_DIM, label=var, by=CometMirrorGUI.SIMULATION_DIM)
                plots.append(plot)

            overlay = hv.Overlay(plots).opts(title=",".join(variables_list), xlabel="Variable", ylabel="Value")

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
        template = pn.template.BootstrapTemplate(title="Bootstrap Template")
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
