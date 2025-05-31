# --- START OF FILE graph_visualization_qt.py ---
import matplotlib.pyplot as plt
import numpy as np
import logging
import io      
import cv2 # Only needed if get_frame_as_array uses cv2.imdecode, which it will

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GraphVisualizerQt: # Renamed to avoid conflict if old one is in same dir
    def __init__(self, processor):
        self.processor = processor
        self.figure = None # Will be set by MainAppWindow
        self.ax = None     # Will be set by MainAppWindow
        self.lines = {} 
        self.full_data_cache = {}
        self.active_legend = None
        self.default_dpi = 100 
        self.current_time_indicator_on_graph = None # Store ref to the axvline on graph

    def set_figure_axes(self, fig, ax):
        """Called by the main Qt app to provide the drawing context."""
        self.figure = fig
        self.ax = ax
        # Initial setup of axes properties if needed, or rely on plot_tooth_lines
        if self.ax:
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Average Force (N)")
            self.ax.set_title("Average Bite Force Over Time")
            self.ax.grid(True)
            if self.processor.timestamps and len(self.processor.timestamps) > 0:
                self.ax.set_xlim(self.processor.timestamps[0], self.processor.timestamps[-1])
            else: 
                self.ax.set_xlim(0, 1) 
            # Ylim will be set more dynamically in plot_tooth_lines or update
    def create_graph_figure(self, figsize=(10, 4)): # figsize can be adjusted
        """Creates or clears the Matplotlib figure and axes, and sets initial properties."""
        if self.figure is None or self.ax is None:
            self.figure, self.ax = plt.subplots(figsize=figsize, dpi=self.default_dpi)
            logging.info("Matplotlib figure and axes created.")
        else:
            self.ax.clear() 
            self.lines.clear() # Clear line references
            self.full_data_cache.clear() # Clear data cache
            if self.active_legend:
                try: self.active_legend.remove()
                except AttributeError: pass # May have been removed by ax.clear()
                self.active_legend = None
            logging.info("Matplotlib axes cleared for new plot content.")
        
        # Set fixed X-axis limits based on the entire dataset
        if self.processor.timestamps and len(self.processor.timestamps) > 0:
            self.ax.set_xlim(self.processor.timestamps[0], self.processor.timestamps[-1])
            logging.info(f"Graph X-LIM set to: ({self.processor.timestamps[0]:.2f}, {self.processor.timestamps[-1]:.2f})")
        else: 
            self.ax.set_xlim(0, 1) 
            logging.warning("Graph X-LIM: No timestamps, defaulting to (0,1)")
            
        # --- SET Y-AXIS LIMITS BASED ON OVERALL DATA RANGE ---
        max_y_force = 10.0 # Default minimum sensible top limit
        if self.processor.force_matrix is not None and self.processor.force_matrix.size > 0:
            # Use the pre-calculated max_force_overall from DataProcessor if available and valid
            if hasattr(self.processor, 'max_force_overall') and self.processor.max_force_overall > 0:
                max_y_force = self.processor.max_force_overall
            else: # Fallback: calculate from force_matrix if necessary
                fm_for_ylim = self.processor.force_matrix
                if not np.issubdtype(fm_for_ylim.dtype, np.floating):
                    try: fm_for_ylim = fm_for_ylim.astype(float)
                    except ValueError: fm_for_ylim = np.array([[100.0]], dtype=float) # Default if cast fails
                
                valid_forces = fm_for_ylim[~np.isnan(fm_for_ylim)]
                if valid_forces.size > 0:
                    max_y_force = np.max(valid_forces)
        
        self.ax.set_ylim(bottom=-max_y_force*0.05, top=max_y_force * 1.1 if max_y_force > 0 else 10)
        logging.info(f"Graph Y-LIM set to: ({-max_y_force*0.05:.2f}, {max_y_force * 1.1 if max_y_force > 0 else 10:.2f}) based on max_y_force={max_y_force:.2f}")
        # --- END Y-AXIS LIMITS ---
        
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Average Force (N)")
        self.ax.set_title("Average Bite Force Over Time") # Initial title
        self.ax.grid(True)
        return self.figure, self.ax
    
    def plot_tooth_lines(self, tooth_ids_to_display):
        if self.ax is None:
            logging.error("GraphVisualizerQt: Axes not set. Call create_graph_figure first.")
            self.create_graph_figure() 
            if self.ax is None: return 

        for line_artist in list(self.ax.lines): # Use list() for safe removal while iterating
            if hasattr(line_artist, 'get_gid') and line_artist.get_gid() == "graph_time_indicator_live":
                continue # Don't remove the main time indicator if it's managed here
            line_artist.remove()
        self.lines.clear(); self.full_data_cache.clear()
        if self.active_legend:
            try: self.active_legend.remove()
            except AttributeError: pass
            self.active_legend = None

        if not tooth_ids_to_display:
            title_suffix = "(No tooth selected)"
            self.ax.set_title(f"Average Bite Force Over Time {title_suffix}")
            self.ax.set_ylim(0, 1) # Default sensible Y range if no data
            if self.figure: self.figure.canvas.draw_idle()
            return

        max_y_for_current_selection = 0.0 # Start with 0
        min_y_for_current_selection = 0.0
        has_data_for_ylim = False

        for tooth_id in tooth_ids_to_display: # Pre-fetch all data to determine overall Y range
            _ , forces = self.processor.get_average_force_for_tooth(tooth_id)
            if forces.size > 0:
                has_data_for_ylim = True
                current_max = np.nanmax(forces)
                current_min = np.nanmin(forces)
                if np.isfinite(current_max): max_y_for_current_selection = max(max_y_for_current_selection, current_max)
                if np.isfinite(current_min): min_y_for_current_selection = min(min_y_for_current_selection, current_min)
        
        if not has_data_for_ylim or max_y_for_current_selection == 0: # If no data or all forces are zero
            top_y_limit = 10.0 # Default if no data or all zero
            bottom_y_limit = -0.5
        else:
            top_y_limit = max_y_for_current_selection * 1.1
            bottom_y_limit = min_y_for_current_selection - (max_y_for_current_selection * 0.05) # Some space below min
            if bottom_y_limit > -0.1: bottom_y_limit = -0.1 # Ensure 0 is visible

        self.ax.set_ylim(bottom=bottom_y_limit, top=top_y_limit)
        logging.info(f"Graph Y-LIM updated for selection: Bottom {bottom_y_limit:.2f}, Top {top_y_limit:.2f}")

        num_lines = len(tooth_ids_to_display)
        colors = plt.cm.viridis(np.linspace(0,1,max(1,num_lines)))

        for i, tooth_id in enumerate(tooth_ids_to_display):
            full_times, full_forces = self.processor.get_average_force_for_tooth(tooth_id) # Fetch again for cache
            self.full_data_cache[tooth_id] = (full_times, full_forces)
            # Initially plot empty; update_graph_to_timestamp will fill them
            line, = self.ax.plot([], [], label=f"Tooth {tooth_id}", color=colors[i % len(colors)], lw=1.5)
            self.lines[tooth_id] = line
        
        self.active_legend = self.ax.legend(loc='upper right')
        title_suffix = f"(Teeth: {', '.join(map(str, tooth_ids_to_display))})"
        self.ax.set_title(f"Average Bite Force Over Time {title_suffix}")
        
        logging.info("Graph lines plotted for teeth: %s", tooth_ids_to_display)
        if self.figure: self.figure.canvas.draw_idle()

    def update_graph_to_timestamp(self, current_timestamp, tooth_ids_currently_plotted):
        if self.figure is None or self.ax is None: return
        # logging.debug(f"GRAPH: Updating to T={current_timestamp:.2f} for teeth {tooth_ids_currently_plotted}") # General call log
        changed_data_for_frame = False # Flag to see if any line data was actually set

        for tooth_id in tooth_ids_currently_plotted: 
            if tooth_id in self.lines and tooth_id in self.full_data_cache:
                full_times, full_forces = self.full_data_cache[tooth_id]
                if full_times is not None and len(full_times) > 0:
                    idx_up_to_time = np.searchsorted(full_times, current_timestamp, side='right')
                    times_to_plot = full_times[:idx_up_to_time]
                    forces_to_plot = full_forces[:idx_up_to_time]
                    
                    if len(times_to_plot) > 0 : 
                        logging.debug(f"GRAPH_DATA Tooth {tooth_id} @ T={current_timestamp:.2f}: "
                                    f"Plotting {len(times_to_plot)} points. "
                                    f"X range: ({times_to_plot[0]:.2f} to {times_to_plot[-1]:.2f}), "
                                    f"Y range: ({np.min(forces_to_plot):.2f} to {np.max(forces_to_plot):.2f})")
                        self.lines[tooth_id].set_data(times_to_plot, forces_to_plot)
                        changed_data_for_frame = True
                    elif self.lines[tooth_id].get_xdata().size > 0 or self.lines[tooth_id].get_ydata().size > 0: 
                        # If line had data before but now should be empty
                        logging.debug(f"GRAPH_DATA Tooth {tooth_id} @ T={current_timestamp:.2f}: No data to plot (setting to empty).")
                        self.lines[tooth_id].set_data([], [])
                        changed_data_for_frame = True
                    # else: line is already empty and no new data, no change needed
                else: # full_times or full_forces is None or empty
                    self.lines[tooth_id].set_data([], []) 
                    if self.lines[tooth_id].get_xdata().size > 0: changed_data_for_frame = True # It became empty
            else:
                logging.warning(f"GRAPH_DATA: Tooth {tooth_id} not in self.lines or self.full_data_cache.")
        
        # The figure.canvas.draw_idle() is called in MainAppWindow.animation_step
        # if changed_data_for_frame and self.figure:
        #     logging.debug("GRAPH_DATA: Requesting canvas draw_idle due to data change.")
        #     self.figure.canvas.draw_idle()
    
    def update_time_indicator(self, current_timestamp):
        """Updates or creates the vertical time indicator line."""
        if not self.ax or not self.figure: return

        if self.current_time_indicator_on_graph:
            try: self.current_time_indicator_on_graph.remove()
            except (ValueError, AttributeError): pass # Already removed or None
            self.current_time_indicator_on_graph = None
        
        if current_timestamp is not None:
            self.current_time_indicator_on_graph = self.ax.axvline(
                current_timestamp, color='grey', linestyle=':', lw=1, gid="graph_time_indicator_live"
            )
        # Figure redraw will be handled by the main animation loop via draw_idle() on canvas

    def get_frame_as_array(self, current_timestamp, tooth_ids_to_display):
        if self.figure is None or self.ax is None:
            logging.warning("Graph figure not initialized for get_frame_as_array.")
            return None # Cannot generate frame

        # Ensure graph is updated to the specific timestamp for the screenshot
        # This might redraw lines if tooth_ids_to_display changed from current state
        if set(tooth_ids_to_display) != set(self.lines.keys()):
            self.plot_tooth_lines(tooth_ids_to_display) 
            
        self.update_graph_to_timestamp(current_timestamp, tooth_ids_to_display)
        self.update_time_indicator(current_timestamp) # Ensure indicator is on for screenshot
        
        self.figure.canvas.draw() # Ensure all drawing commands are processed
        
        buf = io.BytesIO()
        try:
            self.figure.savefig(buf, format='png', dpi=self.default_dpi, bbox_inches='tight', facecolor=self.figure.get_facecolor())
            buf.seek(0)
            img_array_png = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            frame_bgr = cv2.imdecode(img_array_png, cv2.IMREAD_COLOR)
        except Exception as e:
            logging.error(f"Error saving Matplotlib figure to buffer: {e}")
            frame_bgr = None
        finally:
            buf.close()
            if self.current_time_indicator_on_graph: # Clean up indicator used for screenshot
                try: self.current_time_indicator_on_graph.remove()
                except (ValueError, AttributeError): pass
                self.current_time_indicator_on_graph = None

        if frame_bgr is None: logging.error("Failed to decode Matplotlib figure to image array.")
        return frame_bgr
# --- END OF FILE graph_visualization_qt.py ---