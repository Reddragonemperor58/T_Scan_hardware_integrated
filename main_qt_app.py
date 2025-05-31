# --- START OF FILE main_qt_app.py ---
import sys
import logging
import time
import numpy as np
import cv2 
import atexit
import os

# ... (Qt imports as before) ...
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import QTimer, Qt
import vedo
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


from data_acquisition import SensorDataReader
from data_processing import DataProcessor
from graph_visualization_qt import GraphVisualizerQt 
from points_array import PointsArray # Import for potential direct use or reference
from hardware_grid_visualizer_qt import HardwareGridVisualizerQt # New visualizer
from hardware_3d_bar_visualizer_qt import Hardware3DBarVisualizerQt # New 3D bar from HW data

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from vedo import Plotter, settings # Import base Plotter

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

_main_app_window_instance_for_atexit = None
def cleanup_on_exit(): # ... (same as before) ...
    global _main_app_window_instance_for_atexit
    if _main_app_window_instance_for_atexit and hasattr(_main_app_window_instance_for_atexit, 'video_writer'):
        if _main_app_window_instance_for_atexit.video_writer is not None and _main_app_window_instance_for_atexit.video_writer.isOpened():
            logging.info("ATEIXT: Releasing OpenCV video writer..."); _main_app_window_instance_for_atexit.video_writer.release(); _main_app_window_instance_for_atexit.video_writer = None
            logging.info("ATEIXT: OpenCV video writer released.")
atexit.register(cleanup_on_exit)

class VedoQtCanvas(QVTKRenderWindowInteractor): # Same as before
    def __init__(self, parent=None): 
        super().__init__(parent)
        # --- ADD THESE LINES ---
        # Ensure the interactor and render window are initialized.
        # This might be done implicitly by Plotter(qt_widget=self) later,
        # but being explicit can sometimes help.
        if self.GetRenderWindow() and self.GetRenderWindow().GetInteractor():
            self.GetRenderWindow().GetInteractor().Initialize()
            # self.Start() # Start is usually for the blocking event loop, not always needed here
            # when Qt's event loop is primary. Try with and without self.Start().
            # If self.Start() blocks, then it's not right here.
        else:
            logging.warning("VedoQtCanvas: RenderWindow or Interactor not immediately available after super().__init__")
        # --- END ADD ---

    def GetPlotter(self, **kwargs_for_plotter): 
        plt = Plotter(qt_widget=self, **kwargs_for_plotter)
        # After plotter is created, it has initialized the render window and interactor.
        # It's good to ensure the interactor is started if not done automatically by Plotter.
        # This is usually done by the Qt event loop when the widget is shown.
        # if plt.interactor and not plt.interactor.GetInitialized(): # Check if already initialized
        #     plt.interactor.Initialize()
        #     # plt.interactor.Start() # Not here, Qt's loop runs it
        return plt    
    def closeEvent(self, event): self.Finalize(); super().closeEvent(event)

class EmbeddedVedoMultiViewWidget(QWidget):
    def __init__(self, processor_instance,
                 HardwareGridVisualizerClass, # Assuming this is the first Vedo visualizer
                 Hw3DBarVisualizerClass,          # Assuming this is the second Vedo visualizer
                 parent_main_window,          # Changed from parent=None
                 plotter_kwargs=None):
        super().__init__(parent_main_window) # Pass parent to QWidget
        if plotter_kwargs is None: plotter_kwargs = {}

        self.vlayout = QVBoxLayout(self); self.vlayout.setContentsMargins(0,0,0,0)
        self.vedo_canvas = VedoQtCanvas(self); self.vlayout.addWidget(self.vedo_canvas)
        
        # Default arguments for the main plotter
        plotter_creation_args = {'shape':(1,2), 'sharecam':False} 
        
        # Update with any passed plotter_kwargs (this will include 'title' if provided)
        if plotter_kwargs: 
            plotter_creation_args.update(plotter_kwargs) 
        
        # If 'title' was not in plotter_kwargs, set a default one
        if 'title' not in plotter_creation_args:
            plotter_creation_args['title'] = "Dental Visualizations" # Default title

        # --- CORRECTED CALL ---
        self.main_plotter = self.vedo_canvas.GetPlotter(**plotter_creation_args) 
        # --- END CORRECTION ---
        
        if not self.main_plotter or len(self.main_plotter.renderers) < 2:
            logging.error("Failed to create main Vedo Plotter with 2 sub-renderers."); return

        # Pass the main plotter and renderer index to visualizers
        self.grid_visualizer = HardwareGridVisualizerClass(processor_instance, self.main_plotter, 0)
        self.bar_visualizer = Hw3DBarVisualizerClass(processor_instance, self.main_plotter, 1) # Use new class
        
        # Link MainAppWindow for callbacks
        if hasattr(self.grid_visualizer, 'set_main_app_window_ref'):
            self.grid_visualizer.set_main_app_window_ref(parent_main_window)
        if hasattr(self.bar_visualizer, 'set_main_app_window_ref'):
            self.bar_visualizer.set_main_app_window_ref(parent_main_window)

        # Call setup_scene after visualizers are fully initialized and linked
        if hasattr(self.grid_visualizer, 'setup_scene'): self.grid_visualizer.setup_scene()
        if hasattr(self.bar_visualizer, 'setup_scene'): self.bar_visualizer.setup_scene()
        
        if hasattr(self.grid_visualizer, 'tooth_cell_definitions') and self.grid_visualizer.tooth_cell_definitions:
            if hasattr(processor_instance, 'calculate_cof_trajectory'):
                 processor_instance.calculate_cof_trajectory(self.grid_visualizer.tooth_cell_definitions)

        if hasattr(self, '_dispatch_mouse_click'): # Check if method exists
            self.main_plotter.add_callback('mouse click', self._dispatch_mouse_click)
        
        self.Render() # Initial render
    
    def _dispatch_mouse_click(self, event): # event is vedo.interaction.Event
        if not event: return

        picked_actor = event.actor
        
        # event.at gives the renderer index for subplot clicks
        renderer_index_of_click = getattr(event, 'at', None)

        logging.debug(f"DISPATCH_CLICK: Event received! Actor: {picked_actor.name if picked_actor else 'None'}. "
                      f"Clicked Renderer Index (event.at): {renderer_index_of_click}.")
        
        if renderer_index_of_click is not None:
            if renderer_index_of_click == self.grid_visualizer.renderer_index: # Assuming visualizers store their index
                logging.debug("Dispatching click to Grid Visualizer (matched event.at).")
                if hasattr(self.grid_visualizer, '_on_mouse_click'):
                    self.grid_visualizer._on_mouse_click(event) # Pass original event
                return 
            elif renderer_index_of_click == self.bar_visualizer.renderer_index:
                logging.debug("Dispatching click to 3D Bar Visualizer (matched event.at).")
                if hasattr(self.bar_visualizer, '_on_mouse_click'):
                    self.bar_visualizer._on_mouse_click(event) # Pass original event
                return
            # else: # Click was in a renderer index not assigned or out of bounds
                # logging.warning(f"Click in renderer index {renderer_index_of_click}, but no visualizer assigned.")
        
        # Fallback if event.at was None (e.g. click outside any specific renderer viewport but still in window)
        # OR if an actor was picked whose renderer couldn't be determined via event.at
        # This part is less likely to be hit if event.at is reliable for subplots.
        logging.info("Click not dispatched to a specific sub-renderer via event.at. Treating as general deselect.")
        
        # General deselect logic (as before, ensuring event.actor is None for visualizer handlers)
        original_actor_for_fallback = event.actor 
        event.actor = None 
        if hasattr(self.grid_visualizer, '_on_mouse_click'):
            self.grid_visualizer._on_mouse_click(event)
        if hasattr(self.bar_visualizer, '_on_mouse_click'):
            self.bar_visualizer._on_mouse_click(event)
        event.actor = original_actor_for_fallback
        

    def update_views(self, timestamp, latest_hardware_flat_data=None, sensitivity=1): # Add data args
        if self.grid_visualizer and hasattr(self.grid_visualizer, 'animate'):
            self.main_plotter.at(0) 
            self.grid_visualizer.animate(timestamp, latest_hardware_flat_data, sensitivity) # Pass data
        if self.bar_visualizer and hasattr(self.bar_visualizer, 'animate'):
            self.main_plotter.at(1) 
            self.bar_visualizer.animate(timestamp, latest_hardware_flat_data, sensitivity) # Pass data here too
        if hasattr(self.vedo_canvas, 'Render') and self.vedo_canvas.isVisible(): self.vedo_canvas.Render()
        elif self.main_plotter: self.main_plotter.render()

        if self.vedo_canvas and hasattr(self.vedo_canvas, 'GetRenderWindow') and self.vedo_canvas.GetRenderWindow():
            # logging.debug(f"EMBEDDED: Calling Render on vedo_canvas for {self.main_plotter.title}")
            self.vedo_canvas.GetRenderWindow().Render() # More direct VTK render call               # logging.debug(f"EMBEDDED: Calling Render on vedo_canvas for {self.main_plotter.title}")

    def get_frame_as_array(self, timestamp, latest_hardware_flat_data=None, sensitivity=1):
        # 1. Ensure both visualizers update their scenes for this timestamp
        self.update_views(timestamp, latest_hardware_flat_data, sensitivity) 
        
        # 2. Now that scenes are updated, take a single screenshot of the main plotter
        if self.main_plotter and self.main_plotter.window:
            # It might be necessary to ensure the main plotter itself renders once *after*
            # both sub-renderers have updated their actors and had their cameras reset.
            # The self.vedo_canvas.Render() in update_views should handle this.
            return self.main_plotter.screenshot(asarray=True)
        return None
    
    def Render(self): # Expose Render method of the canvas
        if hasattr(self.vedo_canvas, 'Render'): self.vedo_canvas.Render()
        elif self.main_plotter: self.main_plotter.render()

    def get_grid_visualizer(self): return self.grid_visualizer
    def get_bar_visualizer(self): return self.bar_visualizer


class MatplotlibCanvas(FigureCanvas): # ... (same as before) ...
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi); self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig); self.setParent(parent)

class MainAppWindow(QMainWindow):
    def __init__(self, processor, hw_data_source=None): 
        super().__init__()
        self.processor = processor
        self.hw_data_source = hw_data_source 
        self.current_timestamp_idx = 0
        self.animation_timer = QTimer(self)
        self.is_animating = False
        self.graph_time_indicator = None
        self.setWindowTitle("Dental Force Visualization Suite (PyQt)")
        self.setGeometry(50, 50, 1800, 960) 
        self.initial_graph_teeth = []
        self.currently_graphed_tooth_ids = []
        self.last_animated_timestamp = None 
        self.output_video_filename="composite_dental_animation.mp4" 
        self.canvas_width=1920; self.canvas_height=1080 
        self.fps = 10 
        self.video_writer = None 
        
        global _main_app_window_instance_for_atexit
        _main_app_window_instance_for_atexit = self 
        
        # 1. Matplotlib Graph Setup
        self.graph_qt_canvas = MatplotlibCanvas(self) # Default size, can be adjusted by layout
        self.graph_visualizer = GraphVisualizerQt(self.processor)
        self.graph_visualizer.set_figure_axes(self.graph_qt_canvas.fig, self.graph_qt_canvas.axes)
        if self.processor.tooth_ids:
            self.initial_graph_teeth=[self.processor.tooth_ids[0],self.processor.tooth_ids[1]] if len(self.processor.tooth_ids)>=2 else self.processor.tooth_ids[:1]
            self.currently_graphed_tooth_ids = list(self.initial_graph_teeth) 
            if self.initial_graph_teeth : self.graph_visualizer.plot_tooth_lines(self.initial_graph_teeth)

        # 2. Vedo Multi-View Widget Setup
        self.vedo_multiview_widget = EmbeddedVedoMultiViewWidget(
            self.processor, 
            HardwareGridVisualizerQt, 
            Hardware3DBarVisualizerQt, # Use new class
            self, # Pass self (MainAppWindow) as parent_main_window
            plotter_kwargs={'title': "Dental Force Views"}
        )
        # Link MainAppWindow for callbacks from grid visualizer
        if hasattr(self.vedo_multiview_widget.grid_visualizer, 'set_main_app_window_ref'):
            self.vedo_multiview_widget.grid_visualizer.main_app_window_ref = self
        # Link MainAppWindow for callbacks from 3D bar visualizer (if it has click interactions)
        if hasattr(self.vedo_multiview_widget.bar_visualizer, 'set_main_app_window_ref'):
            self.vedo_multiview_widget.bar_visualizer.main_app_window_ref = self

        # --- INITIALIZE DETAILED INFO LABEL HERE ---
        self.detailed_info_label = QLabel("Click on a tooth/bar to see details.")
        self.detailed_info_label.setWordWrap(True)
        self.detailed_info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.detailed_info_label.setStyleSheet("padding: 5px; background-color: #f0f0f0;") # Added some style
        # --- END INITIALIZATION ---

        # Now call _setup_ui and _setup_animation_timer
        self._setup_ui()
        self._setup_animation_timer()
        
        # Initial render of views
        if self.processor.timestamps:
            first_ts = self.processor.timestamps[0]; self.last_animated_timestamp = first_ts
            self.vedo_multiview_widget.update_views(first_ts, self.get_latest_hw_data_for_step(), 1) # Pass initial data
            if self.graph_visualizer.figure: self.graph_visualizer.figure.canvas.draw_idle() # Initial graph draw
        else: # If no timestamps, at least show the Vedo widget structure
            self.vedo_multiview_widget.Render()


    def get_latest_hw_data_for_step(self): # Helper for animation_step
        """Gets latest hardware data if source is available."""
        if self.hw_data_source and self.hw_data_source.running:
            if hasattr(self.hw_data_source, 'get_latest_raw_forces'):
                return self.hw_data_source.get_latest_raw_forces()
        return None
    
    def _initialize_video_writer(self): # ... (same as before) ...
        if self.video_writer is None:
            if os.path.exists(self.output_video_filename):
                try: os.remove(self.output_video_filename); logging.info(f"Removed existing: {self.output_video_filename}")
                except Exception as e: logging.warning(f"Could not remove {self.output_video_filename}: {e}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v'); self.video_writer = cv2.VideoWriter(self.output_video_filename,fourcc,float(self.fps),(self.canvas_width,self.canvas_height))
            global _main_app_window_instance_for_atexit; _main_app_window_instance_for_atexit = self
            if not self.video_writer.isOpened(): logging.error(f"Could not open video writer for {self.output_video_filename}"); self.video_writer = None; _main_app_window_instance_for_atexit = None
            else: logging.info(f"Video writer opened for {self.output_video_filename} at {self.fps} FPS.")
        return self.video_writer is not None and self.video_writer.isOpened()


    def _setup_ui(self):
        central_widget = QWidget(); self.setCentralWidget(central_widget)
        main_vertical_layout = QVBoxLayout(central_widget)

        # --- Top Area: Single Vedo MultiView and Info Panel ---
        top_area_layout = QHBoxLayout()
        top_area_layout.addWidget(self.vedo_multiview_widget, 3) # Vedo views take more space
        top_area_layout.addWidget(self.detailed_info_label, 1)   # Info panel
        main_vertical_layout.addLayout(top_area_layout, 3)

        main_vertical_layout.addWidget(self.graph_qt_canvas, 2)
        # ... (controls layout as before) ...
        controls_layout=QHBoxLayout(); self.play_pause_button=QPushButton("Play Animation"); self.play_pause_button.clicked.connect(self.toggle_animation)
        self.reset_3d_view_button = QPushButton("Reset 3D View"); self.reset_3d_view_button.clicked.connect(self.reset_3d_bar_camera_in_multiview) # New handler
        controls_layout.addStretch(1); controls_layout.addWidget(self.play_pause_button); controls_layout.addWidget(self.reset_3d_view_button); controls_layout.addStretch(1)
        main_vertical_layout.addLayout(controls_layout)


    def reset_3d_bar_camera_in_multiview(self):
        """Calls reset on the 3D bar visualizer within the multiview widget."""
        if self.vedo_multiview_widget and hasattr(self.vedo_multiview_widget.bar_visualizer, 'reset_camera_view'):
            logging.info("Resetting 3D Bar View camera (multiview).")
            self.vedo_multiview_widget.bar_visualizer.reset_camera_view()
            # The EmbeddedVedoMultiViewWidget's update_views or Render method should refresh
            if hasattr(self.vedo_multiview_widget, 'Render'):
                self.vedo_multiview_widget.Render()


    def animation_step(self): 
        if not self.processor.timestamps: self.toggle_animation(); return # Or use live time
        
        # --- Get Live Hardware Data ---
        latest_hardware_flat_data = None
        current_timestamp_for_display = time.time() # Use actual time for display
        sensitivity_from_ui = int(self.sens_combo.get()) if hasattr(self, 'sens_combo') else 1 # Get sensitivity

        if self.hw_data_source and self.hw_data_source.running:
            latest_hardware_flat_data = self.hw_data_source.get_latest_raw_forces() # Assumes method in hw_data_source
            # The timestamp from hardware data might be more relevant if available,
            # otherwise, use animation timer's progression.
            # For now, we'll pass the flat data and use the QTimer's timestamp progression.
            # This needs a robust way to get a synchronized timestamp if data is truly live.
            # Let's assume for now the hardware gives data fast enough for each animation frame.

        # Use animation timer's progression for timestamp if not using live hardware timestamps
        if self.processor.timestamps: # Fallback to simulated/preloaded timestamps if no live data
             current_sim_timestamp = self.processor.timestamps[self.current_timestamp_idx]
             self.last_animated_timestamp = current_sim_timestamp
        else:
             self.last_animated_timestamp = current_timestamp_for_display # live time


        # Pass latest_hardware_flat_data to the multiview updater
        self.vedo_multiview_widget.update_views(self.last_animated_timestamp, latest_hardware_flat_data, sensitivity_from_ui)
        
        # Graph and Video Compositing (still based on processor data for now)
        if self.graph_visualizer.figure and self.graph_visualizer.ax:
            self.graph_visualizer.update_graph_to_timestamp(self.last_animated_timestamp, self.currently_graphed_tooth_ids)
            self.graph_visualizer.update_time_indicator(self.last_animated_timestamp) 
            self.graph_qt_canvas.draw_idle()
        
        if self.video_writer and self.video_writer.isOpened():
            frame_vedo_multiview = self.vedo_multiview_widget.get_frame_as_array(self.last_animated_timestamp, latest_hardware_flat_data, sensitivity_from_ui)
            frame_graph = self.graph_visualizer.get_frame_as_array(self.last_animated_timestamp, self.currently_graphed_tooth_ids)
            # ... (compositing logic as before) ...
            if frame_vedo_multiview is not None and frame_graph is not None:
                # ... (resize and place on canvas) ...
                # self.video_writer.write(canvas)
                pass # Ensure canvas is correctly composed

        if self.processor.timestamps: # Only advance if using preloaded timestamps
            self.current_timestamp_idx = (self.current_timestamp_idx + 1) % len(self.processor.timestamps)
        logging.debug(f"Qt App Step: Time {self.last_animated_timestamp:.1f}s")

    # ... (other MainAppWindow methods like update_graph_on_click, update_detailed_info, closeEvent) ...
    # update_graph_on_click and update_detailed_info will not work with hardware grid directly yet.
    


    def _setup_animation_timer(self): self.animation_timer.timeout.connect(self.animation_step)
    
    def toggle_animation(self):
        if self.is_animating:
            # --- PAUSING ---
            self.animation_timer.stop()
            self.play_pause_button.setText("Play Animation")
            logging.info("Animation Paused.")
            # Optionally, release video writer on pause if you only want to record continuous segments
            # if self.video_writer and self.video_writer.isOpened():
            #     logging.info("Releasing video writer on pause.")
            #     self.video_writer.release()
            #     self.video_writer = None 
            #     global _main_app_window_instance_for_atexit # So atexit knows it's handled
            #     _main_app_window_instance_for_atexit = None # If MainAppWindow instance changes or script ends
        else:
            # --- STARTING or RESUMING ---
            if not self.processor.timestamps or len(self.processor.timestamps) == 0:
                logging.warning("No data to animate.")
                self.is_animating = False # Ensure state is correct
                self.play_pause_button.setText("Play Animation")
                return

            # Initialize or re-initialize video writer if not already open
            # This allows recording to start/resume when play is pressed
            if not hasattr(self, 'video_writer') or self.video_writer is None or not self.video_writer.isOpened():
                if not self._initialize_video_writer(): # Try to initialize
                    logging.warning("Video writer could not be initialized. Animation will play without recording.")
                # If _initialize_video_writer fails, self.video_writer will be None,
                # and the compositing step will safely skip writing frames.
            
            # Reset current_timestamp_idx to 0 if you want "Play" to always restart from beginning
            # self.current_timestamp_idx = 0 
            # Or, to resume, just ensure it's valid:
            if self.current_timestamp_idx >= len(self.processor.timestamps):
                self.current_timestamp_idx = 0


            self.animation_timer.start(int(1000 / self.fps))
            self.play_pause_button.setText("Pause Animation")
            logging.info(f"Animation Started/Resumed at {self.fps} FPS.")
        
        self.is_animating = not self.is_animating

    def update_graph_on_click(self, sel_tid=None): # ... (same logic)
        new_ids = [sel_tid] if sel_tid is not None else self.initial_graph_teeth
        if new_ids!=self.currently_graphed_tooth_ids or not self.graph_visualizer.lines:
            self.graph_visualizer.plot_tooth_lines(new_ids); self.currently_graphed_tooth_ids=new_ids
            if self.processor.timestamps:
                curr_t = self.processor.timestamps[self.current_timestamp_idx]
                self.graph_visualizer.update_graph_to_timestamp(curr_t,new_ids)
                self.graph_visualizer.update_time_indicator(curr_t)
            self.graph_qt_canvas.draw_idle()
            
    def update_detailed_info(self, info_str): self.detailed_info_label.setText(info_str)


    def closeEvent(self, event): # ... (same as before) ...
        logging.info("Main window closing..."); self.animation_timer.stop()
        if hasattr(self, 'video_writer') and self.video_writer and self.video_writer.isOpened():
            logging.info("Releasing video writer from MainAppWindow closeEvent.")
            self.video_writer.release(); self.video_writer = None
            global _video_writer_for_atexit; _video_writer_for_atexit = None 
        super().closeEvent(event)

    def request_main_vedo_render(self):
        if hasattr(self.vedo_multiview_widget, 'Render'):
            self.vedo_multiview_widget.Render()
        elif self.vedo_multiview_widget and self.vedo_multiview_widget.main_plotter:
            self.vedo_multiview_widget.main_plotter.render()
        logging.debug("MainAppWindow: Explicit Vedo render requested.")


    def force_render_vedo_views(self, timestamp):
        """Forces an update and render of the Vedo views for a given timestamp."""
        logging.debug(f"MAIN_APP: Forcing Vedo views update for timestamp {timestamp:.2f}")
        if hasattr(self, 'vedo_multiview_widget'):
            self.vedo_multiview_widget.update_views(timestamp) # This calls animate and then Render on canvas
        else:
            logging.warning("MAIN_APP: vedo_multiview_widget not found for forced render.")
            

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # --- Placeholder for HardwareDataReader setup ---
    # For now, we'll still use simulated data via DataProcessor
    # In a real scenario, you'd initialize your HardwareDataReader here
    # and MainAppWindow would poll it.
    sim_reader = SensorDataReader()
    data = sim_reader.simulate_data(duration=10, num_teeth=16, num_sensor_points_per_tooth=4) # Keep this for now
    processor = DataProcessor(data) # Processor still works on this DataFrame structure
    processor.create_force_matrix() 
    # In a true hardware setup, DataProcessor might be bypassed or adapted for the flat array.

    # For testing the HardwareGridVisualizer, we need a way to feed it flat data.
    # Let's create a dummy hw_data_source that just cycles through some data.
    class DummyHWSource:
        def __init__(self, num_valid_sensors):
            self.num_valid_sensors = num_valid_sensors
            self.running = True
            self.iter = 0
        def get_latest_raw_forces(self):
            # Simulate changing data
            self.iter +=1
            return [(self.iter + i*10) % 1001 for i in range(self.num_valid_sensors)] # Simple cycling data
        def connect(self): return True
        def disconnect(self): self.running = False

    # Calculate number of valid cells from PointsArray
    pa = PointsArray()
    num_valid_hw_cells = sum(1 for r in range(44) for c in range(52) if pa.is_valid(c,r))
    
    hw_data_source_for_app = DummyHWSource(num_valid_hw_cells)
    # --- End Placeholder ---

    if not processor.timestamps and not hw_data_source_for_app: # Check both
        logging.error("No data source. Exiting."); sys.exit(-1)
        
    main_window = MainAppWindow(processor, hw_data_source=hw_data_source_for_app) 
    main_window.show()
    sys.exit(app.exec_())

# --- END OF FILE main_qt_app.py ---