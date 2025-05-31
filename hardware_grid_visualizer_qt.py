# --- START OF FILE hardware_grid_visualizer_qt.py ---
import numpy as np
from vedo import Text2D, Rectangle, colors, Plotter # Plotter might be needed for type hinting if passing parent_plotter
import logging
from points_array import PointsArray

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HardwareGridVisualizerQt:
    def __init__(self, processor_placeholder, parent_plotter_instance, renderer_index):
        self.processor_ref = processor_placeholder
        self.parent_plotter = parent_plotter_instance
        self.renderer_index = renderer_index
        self.renderer = parent_plotter_instance.renderers[renderer_index]

        self.hw_rows = 44
        self.hw_cols = 52
        self.points_array_checker = PointsArray()
        self.max_force_for_scaling = 1000.0 
        
        self.cell_rect_actors = {} # Dict: {(r, c): RectangleActor} - PERSISTENT
        self.time_text_actor = None # Will be recreated (simple)
        
        self.timestamps = self.processor_ref.timestamps
        self.current_timestamp_idx = 0
        self.last_animated_timestamp = 0.0
        self.main_app_window_ref = None

        if not self.renderer: logging.error(f"HwGridViz (R{self.renderer_index}): Renderer not provided."); return

    def set_main_app_window_ref(self, main_app_window_instance):
        self.main_app_window_ref = main_app_window_instance

    def setup_scene(self):
        if not self.renderer or not self.parent_plotter: return
        logging.info(f"HwGridViz (R{self.renderer_index}): Setting up scene...")
        self.parent_plotter.at(self.renderer_index)
        cam = self.parent_plotter.camera
        cam.ParallelProjectionOn()

        self._create_and_add_grid_rects_once() # Create Rectangles ONCE and add to renderer

        # Camera fitting logic (ensure it uses self.parent_plotter.camera and self.renderer.ResetCamera())
        cell_render_size = 0.25 
        grid_render_width = self.hw_cols * cell_render_size
        grid_render_height = self.hw_rows * cell_render_size
        # Center the grid view: calculate center of the drawn grid
        # Assuming _create_and_add_grid_rects_once centers the grid around (0,0) in its own XY
        cam.SetFocalPoint(0, 0, 0) # Focus on the center of the grid
        cam.SetPosition(0, 0, 20)  # Position camera along Z-axis
        cam.SetViewUp(0, 1, 0)     # Y is up
        cam.SetParallelScale(grid_render_height / 1.9) # Adjust zoom
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        self.renderer.SetBackground(0.92, 0.92, 0.98) # Light blueish-grey for this view
        logging.info(f"HwGridViz (R{self.renderer_index}): Scene setup complete.")

    def _create_and_add_grid_rects_once(self):
        if not self.renderer: return
        if self.cell_rect_actors: # Should only be called once, but defensive
            self.renderer.RemoveActors(list(self.cell_rect_actors.values())) # Pass actual actor objects
            self.cell_rect_actors.clear()

        cell_size = 0.25; padding = 0.01 
        effective_cell_draw_size = cell_size - padding
        half_draw_size = effective_cell_draw_size / 2.0
        total_grid_visual_width = self.hw_cols * cell_size
        total_grid_visual_height = self.hw_rows * cell_size
        offset_x = -total_grid_visual_width / 2; offset_y = -total_grid_visual_height / 2
        
        rect_actors_to_add_vtk = []
        for r_idx in range(self.hw_rows):
            for c_idx in range(self.hw_cols):
                cell_center_x = offset_x + (c_idx * cell_size) + cell_size / 2
                cell_center_y = offset_y + ((self.hw_rows - 1 - r_idx) * cell_size) + cell_size/2 
                
                p1x = cell_center_x - half_draw_size; p1y = cell_center_y - half_draw_size
                p2x = cell_center_x + half_draw_size; p2y = cell_center_y + half_draw_size
                
                rect = Rectangle((p1x, p1y), (p2x, p2y), c='lightgrey', alpha=0.1)
                rect.lw(0)
                if not self.points_array_checker.is_valid(c_idx, r_idx): rect.alpha(0) 
                self.cell_rect_actors[(r_idx, c_idx)] = rect
                if hasattr(rect, 'actor'): rect_actors_to_add_vtk.append(rect.actor)
        
        if rect_actors_to_add_vtk:
            for act in rect_actors_to_add_vtk: self.renderer.AddActor(act)
        logging.info(f"HwGridViz (R{self.renderer_index}): Created {len(self.cell_rect_actors)} cell rectangles.")

    def _value_to_color_hardware(self, value, sensitivity=1): # Same
        # ... color mapping ...
        mapped_value = (value / sensitivity * 255) // self.max_force_for_scaling
        mapped_value = min(255, max(0, int(mapped_value)))
        r,g,b = 211,211,211 
        if mapped_value > 204: r=255; g=max(0,int(150-((mapped_value-204)*150/51))); b=0
        elif mapped_value > 140: r=int(139+((mapped_value-140)*116/64)); g=int((mapped_value-140)*150/64); b=0
        elif mapped_value > 76: g=int(255-((mapped_value-76)*155/64)); r=int(((mapped_value-76)/64)*100); b=0
        elif mapped_value > 12: r=0; g=int(255-((mapped_value-12)*155/64)); b=int(100-((mapped_value-12)*50/64))
        return (r/255.0, g/255.0, b/255.0)


    def render_grid_view(self, timestamp, hardware_data_flat_array, sensitivity=1):
        if not self.renderer or not self.parent_plotter: return
        self.parent_plotter.at(self.renderer_index) # Activate renderer

        # --- Only recreate time_text_actor ---
        if self.time_text_actor: self.renderer.RemoveActor(self.time_text_actor.actor)
        self.time_text_actor = Text2D(f"HW Grid - T: {timestamp:.1f}s", pos="bottom-left", c='k', s=0.7)
        self.renderer.AddActor(self.time_text_actor.actor)
        # ---

        if hardware_data_flat_array is None: return

        data_idx = 0
        for r_idx in range(self.hw_rows):
            for c_idx in range(self.hw_cols):
                rect_actor = self.cell_rect_actors.get((r_idx, c_idx))
                if not rect_actor: continue 

                if self.points_array_checker.is_valid(c_idx, r_idx):
                    if data_idx < len(hardware_data_flat_array):
                        value = hardware_data_flat_array[data_idx]
                        color = self._value_to_color_hardware(value, sensitivity)
                        # --- UPDATE EXISTING RECT ACTOR ---
                        rect_actor.color(color).alpha(1.0 if value > 5 else 0.2) 
                        data_idx += 1
                    else: 
                        rect_actor.alpha(0.2).color('lightgrey') 
                # Invalid cells' alpha remains 0 from init
        # No self.renderer.render() here

    def animate(self, timestamp_to_render, hardware_data_for_timestamp=None, sensitivity=1):
        self.last_animated_timestamp = timestamp_to_render
        self.render_grid_view(timestamp_to_render, hardware_data_for_timestamp, sensitivity)

    def get_frame_as_array(self, timestamp_to_render, hardware_data_for_timestamp=None, sensitivity=1):
        if not self.renderer or not self.parent_plotter: return None
        self.parent_plotter.at(self.renderer_index)
        self.render_grid_view(timestamp_to_render, hardware_data_for_timestamp, sensitivity) # Update actors
        # The main plotter will be rendered once before screenshotting by EmbeddedVedoMultiViewWidget
        return None # This visualizer doesn't return the frame; the main widget does.

    # _on_mouse_click can be added later if needed for this raw grid
# --- END OF FILE hardware_grid_visualizer_qt.py ---