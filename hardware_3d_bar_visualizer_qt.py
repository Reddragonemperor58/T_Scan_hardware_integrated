# --- START OF FILE hardware_3d_bar_visualization_qt.py ---
import numpy as np
from vedo import Text2D, Box, Line, Grid, Plane, Text3D, colors # Plotter passed in
import logging
from points_array import PointsArray 
import vtk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Hardware3DBarVisualizerQt:
    def __init__(self, processor_instance, parent_plotter_instance, renderer_index):
        self.processor = processor_instance 
        self.parent_plotter = parent_plotter_instance
        self.renderer_index = renderer_index
        self.renderer = parent_plotter_instance.renderers[renderer_index]

        if self.processor.cleaned_data is None: self.processor.create_force_matrix()
        self.num_data_teeth = len(self.processor.tooth_ids) if self.processor.tooth_ids else 0 # Not directly used for hw grid
        
        self.hw_rows = 44
        self.hw_cols = 52
        self.points_array_checker = PointsArray()
        
        self.arch_layout_width = 14.0 # Reference for positioning calculations
        self.arch_layout_depth = 8.0
        self.bar_base_size = 0.22
        
        self.hw_cell_bar_base_positions_and_ids = [] # List of {'col':c,'row':r,'pos':np.array}
        
        self.max_force_for_scaling = 1000.0 
        self.max_bar_height = 1.5 # Adjusted max height
        self.min_bar_height = 0.01        
        self.grid_center_y = 0.0 # Will be based on hw_rows * bar_base_size
        self.initial_camera_settings = {} 

        self.timestamps = self.processor.timestamps 
        self.current_timestamp_idx = 0; self.last_animated_timestamp = None 
        
        self.force_bar_actors_dict = {} # Dict: {(r,c): BoxActor} - PERSISTENT
        self.time_text_actor = None       
        self.floor_actor = None    
        # self.static_arch_line_actor = None # Optional for this view
        # self.static_cell_number_actors = [] # Likely too cluttered

        self.main_app_window_ref = None

        if not self.renderer: logging.error(f"Hw3DBarViz (R{self.renderer_index}): Renderer not provided."); return
        # setup_scene will be called by EmbeddedVedoMultiViewWidget

    def set_main_app_window_ref(self, main_app_window_instance):
        self.main_app_window_ref = main_app_window_instance

    def setup_scene(self):
        if not self.renderer or not self.parent_plotter: return
        logging.info(f"Hw3DBarViz (R{self.renderer_index}): Setting up scene...")
        self.parent_plotter.at(self.renderer_index)
        cam = self.parent_plotter.camera
        
        # Calculate grid center for floor and camera based on actual bar layout
        temp_total_grid_width = self.hw_cols * self.bar_base_size
        temp_total_grid_height = self.hw_rows * self.bar_base_size
        self.grid_center_x = 0 # Assuming _create_hw_cell_bar_positions centers at X=0
        self.grid_center_y = 0 # Assuming _create_hw_cell_bar_positions centers at Y=0

        floor_size_x = temp_total_grid_width * 1.1 
        floor_size_y = temp_total_grid_height * 1.1
        self.floor_actor = Grid(s=(floor_size_x, floor_size_y), res=(self.hw_cols//2, self.hw_rows//2), c='gainsboro', alpha=0.3)
        self.floor_actor.pos(self.grid_center_x, self.grid_center_y, -0.05) 
        self.renderer.AddActor(self.floor_actor.actor)

        self._create_and_add_bars_once() # Creates Box actors once

        # Initial camera position
        cam.SetPosition(self.grid_center_x, self.grid_center_y - temp_total_grid_height*0.8, self.max_bar_height * 3)
        cam.SetFocalPoint(self.grid_center_x, self.grid_center_y, self.max_bar_height / 2)
        cam.SetViewUp(0, 0, 1) # Z is up for this perspective
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        self.renderer.SetBackground(0.98, 0.95, 0.92) 
        logging.info(f"Hw3DBarViz (R{self.renderer_index}): Scene setup complete.")


    def _create_hw_cell_bar_positions(self): # Renamed to avoid confusion, called by _create_and_add_bars_once
        self.hw_cell_bar_base_positions_and_ids = [] 
        total_vis_width = self.hw_cols * self.bar_base_size
        total_vis_height = self.hw_rows * self.bar_base_size
        offset_x = -total_vis_width / 2 + self.bar_base_size / 2
        offset_y = -total_vis_height / 2 + self.bar_base_size / 2 
        for r_idx in range(self.hw_rows):
            for c_idx in range(self.hw_cols):
                if self.points_array_checker.is_valid(c_idx, r_idx):
                    bar_x = offset_x + c_idx * self.bar_base_size
                    bar_y = offset_y + ((self.hw_rows - 1 - r_idx) * self.bar_base_size) 
                    self.hw_cell_bar_base_positions_and_ids.append(
                        {'col': c_idx, 'row': r_idx, 'pos': np.array([bar_x, bar_y, 0.0])} # Base Z is 0
                    )
        # Update grid center based on actual positions if needed (though offsets should center it)
        if self.hw_cell_bar_base_positions_and_ids:
            all_x = [p['pos'][0] for p in self.hw_cell_bar_base_positions_and_ids]
            all_y = [p['pos'][1] for p in self.hw_cell_bar_base_positions_and_ids]
            self.grid_center_x = np.mean(all_x) if all_x else 0
            self.grid_center_y = np.mean(all_y) if all_y else 0


    def _create_and_add_bars_once(self):
        if not self.renderer: return
        self._create_hw_cell_bar_positions() # Populate base positions

        if self.force_bar_actors_dict: # Clear if called again
            for bar_actor in self.force_bar_actors_dict.values(): self.renderer.RemoveActor(bar_actor.actor)
            self.force_bar_actors_dict.clear()

        bar_vtk_actors_to_add = []
        for cell_info in self.hw_cell_bar_base_positions_and_ids:
            base_pos = cell_info['pos']
            r_idx, c_idx = cell_info['row'], cell_info['col']
            
            # Create with initial minimal height at Z=0 base
            bar = Box(pos=(base_pos[0], base_pos[1], self.min_bar_height / 2.0), # Centered for initial min_height
                      length=self.bar_base_size*0.85, 
                      width=self.bar_base_size*0.85,  
                      height=self.min_bar_height, # Start with min height
                      c='lightgrey', alpha=0.1)
            bar.name = f"HWBar_c{c_idx}_r{r_idx}"
            bar.pickable = True # If you want to pick bars
            
            self.force_bar_actors_dict[(r_idx, c_idx)] = bar
            if hasattr(bar, 'actor'): bar_vtk_actors_to_add.append(bar.actor)
        
        if bar_vtk_actors_to_add:
            for act in bar_vtk_actors_to_add: self.renderer.AddActor(act)
        logging.info(f"Hw3DBarViz (R{self.renderer_index}): Created {len(self.force_bar_actors_dict)} static bar Box actors.")


    def _value_to_color_hardware(self, value, sensitivity=1): # Same as before
        # ... (color mapping logic from HardwareGridVisualizerQt) ...
        mapped_value = (value / sensitivity * 255) // self.max_force_for_scaling 
        mapped_value = min(255, max(0, int(mapped_value)))
        r,g,b = 200,200,200 
        if mapped_value > 204: r=255; g=max(0,int(150-((mapped_value-204)*150/51))); b=0
        elif mapped_value > 140: r=int(139+((mapped_value-140)*116/64)); g=int((mapped_value-140)*150/64); b=0
        elif mapped_value > 76: g=int(255-((mapped_value-76)*155/64)); r=int(((mapped_value-76)/64)*100); b=0
        elif mapped_value > 12: r=0; g=int(255-((mapped_value-12)*155/64)); b=int(100-((mapped_value-12)*50/64))
        return (r/255.0, g/255.0, b/255.0)

    def render_display(self, timestamp, hardware_data_flat_array, sensitivity=1):
        if not self.renderer or not self.parent_plotter: return
        self.parent_plotter.at(self.renderer_index)

        # Update Time text (recreated)
        if self.time_text_actor: self.renderer.RemoveActor(self.time_text_actor.actor)
        self.time_text_actor = Text2D(f"HW 3D - T: {timestamp:.1f}s", pos="bottom-right", c='k', s=0.7)
        self.renderer.AddActor(self.time_text_actor.actor) # Add new one

        if hardware_data_flat_array is None or not self.hw_cell_bar_base_positions_and_ids: 
            # Hide all bars if no data
            for bar_actor in self.force_bar_actors_dict.values(): bar_actor.alpha(0)
            return

        data_idx = 0
        for cell_info in self.hw_cell_bar_base_positions_and_ids:
            base_pos = cell_info['pos']
            r_idx, c_idx = cell_info['row'], cell_info['col']
            bar_actor = self.force_bar_actors_dict.get((r_idx, c_idx))

            if not bar_actor: continue

            if data_idx < len(hardware_data_flat_array):
                value = hardware_data_flat_array[data_idx]
                norm_force = min(1.0, max(0.0, (value / sensitivity) / self.max_force_for_scaling))
                new_bar_h = self.min_bar_height + norm_force * (self.max_bar_height - self.min_bar_height)
                
                if value < 5 : new_bar_h = 0.0 # Threshold for very low values to be invisible

                if new_bar_h < self.min_bar_height / 2: # Effectively zero or very small
                    bar_actor.alpha(0) # Make it invisible
                    # Optionally scale Z to be very small if alpha(0) isn't enough for some reason
                    # current_actor_height = bar_actor.bounds()[5] - bar_actor.bounds()[4]
                    # if current_actor_height > 0: bar_actor.scale([1,1,0.001/current_actor_height], reset=False)
                else:
                    color = self._value_to_color_hardware(value, sensitivity)
                    bar_actor.color(color).alpha(0.92)

                    # Update height and position
                    # Get current height of the Box actor
                    # bounds() returns [xmin,xmax, ymin,ymax, zmin,zmax] for the scaled actor
                    # For a Box actor, height is along its Z axis if not rotated
                    current_actor_height = bar_actor.bounds()[5] - bar_actor.bounds()[4]
                    if abs(current_actor_height) < 1e-6: current_actor_height = self.min_bar_height # Avoid div by zero if collapsed
                    
                    # Calculate scaling factor for Z.
                    # Actor is already scaled, so new scale is relative to current scale (which is 1 for Z if not previously scaled)
                    # The Box is created with initial height of min_bar_height.
                    # We need to scale based on its *original* unscaled height (which was min_bar_height)
                    # Simpler: if the Box was created with height=1 and then scaled, it's easier.
                    # Since we create it with min_bar_height, let's try to set vertices or re-mesh (more expensive).
                    # The easiest is to scale from a reference height of 1.
                    # Let's assume the Box was created with height 1 and then scaled to min_bar_height.
                    # For now, let's try recreating the bar if height changes significantly, as scaling a Box's height
                    # while keeping base fixed and re-centering is tricky without direct vertex manipulation or more complex transforms.
                    # This is a place where recreating might be simpler than updating a Box's height.
                    
                    # Recreating for simplicity of height update:
                    self.renderer.RemoveActor(bar_actor.actor) # Remove old
                    new_bar_center_z = base_pos[2] + new_bar_h / 2.0
                    new_bar = Box(pos=(base_pos[0], base_pos[1], new_bar_center_z),
                                  length=self.bar_base_size*0.85, width=self.bar_base_size*0.85,  
                                  height=new_bar_h, c=color, alpha=0.92)
                    new_bar.name = bar_actor.name; new_bar.pickable = bar_actor.pickable
                    self.force_bar_actors_dict[(r_idx, c_idx)] = new_bar # Replace in dict
                    self.renderer.AddActor(new_bar.actor) # Add new one
                data_idx += 1
            else: 
                bar_actor.alpha(0.1).color('lightgrey')
        
        # No self.renderer.render() here

    def animate(self, timestamp_to_render, hardware_data_for_timestamp=None, sensitivity=1):
        self.last_animated_timestamp = timestamp_to_render
        self.render_display(timestamp_to_render, hardware_data_for_timestamp, sensitivity)

    def get_frame_as_array(self, timestamp_to_render, hardware_data_for_timestamp=None, sensitivity=1):
        # ... (same as GridVisualizer's, ensuring render_display is called) ...
        if not self.renderer or not self.parent_plotter: return None
        self.parent_plotter.at(self.renderer_index)
        self.render_display(timestamp_to_render, hardware_data_for_timestamp, sensitivity) 
        # Main plotter screenshot in EmbeddedVedoMultiViewWidget will capture this.
        return None 

    def reset_camera_view(self): # ... (same as before, using self.parent_plotter.camera) ...
        if not self.renderer or not self.parent_plotter: return
        self.parent_plotter.at(self.renderer_index)
        cam = self.parent_plotter.camera
        focus_x = self.grid_center_x; focus_y = self.grid_center_y # Use calculated centers
        pos=(focus_x, focus_y - (self.hw_rows * self.bar_base_size) * 0.8, self.max_bar_height * 2.5) # Adjusted Y
        focal=(focus_x, focus_y, self.max_bar_height / 3)
        vup=(0,0,1) # Z-up
        if not self.initial_camera_settings: # Store if first time
            self.initial_camera_settings = {'position':pos, 'focal_point':focal, 'viewup':vup}
        cam.SetPosition(self.initial_camera_settings['position'])
        cam.SetFocalPoint(self.initial_camera_settings['focal_point'])
        cam.SetViewUp(self.initial_camera_settings['viewup'])
        self.renderer.ResetCamera(); self.renderer.ResetCameraClippingRange()
        logging.info(f"Hw3DBarViz (R{self.renderer_index}): Camera reset.")
        if hasattr(self.parent_plotter.qt_widget, 'Render'): self.parent_plotter.qt_widget.Render()
        elif self.parent_plotter.window: self.parent_plotter.render()


    def _on_mouse_click(self, event): # ... (similar to GridVisualizer's, checking actor name "HWBar_...") ...
        if not self.renderer or getattr(event,'renderer',None) != self.renderer: return 
        # ... (rest of click handling for bars) ...
# --- END OF FILE hardware_3d_bar_visualization_qt.py ---