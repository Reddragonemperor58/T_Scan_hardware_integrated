# --- START OF FILE dental_arch_3d_bar_visualization_qt.py ---
import numpy as np
from vedo import Text2D, Cylinder, Box, Line, Axes, Grid, Plane, Text3D, colors 
import logging
import vtk 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DentalArch3DBarVisualizerQt:
    def __init__(self, processor, parent_plotter_instance, renderer_index):
        self.processor = processor
        self.parent_plotter = parent_plotter_instance 
        self.renderer_index = renderer_index         
        self.renderer = parent_plotter_instance.renderers[renderer_index] 

        if self.processor.cleaned_data is None: self.processor.create_force_matrix()
        self.num_data_teeth = len(self.processor.tooth_ids) if self.processor.tooth_ids else 0
        
        self.arch_layout_width = 14.0; self.arch_layout_depth = 8.0; self.bar_base_radius = 0.5     
        self.tooth_bar_base_positions = [] 
        self.max_force_for_scaling = self.processor.max_force_overall if hasattr(self.processor,'max_force_overall') else 100.0
        self.max_bar_height = 5.0; self.min_bar_height = 0.1 
        self.grid_center_y = -self.arch_layout_depth * 0.3 
        self.initial_camera_settings = {} 

        self.timestamps = self.processor.timestamps; self.current_timestamp_idx = 0; self.last_animated_timestamp = None 
        self.force_bar_actors = []; self.time_text_actor = None; self.arch_base_line_actor = None; self.tooth_label_actors = []    
        self.floor_grid_actor = None; self.axes_actor_local = None    
        self.selected_tooth_id_3dbar = None 
        self.main_app_window_ref = None

        if self.num_data_teeth == 0:
            logging.error(f"3DBarVizQt (Renderer {self.renderer_index}): No tooth data."); return

    def set_animation_controller_for_interaction(self, controller):
        self.main_app_window_ref = controller

    def setup_scene(self):
        if not self.renderer or not self.parent_plotter: logging.error(f"3DBarVizQt (R{self.renderer_index}): Renderer/ParentPlotter missing."); return
        if self.num_data_teeth == 0:
            if self.renderer: self.renderer.AddActor(Text2D("No 3D Bar Data", c='red', s=1.5).actor); return

        self.parent_plotter.at(self.renderer_index) 
        floor_size_x=self.arch_layout_width*1.5; floor_size_y=self.arch_layout_depth*1.8
        floor_grid_res=(10,10); self.floor_grid_actor=Grid(s=(floor_size_x,floor_size_y),res=floor_grid_res,c='gainsboro',alpha=0.4)
        self.floor_grid_actor.pos(0,self.grid_center_y,-0.05); 
        # --- CORRECTED ADD ---
        self.renderer.AddActor(self.floor_grid_actor.actor) 
        # --- END CORRECTION ---
        self.reset_camera_view() # This method will now use self.parent_plotter.camera

        cam = self.parent_plotter.camera # Get camera for the active renderer
        cam.SetPosition(0,-self.arch_layout_depth*2.0,self.max_bar_height*1.5) 
        cam.SetFocalPoint(0,self.grid_center_y*0.5,self.max_bar_height/4)
        cam.SetViewUp(0,0.4,0.6); 
        self.renderer.ResetCamera() 
        self.renderer.ResetCameraClippingRange()

        self.initial_camera_settings = {'position':cam.GetPosition(),'focal_point':cam.GetFocalPoint(),'viewup':cam.GetViewUp()}

        self.tooth_bar_base_positions = self._create_bar_base_positions(self.num_data_teeth, self.arch_layout_width, self.arch_layout_depth)
        self._initialize_static_elements() 
        
        # if self.parent_plotter and hasattr(self, '_on_mouse_click'):
        #     self.parent_plotter.add_callback('mouse click', self._on_mouse_click)

    def _create_bar_base_positions(self, num_teeth, total_width, total_depth):
        if num_teeth == 0: return [] # x_coords not defined here
        positions_3d = []
        if num_teeth == 1: 
            x_coords = np.array([0.0]) # x_coords defined here
        else: 
            x_coords = np.linspace(-total_width / 2, total_width / 2, num_teeth) # x_coords defined here
        
        # This line uses x_coords. If num_teeth was 0, x_coords was never defined.
        k = total_depth / ((total_width / 2)**2) if total_width != 0 else 0
        # The list comprehension is outside the if/else for num_teeth == 1
        return [np.array([x,total_depth-k*(x**2)-total_depth*0.8,0.0]) for x in x_coords]
    

    def _initialize_static_elements(self): 
        if not self.renderer or not self.tooth_bar_base_positions: return
        static_actors_to_add_vedo = [] # Collect Vedo objects
        if len(self.tooth_bar_base_positions)>1:
            line_pts=[(p[0],p[1],0.01) for p in self.tooth_bar_base_positions]
            self.arch_base_line_actor=Line(line_pts,c='dimgray',lw=2,alpha=0.7); static_actors_to_add_vedo.append(self.arch_base_line_actor)
            self.tooth_label_actors=[]
            for i,pos in enumerate(self.tooth_bar_base_positions):
                if i < len(self.processor.tooth_ids):
                    tid=self.processor.tooth_ids[i]
                    lbl_pos=(pos[0],pos[1]-self.bar_base_radius*0.7,-0.1) 
                    lbl=Text3D(str(tid),pos=lbl_pos,s=0.20,c=(0.2,0.2,0.2),depth=0.01,justify='ct')
                    self.tooth_label_actors.append(lbl); static_actors_to_add_vedo.append(lbl)
        if static_actors_to_add_vedo: 
            for vo in static_actors_to_add_vedo: self.renderer.AddActor(vo.actor) # Add .actor

    def render_display(self, timestamp): 
        if not self.tooth_bar_base_positions or not self.renderer: return
        self.parent_plotter.at(self.renderer_index) 
        
        actors_to_remove_vtk = []
        if self.time_text_actor: actors_to_remove_vtk.append(self.time_text_actor.actor)
        for act in self.force_bar_actors: actors_to_remove_vtk.append(act.actor)
        for vtk_act in actors_to_remove_vtk: 
            if vtk_act: self.renderer.RemoveActor(vtk_act)
        
        self.force_bar_actors.clear(); self.time_text_actor = None
        
        current_vedo_actors_to_add = []
        self.time_text_actor = Text2D(f"Time: {timestamp:.1f}s",pos="bottom-right",c='k',bg=(1,1,1),alpha=0.6,s=0.7)
        current_vedo_actors_to_add.append(self.time_text_actor)

        for i,base_pos in enumerate(self.tooth_bar_base_positions):
            if i >= len(self.processor.tooth_ids): continue
            tooth_id=self.processor.tooth_ids[i]; _,f_series=self.processor.get_average_force_for_tooth(tooth_id)
            curr_f=0.0
            if self.timestamps and len(f_series)==len(self.timestamps):
                try: idx=np.argmin(np.abs(np.array(self.timestamps)-timestamp)); curr_f=f_series[idx]
                except: pass 
            if not np.isfinite(curr_f): curr_f=0.0
            norm_f=min(1.0,max(0.0,curr_f/self.max_force_for_scaling))
            bar_h=self.min_bar_height+norm_f*(self.max_bar_height-self.min_bar_height)
            if curr_f<1e-3: bar_h=0.0 
            if bar_h>1e-4: 
                if norm_f<0.01:clr=(0.1,0.1,0.6) 
                elif norm_f<0.25:clr=(0.2,0.4,1) 
                elif norm_f<0.5:clr=(0.1,0.8,0.4) 
                elif norm_f<0.75:clr=(1,0.9,0.1)   
                elif norm_f<0.9:clr=(1,0.4,0) 
                else: clr=(0.9,0.0,0.2)                    
                bar_cz=base_pos[2]+bar_h/2.0
                bar=Box(pos=(base_pos[0],base_pos[1],bar_cz),length=self.bar_base_radius*1.6,width=self.bar_base_radius*1.6,height=bar_h,c=clr,alpha=0.92)
                bar.name=f"Bar_Tooth_{tooth_id}"; bar.pickable=True
                if self.selected_tooth_id_3dbar == tooth_id: bar.color('yellow').alpha(1.0)
                self.force_bar_actors.append(bar); current_vedo_actors_to_add.append(bar)
        
        if current_vedo_actors_to_add: 
            for vo in current_vedo_actors_to_add: self.renderer.AddActor(vo.actor)

    def animate(self, timestamp_to_render): 
        if not self.timestamps: return
        self.last_animated_timestamp = timestamp_to_render
        self.render_display(timestamp_to_render)

    def get_frame_as_array(self, timestamp_to_render): # Not used if MainApp screenshots main plotter
        logging.warning("get_frame_as_array called on individual 3DBar viz; main plotter should screenshot.")
        return None 

    def reset_camera_view(self):
        if not self.renderer or not self.parent_plotter: return
        
        self.parent_plotter.at(self.renderer_index) # ***** IMPORTANT *****
        
        cam = self.parent_plotter.camera # Now refers to the correct camera
        # Define your desired initial camera parameters here
        pos = self.initial_camera_settings.get('position', (0, -self.arch_layout_depth * 2.0, self.max_bar_height * 1.5))
        focal_point = self.initial_camera_settings.get('focal_point', (0, self.grid_center_y * 0.5, self.max_bar_height / 4))
        view_up = self.initial_camera_settings.get('viewup', (0, 0.4, 0.6))

        cam.SetPosition(pos)
        cam.SetFocalPoint(focal_point)
        cam.SetViewUp(view_up)
        
        # ResetCamera on the specific renderer is crucial for subplots
        self.renderer.ResetCamera() 
        self.renderer.ResetCameraClippingRange()

        if not self.initial_camera_settings: # Store initial settings
            self.initial_camera_settings = {'position':cam.GetPosition(), 'focal_point':cam.GetFocalPoint(), 'viewup':cam.GetViewUp()}
            
        logging.info(f"3DBarViz (R{self.renderer_index}): Camera (re)set.")


    def _on_mouse_click(self, event): # event is vedo.interaction.Event
        # The dispatcher (_dispatch_mouse_click in EmbeddedVedoMultiViewWidget) should have
        # already determined that this event is relevant to this renderer's actors or its background.
        # A final sanity check using event.renderer can be added if necessary, but ideally dispatcher handles it.
        # if not self.renderer or getattr(event, 'renderer', None) != self.renderer:
        #     return

        clicked_tooth_id_parsed = None
        
        if event.actor: # An actor within this renderer was clicked
            actor_name = event.actor.name
            logging.info(f"3DBarVizQt (R{self.renderer_index if hasattr(self, 'renderer_index') else 'N/A'}) Processing Click: Actor '{actor_name}' at {event.picked3d}")
            if actor_name and actor_name.startswith("Bar_Tooth_"): # Check for Bar_Tooth_ prefix
                try: 
                    clicked_tooth_id_parsed = int(actor_name.split("_")[-1])
                except ValueError: 
                    logging.warning(f"3DBarVizQt: Could not parse tooth_id from actor name: {actor_name}")
        else: # Background of this specific renderer was clicked (event.actor is None)
            logging.info(f"3DBarVizQt (R{self.renderer_index if hasattr(self, 'renderer_index') else 'N/A'}): Processing background click for its renderer.")

        # Update selection state for highlighting within this visualizer
        if clicked_tooth_id_parsed is not None: # Clicked on a recognizable bar
            if self.selected_tooth_id_3dbar == clicked_tooth_id_parsed: # Clicking the same selected bar
                self.selected_tooth_id_3dbar = None # Deselect it
                logging.info(f"--- 3DBarVizQt: Tooth {clicked_tooth_id_parsed} deselected for highlight. ---")
            else: # Clicking a new bar or a previously unselected bar
                self.selected_tooth_id_3dbar = clicked_tooth_id_parsed
                logging.info(f"--- 3DBarVizQt: Tooth {clicked_tooth_id_parsed} selected for highlight. ---")
        elif event.actor is None : # True background click for this renderer (or global deselect propagated)
             if self.selected_tooth_id_3dbar is not None: 
                 logging.info(f"--- 3DBarVizQt: Deselecting tooth {self.selected_tooth_id_3dbar} (background click). ---")
             self.selected_tooth_id_3dbar = None
        elif event.actor is not None and clicked_tooth_id_parsed is None: # Clicked unrecognized actor in this renderer
            if self.selected_tooth_id_3dbar is not None:
                logging.info(f"--- 3DBarVizQt: Deselecting tooth {self.selected_tooth_id_3dbar} (unrecognized actor click). ---")
            self.selected_tooth_id_3dbar = None
            
        # Signal MainAppWindow (via main_app_window_ref) to update other UI parts
        if self.main_app_window_ref: 
            # 1. Update the graph
            self.main_app_window_ref.update_graph_on_click(self.selected_tooth_id_3dbar) # Pass current selection
            
            # 2. Update the detailed info panel in MainAppWindow
            detail_info_text = "Click a tooth/bar for details." # Default message
            if self.selected_tooth_id_3dbar is not None:
                timestamp_for_info = self.last_animated_timestamp 
                if timestamp_for_info is None and self.timestamps and len(self.timestamps) > 0 : 
                    # If animation hasn't started, use the current index (likely 0)
                    ts_idx = self.current_timestamp_idx if self.current_timestamp_idx < len(self.timestamps) else 0
                    timestamp_for_info = self.timestamps[ts_idx]
                elif timestamp_for_info is None: # Absolute fallback
                    timestamp_for_info = 0.0
                
                # For 3D bar, we typically show average force for the selected tooth
                _ , avg_force_series = self.processor.get_average_force_for_tooth(self.selected_tooth_id_3dbar)
                current_avg_force = 0.0
                if self.timestamps and len(avg_force_series) == len(self.timestamps):
                    try:
                        # Find the index for the current timestamp
                        time_idx_info = np.argmin(np.abs(np.array(self.timestamps) - timestamp_for_info))
                        current_avg_force = avg_force_series[time_idx_info]
                    except Exception as e:
                        logging.debug(f"3DBarVizQt: Error getting avg force for info panel: {e}")
                
                detail_info_text = (f"3D Bar - Tooth ID: {self.selected_tooth_id_3dbar}\n"
                                    f"Avg Force @ {timestamp_for_info:.1f}s: {current_avg_force:.1f} N")
            self.main_app_window_ref.update_detailed_info(detail_info_text)
        
        # Conditional re-render if animation is paused to show highlight changes
        if self.main_app_window_ref and hasattr(self.main_app_window_ref, 'is_animating') and \
           not self.main_app_window_ref.is_animating:
            if hasattr(self.main_app_window_ref, 'force_render_vedo_views'):
                logging.info(f"3DBarVizQt (R{self.renderer_index if hasattr(self, 'renderer_index') else 'N/A'}): Click - main animation paused, requesting main Vedo render.")
                current_t_render = self.last_animated_timestamp if self.last_animated_timestamp is not None else (self.timestamps[0] if self.timestamps else 0.0)
                if current_t_render is not None: 
                    self.main_app_window_ref.force_render_vedo_views(current_t_render)