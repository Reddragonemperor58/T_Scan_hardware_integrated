import numpy as np
from vedo import Text2D, Line, Rectangle, Text3D, Grid, Sphere, colors # Plotter not imported here
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DentalArchGridVisualizerQt:
    def __init__(self, processor, parent_plotter_instance, renderer_index):
        self.processor = processor
        self.parent_plotter = parent_plotter_instance
        self.renderer_index = renderer_index
        self.renderer = parent_plotter_instance.renderers[renderer_index] # Get the specific renderer

        if self.processor.cleaned_data is None: self.processor.create_force_matrix()
        self.num_data_teeth = len(self.processor.tooth_ids) if self.processor.tooth_ids else 0
        
        self.arch_layout_width = 16.0
        self.arch_layout_depth = 10.0
        self.tooth_cell_definitions = {} 
        
        self.max_force_for_scaling = self.processor.max_force_overall if hasattr(self.processor, 'max_force_overall') else 100.0
        
        self.timestamps = self.processor.timestamps
        self.current_timestamp_idx = 0
        self.last_animated_timestamp = None 
        self.selected_tooth_id_grid = None

        self.grid_outline_actors = {} 
        self.tooth_label_actors = {}  
        self.intra_tooth_heatmap_actors_list = [] 
        self.force_percentage_actors_list = []    
        self.force_percentage_bg_actors_list = [] 
        self.left_right_bar_actor_left = None; self.left_right_bar_actor_right = None
        self.left_bar_label_actor = None; self.right_bar_label_actor = None
        self.left_bar_percentage_actor = None; self.right_bar_percentage_actor = None
        self.cof_trajectory_line_actor = None; self.cof_current_marker_actor = None; self.time_text_actor = None   
        self.selected_tooth_info_text_actor = None        
        self.main_app_window_ref = None # Will be set by EmbeddedVedoMultiViewWidget

        if self.num_data_teeth == 0:
            logging.error(f"GridVizQt (R{self.renderer_index}): No tooth data to initialize scene.")
            if self.renderer: self.renderer.AddActor(Text2D("No Grid Data Available", c='red', s=1.5).actor)
            return 
        
        # Layout definitions are created here, ready for setup_scene
        self.tooth_cell_definitions = self._define_explicit_tscan_layout(self.num_data_teeth)

        if self.parent_plotter and self.parent_plotter.interactor: # Use parent_plotter
            # Example for making grid view strictly 2D interactive
            # This might affect all renderers if share_interactor is effectively true
            # or if only one interactor style can be set on the QVTK widget.
            # For distinct interaction styles per sub-renderer, it's more complex.
            # self.parent_plotter.interactor.SetInteractorStyle(vtk.vtkInteractorStyleImage())
            pass
        

    def set_main_app_window_ref(self, main_app_window_instance): # New method
        self.main_app_window_ref = main_app_window_instance
        
    def setup_scene(self):
        """Called once by the embedding widget to set up static elements and camera for this renderer."""
        if not self.renderer or not self.parent_plotter: 
            logging.error(f"GridVizQt (R{self.renderer_index if hasattr(self, 'renderer_index') else 'N/A'}): Renderer/ParentPlotter missing.")
            return
        if self.num_data_teeth == 0:
            logging.info(f"GridVizQt (R{self.renderer_index if hasattr(self, 'renderer_index') else 'N/A'}): No data, adding placeholder text.")
            if self.renderer: self.renderer.AddActor(Text2D("No Grid Data", c='red', s=1.5).actor)
            # Optionally set a default camera for the empty view
            self.parent_plotter.at(self.renderer_index)
            cam = self.parent_plotter.camera
            cam.SetPosition(0,0,10); cam.SetFocalPoint(0,0,0); cam.SetViewUp(0,1,0)
            cam.ParallelProjectionOn()
            cam.SetParallelScale(5)
            self.renderer.ResetCamera()
            return

        logging.info(f"GridVizQt (R{self.renderer_index}): Setting up scene...")
        self.parent_plotter.at(self.renderer_index) # Activate this renderer for plotter-level ops
        
        cam = self.parent_plotter.camera # Get the camera for the active renderer
        cam.ParallelProjectionOn() 
        
        # Remove test sphere if it was added previously
        # for actor in self.renderer.GetActors():
        #     if isinstance(actor, vtk.vtkSphereSource): # This check is too low-level
        #         self.renderer.RemoveActor(actor)
        # Better to manage test actors by name or specific reference if needed.
        # For now, assuming it's not there or was removed if this setup is called multiple times.

        self._initialize_static_grid_elements() # Adds outlines and labels to self.renderer
        self._fit_camera_to_grid() # Sets camera via self.parent_plotter.camera for this active renderer

        # --- Attempt to lock down 2D view (after fitting) ---
        # cam is already self.parent_plotter.camera for the active renderer
        if cam: # Ensure cam object is valid
            # These methods are on the vtkCamera object
            cam.SetFreezeFocalPoint(True) 
            # cam.SetFreezePosition(True) # This would prevent panning and zooming by mouse drag
            # cam.SetFreezeนู่นนี่นั่น(True) # This would prevent rotation if the style allowed it
            logging.info(f"GridVizQt (R{self.renderer_index}): Camera focal point frozen.")
        # --- End Lock Down ---

        if self.tooth_cell_definitions and hasattr(self.processor, 'calculate_cof_trajectory'):
            self.processor.calculate_cof_trajectory(self.tooth_cell_definitions)
        
        # if self.parent_plotter and hasattr(self, '_on_mouse_click'):
        #      self.parent_plotter.add_callback('mouse click', self._on_mouse_click)
        
        actor_collection = self.renderer.GetActors()
        num_actors = actor_collection.GetNumberOfItems()
        logging.info(f"GridVizQt (R{self.renderer_index}): Scene setup complete. Renderer actors: {num_actors}")


    def _initialize_static_grid_elements(self): 
        if not self.tooth_cell_definitions or not self.renderer: 
            logging.warning(f"GridVizQt (R{self.renderer_index if hasattr(self, 'renderer_index') else 'N/A'}): Cannot initialize static elements - no definitions or renderer.")
            return
        
        # Clear previous static actors by removing their VTK actors from the renderer
        if self.grid_outline_actors: 
            for vedo_outline_object in self.grid_outline_actors.values():
                if hasattr(vedo_outline_object, 'actor') and vedo_outline_object.actor:
                    self.renderer.RemoveActor(vedo_outline_object.actor)
        if self.tooth_label_actors:
            for vedo_label_object in self.tooth_label_actors.values():
                if hasattr(vedo_label_object, 'actor') and vedo_label_object.actor:
                    self.renderer.RemoveActor(vedo_label_object.actor)
        
        self.grid_outline_actors.clear()
        self.tooth_label_actors.clear()
        
        for _layout_idx, cell_prop in self.tooth_cell_definitions.items():
            tooth_id=cell_prop['actual_id']; cx,cy=cell_prop['center']; w,h=cell_prop['width'],cell_prop['height']
            p1,p2=(cx-w/2,cy-h/2),(cx+w/2,cy+h/2); 
            outline=Rectangle(p1,p2,c=(0.3,0.3,0.3),alpha=0.8).lw(1.0)
            outline.name=f"Outline_Tooth_{tooth_id}";outline.pickable=True
            self.grid_outline_actors[tooth_id]=outline
            # --- CORRECTED: Add individual VTK actor ---
            if hasattr(outline, 'actor'): self.renderer.AddActor(outline.actor)

            lbl_pos=(cx,cy+h*0.5+0.2); txt_s=h*0.30; txt_s=max(0.25,min(txt_s,0.5)) 
            lbl=Text3D(str(tooth_id),pos=(lbl_pos[0],lbl_pos[1],0.12),s=txt_s,c=(0.05,0.05,0.3),justify='cc',depth=0.01)
            lbl.name = f"Label_Tooth_{tooth_id}"; lbl.pickable = False 
            self.tooth_label_actors[tooth_id]=lbl
            # --- CORRECTED: Add individual VTK actor ---
            if hasattr(lbl, 'actor'): self.renderer.AddActor(lbl.actor)
        
        actor_collection = self.renderer.GetActors()
        num_actors = actor_collection.GetNumberOfItems()
        logging.info(f"GridVizQt (R{self.renderer_index if hasattr(self, 'renderer_index') else 'N/A'}): Static elements initialized. Actors in this renderer: {num_actors}")
        
        # If you need to iterate and log details (example):
        # actor_collection.InitTraversal()
        # an_actor = actor_collection.GetNextActor()
        # while an_actor:
        #     # Note: Getting name back from a raw vtkActor is not straightforward unless set via UserData or similar
        #     # Vedo objects store .name, but vtkActor itself doesn't have a .name property directly.
        #     # logging.info(f"GridVizQt: Actor in renderer: {type(an_actor)}") 
        #     an_actor = actor_collection.GetNextActor()
        # --- END CORRECTED LOGGING ---

    def render_arch(self, timestamp): # Updates actors on self.renderer
        if not self.tooth_cell_definitions or not self.renderer: return
        self.parent_plotter.at(self.renderer_index) # Set active renderer for parent plotter context
        
        # 1. Clear all previously added dynamic actors from this frame from self.renderer
        actors_to_remove_vtk = []
        if self.time_text_actor: actors_to_remove_vtk.append(self.time_text_actor.actor)
        for act_list in [self.intra_tooth_heatmap_actors_list, 
                         self.force_percentage_bg_actors_list, 
                         self.force_percentage_actors_list]:
            for act in act_list: actors_to_remove_vtk.append(act.actor)
        
        dynamic_lr_actors = [self.left_right_bar_actor_left, self.left_right_bar_actor_right,
                             self.left_bar_label_actor, self.right_bar_label_actor,
                             self.left_bar_percentage_actor, self.right_bar_percentage_actor,
                             self.cof_trajectory_line_actor, self.cof_current_marker_actor,
                             self.selected_tooth_info_text_actor]
        for act in dynamic_lr_actors:
            if act: actors_to_remove_vtk.append(act.actor)
        
        for vtk_act in actors_to_remove_vtk: # Remove VTK actors
            if vtk_act: self.renderer.RemoveActor(vtk_act)

        # Reset Python lists/references
        self.intra_tooth_heatmap_actors_list.clear(); self.force_percentage_bg_actors_list.clear(); self.force_percentage_actors_list.clear()
        self.time_text_actor=None; self.left_right_bar_actor_left=None; self.left_right_bar_actor_right=None
        self.left_bar_label_actor=None; self.right_bar_label_actor=None; self.left_bar_percentage_actor=None; self.right_bar_percentage_actor=None
        self.cof_trajectory_line_actor=None; self.cof_current_marker_actor=None; self.selected_tooth_info_text_actor = None
        
        # 2. Create and collect all new actors for the current frame
        current_actors_to_add_vedo_objects = [] # Store Vedo objects
        
        self.time_text_actor = Text2D(f"Time: {timestamp:.1f}s",pos="bottom-left",c='k',bg=(1,1,1),alpha=0.7,s=0.7)
        current_actors_to_add_vedo_objects.append(self.time_text_actor)
        
        ordered_pairs, forces_all_sensor_points = self.processor.get_all_forces_at_time(timestamp)
        if not ordered_pairs: 
            if current_actors_to_add_vedo_objects and self.renderer:
                 for vo in current_actors_to_add_vedo_objects: self.renderer.AddActor(vo.actor)
            return
        
        # ... (rest of logic to calculate forces, create heatmaps, percentages, L/R bars, COF) ...
        # ... (When creating actors, append the Vedo object to current_actors_to_add_vedo_objects) ...
        force_map_all_sensors = {p:f for p,f in zip(ordered_pairs,forces_all_sensor_points)}
        total_force_on_arch_this_step = sum(f for f in forces_all_sensor_points if np.isfinite(f) and f > 0); total_force_on_arch_this_step = max(total_force_on_arch_this_step,1e-6)
        force_left_side=0; force_right_side=0
        for _layout_idx, cell_prop in self.tooth_cell_definitions.items():
            tooth_id = cell_prop['actual_id']
            outline_actor = self.grid_outline_actors.get(tooth_id) 
            if outline_actor:
                if self.selected_tooth_id_grid == tooth_id: outline_actor.color('lime').lw(3.0).alpha(1.0) 
                else: outline_actor.color((0.3,0.3,0.3)).lw(1.0).alpha(0.8)
            current_tooth_total_force = 0
            sensor_ids_for_this_tooth = [spid for tid,spid in self.processor.ordered_tooth_sensor_pairs if tid==tooth_id]
            forces_on_this_tooth_sensors = {spid: np.nan_to_num(force_map_all_sensors.get((tooth_id,spid),0.0)) for spid in sensor_ids_for_this_tooth}
            current_tooth_total_force = sum(forces_on_this_tooth_sensors.values())
            if cell_prop['center'][0] < -0.01: force_right_side += current_tooth_total_force
            elif cell_prop['center'][0] > 0.01: force_left_side += current_tooth_total_force
            else: force_left_side+=current_tooth_total_force/2.0; force_right_side+=current_tooth_total_force/2.0
            heatmap_actor = self._create_intra_tooth_heatmap(cell_prop, forces_on_this_tooth_sensors)
            if heatmap_actor:
                if self.selected_tooth_id_grid == tooth_id: heatmap_actor.alpha(1.0)
                else: heatmap_actor.alpha(0.75)
                self.intra_tooth_heatmap_actors_list.append(heatmap_actor) 
            perc = (current_tooth_total_force/total_force_on_arch_this_step)*100
            text_s = cell_prop['height']*0.20; text_s = max(0.20,min(text_s,0.45)) 
            perc_pos_xy = (cell_prop['center'][0],cell_prop['center'][1]-cell_prop['height']*0.70); pz = 0.16 
            num_chars=len(f"{perc:.1f}%"); bg_w_est=text_s*num_chars*0.50; bg_h_est=text_s*1.0
            bg_w_est=max(cell_prop['width']*0.25,bg_w_est); bg_h_est=max(cell_prop['height']*0.15,bg_h_est)
            p1_bg=(perc_pos_xy[0]-bg_w_est/2,perc_pos_xy[1]-bg_h_est/2);p2_bg=(perc_pos_xy[0]+bg_w_est/2,perc_pos_xy[1]+bg_h_est/2)
            pbg_rgb=(0.95,0.95,0.85);pbg_a=0.75
            p_bg=Rectangle(p1_bg,p2_bg,c=pbg_rgb,alpha=pbg_a);p_bg.z(pz-0.02) 
            self.force_percentage_bg_actors_list.append(p_bg)
            p_lbl=Text3D(f"{perc:.1f}%",pos=(perc_pos_xy[0],perc_pos_xy[1],pz),s=text_s,c='k',justify='cc',depth=0.01) 
            self.force_percentage_actors_list.append(p_lbl)
        
        current_actors_to_add_vedo_objects.extend(self.intra_tooth_heatmap_actors_list)
        current_actors_to_add_vedo_objects.extend(self.force_percentage_bg_actors_list)
        current_actors_to_add_vedo_objects.extend(self.force_percentage_actors_list)

        perc_l=(force_left_side/total_force_on_arch_this_step)*100; perc_r=(force_right_side/total_force_on_arch_this_step)*100
        if self.tooth_cell_definitions:
            min_y_overall=min(p['center'][1]-p['height']/2 for p in self.tooth_cell_definitions.values())
            bar_base_y=min_y_overall-1.8; bar_overall_width=self.arch_layout_width*0.30; bar_max_h=0.8 
            left_bar_h=max(0.02,(perc_l/100.0)*bar_max_h); right_bar_h=max(0.02,(perc_r/100.0)*bar_max_h)
            bar_label_s=0.25; bar_perc_s=0.22
            bar_z=0.05; text_on_bar_z=bar_z+0.02; label_above_bar_z=bar_z+0.03
            l_bar_cx=-bar_overall_width*0.8; l_p1=(l_bar_cx-bar_overall_width/2,bar_base_y); l_p2=(l_bar_cx+bar_overall_width/2,bar_base_y+left_bar_h)
            self.left_right_bar_actor_left=Rectangle(l_p1,l_p2,c='g',alpha=0.85).z(bar_z)
            left_label_pos=(l_bar_cx,bar_base_y+left_bar_h+0.20,label_above_bar_z); self.left_bar_label_actor=Text3D("Left",pos=left_label_pos,s=bar_label_s,c='k',justify='cb',depth=0.01)
            if left_bar_h>0.02: left_perc_pos=(l_bar_cx,bar_base_y+left_bar_h/2,text_on_bar_z); self.left_bar_percentage_actor=Text3D(f"{perc_l:.0f}%",pos=left_perc_pos,s=bar_perc_s,c='w',justify='cc',depth=0.01)
            else: self.left_bar_percentage_actor = None
            r_bar_cx=bar_overall_width*0.8; r_p1=(r_bar_cx-bar_overall_width/2,bar_base_y); r_p2=(r_bar_cx+bar_overall_width/2,bar_base_y+right_bar_h)
            self.left_right_bar_actor_right=Rectangle(r_p1,r_p2,c='r',alpha=0.85).z(bar_z)
            right_label_pos=(r_bar_cx,bar_base_y+right_bar_h+0.20,label_above_bar_z); self.right_bar_label_actor=Text3D("Right",pos=right_label_pos,s=bar_label_s,c='k',justify='cb',depth=0.01)
            if right_bar_h > 0.02: right_perc_pos=(r_bar_cx,bar_base_y+right_bar_h/2,text_on_bar_z); self.right_bar_percentage_actor=Text3D(f"{perc_r:.0f}%",pos=right_perc_pos,s=bar_perc_s,c='w',justify='cc',depth=0.01)
            else: self.right_bar_percentage_actor = None
            current_actors_to_add_vedo_objects.extend(filter(None,[self.left_right_bar_actor_left,self.left_bar_label_actor,self.left_bar_percentage_actor, self.left_right_bar_actor_right,self.right_bar_label_actor,self.right_bar_percentage_actor]))
        
        cof_pts=self.processor.get_cof_up_to_timestamp(timestamp)
        if len(cof_pts)>1: cof_ln_pts=[(p[0],p[1],0.25) for p in cof_pts];self.cof_trajectory_line_actor=Line(cof_ln_pts,c=(0.8,0.1,0.8),lw=2,alpha=0.6); current_actors_to_add_vedo_objects.append(self.cof_trajectory_line_actor)
        if cof_pts: cx_cof,cy_cof=cof_pts[-1];self.cof_current_marker_actor=Sphere(pos=(cx_cof,cy_cof,0.27),r=0.10,c='darkred',alpha=0.9); current_actors_to_add_vedo_objects.append(self.cof_current_marker_actor)
        
        if current_actors_to_add_vedo_objects and self.renderer: 
            for vedo_obj in current_actors_to_add_vedo_objects:
                if hasattr(vedo_obj, 'actor'): # Check if it's a Vedo object with a .actor attribute
                    self.renderer.AddActor(vedo_obj.actor)
                else: # Should not happen for Vedo shapes
                    logging.warning(f"Trying to add non-actor to renderer: {type(vedo_obj)}")
        # No self.renderer.render() - let Qt widget handle it.
        


    def _fit_camera_to_grid(self): 
        if not self.tooth_cell_definitions or not self.parent_plotter or not self.renderer: return
        
        # --- Ensure correct renderer is active for camera manipulation ---
        self.parent_plotter.at(self.renderer_index)
        # ---

        all_props = list(self.tooth_cell_definitions.values())
        if not all_props: 
            self.parent_plotter.camera.SetParallelScale(10) # Use parent_plotter.camera
            self.renderer.ResetCameraClippingRange() # Important for sub-renderers
            return

        all_x = [p['center'][0]+p['width']/2 for p in all_props]+[p['center'][0]-p['width']/2 for p in all_props]
        all_y = [p['center'][1]+p['height']/2 for p in all_props]+[p['center'][1]-p['height']/2 for p in all_props]
        min_y_bottom = min(p['center'][1]-p['height']/2 for p in all_props) if all_props else 0
        all_y.append(min_y_bottom - 3.0) 
        
        if not all_x or not all_y: 
            self.parent_plotter.camera.SetParallelScale(10)
            self.renderer.ResetCameraClippingRange()
            return

        min_x,max_x=min(all_x),max(all_x); min_y,max_y=min(all_y),max(all_y)
        pad=1.15; vh=(max_y-min_y)*pad; vw=(max_x-min_x)*pad
        scale=max(vh,vw,1.0)/2.0; scale=max(0.1,scale) # Ensure positive scale
        fx,fy=(min_x+max_x)/2,(min_y+max_y)/2

        cam = self.parent_plotter.camera # Use the camera of the active renderer
        cam.SetParallelScale(scale)
        cam.SetFocalPoint(fx,fy,0)
        cam.SetPosition(fx,fy,10) # Z=10 for 2D view from front
        
        self.renderer.ResetCamera() # Apply settings to this specific renderer's camera view
        self.renderer.ResetCameraClippingRange()
        logging.info(f"GridVizQt (R{self.renderer_index}): Camera fit. Pos:{cam.GetPosition()} FP:{cam.GetFocalPoint()} Scale:{cam.GetParallelScale()}")


    def _get_arch_positions_for_layout(self, num_teeth, arch_width, arch_depth): # Same
        # ... (as before) ...
        if num_teeth == 0: return np.array([])
        x_coords = np.array([0.0]) if num_teeth==1 else np.linspace(-arch_width/2,arch_width/2,num_teeth)
        k = arch_depth/((arch_width/2)**2) if arch_width!=0 else 0
        return np.array([[x, arch_depth - k*(x**2), 0] for x in x_coords])

    def _define_explicit_tscan_layout(self, num_teeth_from_data): # Same
        # ... (Your carefully tuned layout logic) ...
        layout = {} 
        if num_teeth_from_data == 0 or not self.processor.tooth_ids: return layout
        base_arch_w_centers=self.arch_layout_width*0.80; base_arch_d_centers=self.arch_layout_depth*0.70
        arch_centers_3d = self._get_arch_positions_for_layout(num_teeth_from_data, base_arch_w_centers, base_arch_d_centers)
        arch_centers_xy = [ac[:2] for ac in arch_centers_3d]
        # ... (rest of the layout logic as in your working version) ...
        if num_teeth_from_data > 1:
            sorted_x_centers = sorted([c[0] for c in arch_centers_xy]); dx = np.abs(np.diff(sorted_x_centers))
            avg_spacing_x = np.mean(dx) if len(dx) > 0 else base_arch_w_centers / num_teeth_from_data
            base_cell_w = avg_spacing_x * 0.90; base_cell_h = base_cell_w * 1.1
        else: base_cell_w = self.arch_layout_width * 0.15; base_cell_h = self.arch_layout_depth * 0.15
        base_cell_w = max(0.7, base_cell_w); base_cell_h = max(0.9, base_cell_h)
        for i in range(num_teeth_from_data):
            actual_id = self.processor.tooth_ids[i]; center_xy = arch_centers_xy[i]
            current_w = base_cell_w; current_h = base_cell_h
            norm_x = abs(center_xy[0]) / (base_arch_w_centers / 2.0) if base_arch_w_centers > 0 else 0
            w_scale=1.0; h_scale=1.0
            if norm_x > 0.75: w_scale=1.35; h_scale=0.85
            elif norm_x > 0.50: w_scale=1.1; h_scale=1.0
            elif norm_x < 0.10: w_scale=0.70; h_scale=1.20
            elif norm_x < 0.35: w_scale=0.85; h_scale=1.10
            final_w=current_w*w_scale; final_h=current_h*h_scale
            final_w=max(0.6,final_w); final_h=max(0.8,final_h)
            layout[i]={'center':center_xy,'width':final_w,'height':final_h,'actual_id':actual_id}
        return layout

    def _create_intra_tooth_heatmap(self, cell_prop, forces_on_this_tooth_sensors):
        # This method creates and returns a new Grid actor for the heatmap.
        # It's called by render_arch each frame in the "recreate actors" model.

        if not forces_on_this_tooth_sensors: 
            # logging.debug(f"No forces_on_this_tooth_sensors for cell {cell_prop.get('actual_id', 'Unknown')}")
            return None
        
        # --- DEFINE num_sp AT THE BEGINNING of the function's logic ---
        num_sp = len(forces_on_this_tooth_sensors)
        if num_sp == 0: 
            # logging.debug(f"Zero sensor points with force data for cell {cell_prop.get('actual_id', 'Unknown')}")
            return None # No data to create a heatmap from
        # --- END DEFINITION ---
        
        cx, cy = cell_prop['center']
        cw, ch = cell_prop['width'], cell_prop['height']
        
        # For a 2x2 point grid (4 points total, forming 1 quad cell for heatmap)
        # we need 1 division in x and 1 division in y for the Grid constructor.
        heatmap_grid_resolution_param = (1, 1) 
        heatmap_grid = Grid(s=[cw * 0.96, ch * 0.96], res=heatmap_grid_resolution_param) # Slightly smaller
        heatmap_grid.pos(cx, cy, 0.05).lw(0).alpha(0.95) # z=0.05 to be above outline slightly
        heatmap_grid.name = f"Heatmap_Tooth_{cell_prop['actual_id']}"
        heatmap_grid.pickable = True 
        
        grid_points_forces = np.zeros(heatmap_grid.npoints, dtype=float) # npoints should be 4 for res=(1,1)

        # Mapping 4 sensor forces to the 4 points of the Grid (res=(1,1))
        # Vedo Grid point order for res=(1,1) (a single cell with 4 points):
        # Point 0: Bottom-Left (-sx/2, -sy/2) relative to grid center
        # Point 1: Bottom-Right ( sx/2, -sy/2)
        # Point 2: Top-Left    (-sx/2,  sy/2)
        # Point 3: Top-Right   ( sx/2,  sy/2)
        # Assuming sensor_point_id 1, 2, 3, 4 in data map to a visual TL, TR, BL, BR
        if num_sp == 4: # Ensure we have exactly 4 forces for this specific mapping
            grid_points_forces[2] = forces_on_this_tooth_sensors.get(1, 0.0) # Sensor 1 (TL) -> Grid Point 2 (Top-Left)
            grid_points_forces[3] = forces_on_this_tooth_sensors.get(2, 0.0) # Sensor 2 (TR) -> Grid Point 3 (Top-Right)
            grid_points_forces[0] = forces_on_this_tooth_sensors.get(3, 0.0) # Sensor 3 (BL) -> Grid Point 0 (Bottom-Left)
            grid_points_forces[1] = forces_on_this_tooth_sensors.get(4, 0.0) # Sensor 4 (BR) -> Grid Point 1 (Bottom-Right)
        elif num_sp > 0: # Fallback if not exactly 4 sensor points (e.g. 1, 2, or 3)
            avg_force = np.mean(list(forces_on_this_tooth_sensors.values()))
            grid_points_forces.fill(avg_force)
        # If num_sp is 0 (already handled by early return), grid_points_forces would remain all zeros.
        
        heatmap_grid.pointdata["forces"] = np.nan_to_num(grid_points_forces)
        
        custom_cmap_rgb = ['darkblue', (0,0,1), (0,1,0), (1,1,0), (1,0,0)] 
        vmax_cmap = max(self.max_force_for_scaling, 1.0) # Avoid vmax=0 for colormap
        heatmap_grid.cmap(custom_cmap_rgb, "forces", vmin=0, vmax=vmax_cmap)
        
        return heatmap_grid

    def render_arch(self, timestamp):
        if not self.tooth_cell_definitions or not self.renderer: 
            if self.renderer and hasattr(self.renderer, 'GetRenderWindow') and self.renderer.GetRenderWindow(): # Check if renderer has window
                 # If called via EmbeddedVedoWidget.update_view, this render might be handled there.
                 # Only render if it's a direct call and the window exists.
                 # self.renderer.GetRenderWindow().Render() 
                 pass 
            return
        
        # Activate this visualizer's renderer context via the parent plotter
        if self.parent_plotter:
            self.parent_plotter.at(self.renderer_index)

        # 1. Clear all DYNAMIC actors from the previous frame from self.renderer
        actors_to_remove_vtk = [] 
        
        # Time text is always recreated
        if self.time_text_actor: actors_to_remove_vtk.append(self.time_text_actor.actor)
        
        # Actors stored in lists are fully recreated
        for actor_list in [self.intra_tooth_heatmap_actors_list, 
                           self.force_percentage_bg_actors_list, 
                           self.force_percentage_actors_list]:
            for act in actor_list: 
                if hasattr(act, 'actor'): actors_to_remove_vtk.append(act.actor)
        
        # L/R bar related actors are fully recreated
        lr_actors_to_clear_refs = [
            self.left_right_bar_actor_left, self.left_right_bar_actor_right,
            self.left_bar_label_actor, self.right_bar_label_actor,
            self.left_bar_percentage_actor, self.right_bar_percentage_actor,
            self.cof_trajectory_line_actor, self.cof_current_marker_actor,
            self.selected_tooth_info_text_actor # This is also dynamic based on click
        ]
        for act in lr_actors_to_clear_refs:
            if act and hasattr(act, 'actor'): actors_to_remove_vtk.append(act.actor)
        
        for vtk_act in actors_to_remove_vtk: 
            if vtk_act: self.renderer.RemoveActor(vtk_act)

        # Reset Python lists/references for actors that are recreated each frame
        self.intra_tooth_heatmap_actors_list.clear() 
        self.force_percentage_bg_actors_list.clear() 
        self.force_percentage_actors_list.clear()
        self.time_text_actor=None; self.left_right_bar_actor_left=None; self.left_right_bar_actor_right=None
        self.left_bar_label_actor=None; self.right_bar_label_actor=None; self.left_bar_percentage_actor=None; self.right_bar_percentage_actor=None
        self.cof_trajectory_line_actor=None; self.cof_current_marker_actor=None
        # self.selected_tooth_info_text_actor is handled by _on_mouse_click for its recreation/removal

        # This list will collect Vedo objects of NEW actors to be added in THIS frame
        current_vedo_objects_to_add = [] 
        
        # Create and add Time Text
        self.time_text_actor = Text2D(f"Time: {timestamp:.1f}s",pos="bottom-left",c='k',bg=(1,1,1),alpha=0.7,s=0.7)
        current_vedo_objects_to_add.append(self.time_text_actor)
        
        # Fetch forces for the current timestamp
        ordered_pairs, forces_all_sensor_points = self.processor.get_all_forces_at_time(timestamp)
        if not ordered_pairs: 
            if current_vedo_objects_to_add and self.renderer:
                for vo in current_vedo_objects_to_add: self.renderer.AddActor(vo.actor)
            # No explicit render call here; let the Qt widget handle it
            return
        
        force_map_all_sensors = {p:f for p,f in zip(ordered_pairs,forces_all_sensor_points)}
        total_force_on_arch_this_step = sum(f for f in forces_all_sensor_points if np.isfinite(f) and f > 0)
        total_force_on_arch_this_step = max(total_force_on_arch_this_step, 1e-6)
        
        force_left_side = 0.0
        force_right_side = 0.0

        # Loop through tooth cells to update/create heatmaps and per-tooth percentages
        for _layout_idx, cell_prop in self.tooth_cell_definitions.items():
            tooth_id = cell_prop['actual_id']
            
            # Update Highlight for STATIC Outline Actor (already in renderer)
            outline_actor = self.grid_outline_actors.get(tooth_id) 
            if outline_actor:
                if self.selected_tooth_id_grid == tooth_id:
                    outline_actor.color('lime').lw(3.0).alpha(1.0) 
                else:
                    outline_actor.color((0.3,0.3,0.3)).lw(1.0).alpha(0.8)
            
            current_tooth_total_force = 0.0
            sensor_ids_for_this_tooth = [spid for tid,spid in self.processor.ordered_tooth_sensor_pairs if tid==tooth_id]
            forces_on_this_tooth_sensors = {spid: np.nan_to_num(force_map_all_sensors.get((tooth_id,spid),0.0)) for spid in sensor_ids_for_this_tooth}
            current_tooth_total_force = sum(forces_on_this_tooth_sensors.values())

            # Accumulate Left/Right forces
            if cell_prop['center'][0] < -0.01: force_right_side += current_tooth_total_force
            elif cell_prop['center'][0] > 0.01: force_left_side += current_tooth_total_force
            else: force_left_side+=current_tooth_total_force/2.0; force_right_side+=current_tooth_total_force/2.0
            
            # Create new heatmap actor for this frame
            heatmap_actor = self._create_intra_tooth_heatmap(cell_prop, forces_on_this_tooth_sensors)
            if heatmap_actor:
                if self.selected_tooth_id_grid == tooth_id: heatmap_actor.alpha(1.0) # Apply highlight alpha
                else: heatmap_actor.alpha(0.75)
                self.intra_tooth_heatmap_actors_list.append(heatmap_actor) 
            
            # Create new per-tooth percentage text and its background for this frame
            perc = (current_tooth_total_force/total_force_on_arch_this_step)*100
            text_s = cell_prop['height']*0.20; text_s = max(0.20,min(text_s,0.45)) 
            perc_pos_xy = (cell_prop['center'][0],cell_prop['center'][1]-cell_prop['height']*0.70); pz = 0.16 
            num_chars=len(f"{perc:.1f}%"); bg_w_est=text_s*num_chars*0.50; bg_h_est=text_s*1.0 # Heuristic width
            bg_w_est=max(cell_prop['width']*0.25,bg_w_est); bg_h_est=max(cell_prop['height']*0.15,bg_h_est)
            p1_bg=(perc_pos_xy[0]-bg_w_est/2,perc_pos_xy[1]-bg_h_est/2);p2_bg=(perc_pos_xy[0]+bg_w_est/2,perc_pos_xy[1]+bg_h_est/2)
            pbg_rgb=(0.95,0.95,0.85);pbg_a=0.75
            p_bg=Rectangle(p1_bg,p2_bg,c=pbg_rgb,alpha=pbg_a);p_bg.z(pz-0.02) 
            self.force_percentage_bg_actors_list.append(p_bg)
            p_lbl=Text3D(f"{perc:.1f}%",pos=(perc_pos_xy[0],perc_pos_xy[1],pz),s=text_s,c='k',justify='cc',depth=0.01) 
            self.force_percentage_actors_list.append(p_lbl)
        
        # Add collected per-tooth dynamic actors to the main list for this frame's plotter.add()
        current_vedo_objects_to_add.extend(self.intra_tooth_heatmap_actors_list)
        current_vedo_objects_to_add.extend(self.force_percentage_bg_actors_list)
        current_vedo_objects_to_add.extend(self.force_percentage_actors_list)

        # Create L/R Distribution Bars and Text (recreated each frame)
        perc_l=(force_left_side/total_force_on_arch_this_step)*100
        perc_r=(force_right_side/total_force_on_arch_this_step)*100
        
        if self.tooth_cell_definitions: # This check is good
            min_y_overall=min(p['center'][1]-p['height']/2 for p in self.tooth_cell_definitions.values())
            bar_base_y=min_y_overall-1.8; bar_overall_width=self.arch_layout_width*0.30; bar_max_h=0.8 
            left_bar_h=max(0.02,(perc_l/100.0)*bar_max_h); right_bar_h=max(0.02,(perc_r/100.0)*bar_max_h)
            bar_label_s=0.25; bar_perc_s=0.22 # Adjusted from previous version
            bar_z=0.05; text_on_bar_z=bar_z+0.02; label_above_bar_z=bar_z+0.03

            l_bar_cx=-bar_overall_width*0.8; 
            l_p1=(l_bar_cx-bar_overall_width/2,bar_base_y); l_p2=(l_bar_cx+bar_overall_width/2,bar_base_y+left_bar_h)
            self.left_right_bar_actor_left=Rectangle(l_p1,l_p2,c='g',alpha=0.85).z(bar_z)
            left_label_pos=(l_bar_cx,bar_base_y+left_bar_h+0.20,label_above_bar_z); 
            self.left_bar_label_actor=Text3D("Left",pos=left_label_pos,s=bar_label_s,c='k',justify='cb',depth=0.01)
            if left_bar_h > 0.02: 
                left_perc_pos=(l_bar_cx,bar_base_y+left_bar_h/2,text_on_bar_z); 
                self.left_bar_percentage_actor=Text3D(f"{perc_l:.0f}%",pos=left_perc_pos,s=bar_perc_s,c='w',justify='cc',depth=0.01)
            else: self.left_bar_percentage_actor = None
            
            r_bar_cx=bar_overall_width*0.8; 
            r_p1=(r_bar_cx-bar_overall_width/2,bar_base_y); r_p2=(r_bar_cx+bar_overall_width/2,bar_base_y+right_bar_h)
            self.left_right_bar_actor_right=Rectangle(r_p1,r_p2,c='r',alpha=0.85).z(bar_z)
            right_label_pos=(r_bar_cx,bar_base_y+right_bar_h+0.20,label_above_bar_z); 
            self.right_bar_label_actor=Text3D("Right",pos=right_label_pos,s=bar_label_s,c='k',justify='cb',depth=0.01)
            if right_bar_h > 0.02:
                right_perc_pos=(r_bar_cx,bar_base_y+right_bar_h/2,text_on_bar_z); 
                self.right_bar_percentage_actor=Text3D(f"{perc_r:.0f}%",pos=right_perc_pos,s=bar_perc_s,c='w',justify='cc',depth=0.01)
            else: self.right_bar_percentage_actor = None
            
            current_vedo_objects_to_add.extend(filter(None,[
                self.left_right_bar_actor_left,self.left_bar_label_actor,self.left_bar_percentage_actor, 
                self.left_right_bar_actor_right,self.right_bar_label_actor,self.right_bar_percentage_actor
            ]))
        
        # COF Rendering (recreated)
        cof_pts=self.processor.get_cof_up_to_timestamp(timestamp)
        if len(cof_pts)>1: 
            cof_ln_pts=[(p[0],p[1],0.25) for p in cof_pts] # Ensure Z is high enough
            self.cof_trajectory_line_actor=Line(cof_ln_pts,c=(0.8,0.1,0.8),lw=2,alpha=0.6)
            current_vedo_objects_to_add.append(self.cof_trajectory_line_actor)
        if cof_pts: 
            cx_cof,cy_cof=cof_pts[-1]
            self.cof_current_marker_actor=Sphere(pos=(cx_cof,cy_cof,0.27),r=0.10,c='darkred',alpha=0.9)
            current_vedo_objects_to_add.append(self.cof_current_marker_actor)
        
        # Add all collected new actors for this frame to the specific renderer
        if current_vedo_objects_to_add and self.renderer: 
            for vo in current_vedo_objects_to_add:
                if hasattr(vo, 'actor'): # Vedo objects have a .actor attribute for the vtkActor
                    self.renderer.AddActor(vo.actor)
                # else: # Should not happen for standard Vedo shapes
                #     logging.warning(f"GridVizQt: Trying to add non-actor object {type(vo)} to renderer.")

        # The final render call to update the screen is handled by EmbeddedVedoMultiViewWidget.update_views()
        # which calls self.vedo_canvas.Render() after this method (via self.visualizer.animate()) completes.



    def animate(self, timestamp_to_render): # Takes timestamp directly
        # ... (same as previous correct version) ...
        if not self.timestamps: return
        self.last_animated_timestamp = timestamp_to_render
        self.render_arch(timestamp_to_render) # Updates actors on self.renderer
        
    def get_frame_as_array(self, timestamp_to_render): # Same as before, ensures render before screenshot
        # ... (same as previous correct version) ...
        if not self.renderer or not self.parent_plotter: return None
        self.parent_plotter.at(self.renderer_index) 
        self.render_arch(timestamp_to_render)    
        if self.renderer and (self.parent_plotter.window or self.parent_plotter.offscreen): # Use parent_plotter for window check
            self.parent_plotter.render() # Render the whole main plotter to ensure this sub-renderer is drawn
        return self.parent_plotter.screenshot(asarray=True) # Screenshot the whole main plotter window
    

    def _on_mouse_click(self, event): # event is vedo.interaction.Event
        # Dispatcher should ensure this event is relevant to this renderer.
        
        clicked_tooth_id_parsed = None
        
        if event.actor: 
            actor_name = event.actor.name
            logging.info(f"GridVizQt (R{self.renderer_index if hasattr(self, 'renderer_index') else 'N/A'}) Processing Click: Actor '{actor_name}' at {event.picked3d}")
            if actor_name and (actor_name.startswith("Heatmap_Tooth_") or actor_name.startswith("Outline_Tooth_")):
                try: clicked_tooth_id_parsed = int(actor_name.split("_")[-1])
                except ValueError: logging.warning(f"GridVizQt: Could not parse tooth_id from actor: {actor_name}")
        else: 
            logging.info(f"GridVizQt (R{self.renderer_index if hasattr(self, 'renderer_index') else 'N/A'}): Processing background click for its renderer.")

        # Update selection state for highlighting
        if clicked_tooth_id_parsed is not None:
            if self.selected_tooth_id_grid == clicked_tooth_id_parsed: self.selected_tooth_id_grid = None 
            else: self.selected_tooth_id_grid = clicked_tooth_id_parsed
            logging.info(f"--- GridVizQt: Tooth selection is now: {self.selected_tooth_id_grid} ---")
        elif event.actor is None : 
             if self.selected_tooth_id_grid is not None: 
                 logging.info(f"--- GridVizQt: Deselecting tooth {self.selected_tooth_id_grid} (background click). ---")
             self.selected_tooth_id_grid = None
        elif event.actor is not None and clicked_tooth_id_parsed is None: # Clicked unrecognized actor in this renderer
            if self.selected_tooth_id_grid is not None:
                logging.info(f"--- GridVizQt: Deselecting tooth {self.selected_tooth_id_grid} (unrecognized actor click). ---")
            self.selected_tooth_id_grid = None
            
        # Signal MainAppWindow
        if self.main_app_window_ref: # This should be an instance of MainAppWindow
            # 1. Update the graph
            self.main_app_window_ref.update_graph_on_click(self.selected_tooth_id_grid)
            
            # 2. Update the detailed info panel
            detail_info_text = "Click a tooth in the grid for details." # Default message
            if self.selected_tooth_id_grid is not None:
                # Determine the timestamp for which to show info
                timestamp_for_info = self.last_animated_timestamp 
                if timestamp_for_info is None: # Fallback if animation hasn't run or last_ts is None
                    if self.timestamps and len(self.timestamps) > 0 : 
                        ts_idx = self.current_timestamp_idx if self.current_timestamp_idx < len(self.timestamps) else 0
                        timestamp_for_info = self.timestamps[ts_idx]
                    else: # Absolute fallback if no timestamps available at all
                        timestamp_for_info = 0.0 
                        logging.warning("GridVizQt: No valid timestamp for detailed info, defaulting to 0.0s")
                
                info_text_lines = [f"Grid - Tooth ID: {self.selected_tooth_id_grid}", # Added "Grid - " prefix
                                   f"Forces @ {timestamp_for_info:.1f}s:"]
                total_force_on_selected_tooth = 0.0
                
                # Get sensor point IDs for this specific selected tooth
                sensor_ids_for_selected_tooth = [
                    spid for tid, spid in self.processor.ordered_tooth_sensor_pairs 
                    if tid == self.selected_tooth_id_grid
                ]
                
                # Get all forces for the determined timestamp
                _pairs, forces_at_timestamp = self.processor.get_all_forces_at_time(timestamp_for_info)
                force_map_for_timestamp = {p: f for p, f in zip(_pairs, forces_at_timestamp)}

                if not sensor_ids_for_selected_tooth:
                    info_text_lines.append("  (No sensor point data definition found for this tooth)")
                else:
                    for sp_id_actual in sensor_ids_for_selected_tooth:
                        force = force_map_for_timestamp.get((self.selected_tooth_id_grid, sp_id_actual), 0.0)
                        info_text_lines.append(f"  Sensor {sp_id_actual}: {force:.1f} N")
                        total_force_on_selected_tooth += force
                
                info_text_lines.append(f"Total on Tooth: {total_force_on_selected_tooth:.1f} N")
                detail_info_text = "\n".join(info_text_lines)
                
            # logging.debug(f"GridVizQt: Updating detailed info with: {detail_info_text}") # Add this for debugging
            self.main_app_window_ref.update_detailed_info(detail_info_text)
        else:
            logging.warning("GridVizQt: main_app_window_ref not set. Cannot update graph or info panel.")
        
        # Conditional re-render if animation is paused
        if self.main_app_window_ref and hasattr(self.main_app_window_ref, 'is_animating') and \
           not self.main_app_window_ref.is_animating:
            if hasattr(self.main_app_window_ref, 'force_render_vedo_views'):
                logging.info(f"GridVizQt (R{self.renderer_index if hasattr(self, 'renderer_index') else 'N/A'}): Click - main animation paused, requesting main Vedo render.")
                current_t_render = self.last_animated_timestamp 
                if current_t_render is None and self.timestamps and len(self.timestamps) > 0:
                    current_t_render = self.timestamps[self.current_timestamp_idx if self.current_timestamp_idx < len(self.timestamps) else 0]
                elif current_t_render is None:
                    current_t_render = 0.0
                
                if current_t_render is not None: 
                    self.main_app_window_ref.force_render_vedo_views(current_t_render)