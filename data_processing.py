# --- START OF FILE data_processing.py ---
import numpy as np
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor:
    def __init__(self, data):
        self.data = data 
        self.cleaned_data = None; self.force_matrix = None; self.timestamps = None
        self.tooth_ids = None; self.num_sensor_points_per_tooth_map = {} 
        self.ordered_tooth_sensor_pairs = []; self.max_force_overall = 100.0
        self.cof_trajectory = [] 

    def clean_data(self):
        if not isinstance(self.data, pd.DataFrame): logging.error("Input not DataFrame."); self.cleaned_data=pd.DataFrame(); return self.cleaned_data
        required_cols = ['timestamp','tooth_id','sensor_point_id','force','contact_time']
        if not all(col in self.data.columns for col in required_cols):
            logging.error(f"Data missing cols: {required_cols}. Got: {list(self.data.columns)}"); self.cleaned_data=pd.DataFrame(); return self.cleaned_data
        self.cleaned_data = self.data.dropna(subset=required_cols).copy()
        try:
            self.cleaned_data['timestamp']=pd.to_numeric(self.cleaned_data['timestamp'])
            self.cleaned_data['tooth_id']=pd.to_numeric(self.cleaned_data['tooth_id']).astype(int)
            self.cleaned_data['sensor_point_id']=pd.to_numeric(self.cleaned_data['sensor_point_id']).astype(int)
            self.cleaned_data['force']=pd.to_numeric(self.cleaned_data['force'],errors='coerce').astype(float)
            self.cleaned_data['contact_time']=pd.to_numeric(self.cleaned_data['contact_time'],errors='coerce').astype(float)
        except Exception as e: logging.error(f"Type conversion error: {e}"); self.cleaned_data=pd.DataFrame(); return self.cleaned_data
        self.cleaned_data.dropna(subset=['force','contact_time'], inplace=True)
        self.cleaned_data = self.cleaned_data[(self.cleaned_data['force'] >= 0) & (self.cleaned_data['contact_time'] >= 0)]
        self.cleaned_data = self.cleaned_data.drop_duplicates(subset=['timestamp','tooth_id','sensor_point_id'], keep='last')
        self.tooth_ids = sorted(self.cleaned_data['tooth_id'].unique())
        self.ordered_tooth_sensor_pairs = []
        if not self.cleaned_data.empty:
            self.num_sensor_points_per_tooth_map = self.cleaned_data.groupby('tooth_id')['sensor_point_id'].nunique().to_dict()
            for tid in self.tooth_ids:
                sp_ids = sorted(self.cleaned_data[self.cleaned_data['tooth_id']==tid]['sensor_point_id'].unique())
                for sp_id in sp_ids: self.ordered_tooth_sensor_pairs.append((tid, sp_id))
            if 'force' in self.cleaned_data.columns and not self.cleaned_data['force'].empty:
                 valid_forces = self.cleaned_data['force'][self.cleaned_data['force'] > 0]
                 self.max_force_overall = valid_forces.max() if not valid_forces.empty else 100.0
        else: self.num_sensor_points_per_tooth_map = {}
        logging.info("Data cleaned: %d rows, %d teeth. Pairs: %d. MaxF: %.1f", len(self.cleaned_data),len(self.tooth_ids),len(self.ordered_tooth_sensor_pairs),self.max_force_overall)
        return self.cleaned_data

    def create_force_matrix(self):
        if self.cleaned_data is None or self.cleaned_data.empty: self.clean_data()
        if self.cleaned_data.empty: self.force_matrix=np.array([]); self.timestamps=[]; return self.force_matrix,self.timestamps
        self.timestamps = sorted(self.cleaned_data['timestamp'].unique())
        if not self.ordered_tooth_sensor_pairs or not self.timestamps: self.force_matrix=np.array([]); return self.force_matrix,self.timestamps
        self.force_matrix = np.full((len(self.timestamps),len(self.ordered_tooth_sensor_pairs)),np.nan,dtype=float)
        try:
            pivot_data = self.cleaned_data[['timestamp','tooth_id','sensor_point_id','force']].copy()
            pivot_data['force'] = pd.to_numeric(pivot_data['force'],errors='coerce').astype(float)
            pivot_df = pivot_data.pivot_table(index='timestamp',columns=['tooth_id','sensor_point_id'],values='force')
            pivot_df.columns = pd.MultiIndex.from_tuples(pivot_df.columns)
            temp_df = pd.DataFrame(index=self.timestamps,columns=pd.MultiIndex.from_tuples(self.ordered_tooth_sensor_pairs),dtype=float)
            temp_df.update(pivot_df); self.force_matrix = temp_df.to_numpy(dtype=float)
        except Exception as e:
            logging.error(f"Pivot error: {e}. Fallback."); pair_idx={p:i for i,p in enumerate(self.ordered_tooth_sensor_pairs)}
            for r,t in enumerate(self.timestamps):
                td=self.cleaned_data[self.cleaned_data['timestamp']==t]
                for _,dp in td.iterrows(): p=(int(dp['tooth_id']),int(dp['sensor_point_id'])); self.force_matrix[r,pair_idx[p]]=float(dp['force'])
        logging.info("Force matrix: %s, dtype=%s",self.force_matrix.shape,self.force_matrix.dtype)
        return self.force_matrix,self.timestamps

    def get_average_force_for_tooth(self, tooth_id):
        if self.force_matrix is None: self.create_force_matrix()
        if self.force_matrix.size==0 or tooth_id not in self.tooth_ids: return self.timestamps or [],np.array([],dtype=float)
        indices = [i for i,(tid,_) in enumerate(self.ordered_tooth_sensor_pairs) if tid==tooth_id]
        if not indices: return self.timestamps or [],np.array([],dtype=float)
        fm_f = self.force_matrix.astype(float) if not np.issubdtype(self.force_matrix.dtype, np.floating) else self.force_matrix
        avg_f = np.nanmean(fm_f[:,indices],axis=1)
        return self.timestamps, np.nan_to_num(avg_f,nan=0.0).astype(float)
        
    def get_all_forces_at_time(self, timestamp):
        if self.force_matrix is None: self.create_force_matrix()
        if self.force_matrix.size==0 or not self.timestamps: return self.ordered_tooth_sensor_pairs,np.array([],dtype=float)
        ts_arr=np.array(self.timestamps); time_idx=np.argmin(np.abs(ts_arr-timestamp))
        forces = self.force_matrix[time_idx,:]
        return self.ordered_tooth_sensor_pairs,np.nan_to_num(forces,nan=0.0).astype(float)

    def calculate_cof_trajectory(self, tooth_cell_definitions, num_sensor_points_per_cell_layout=4):
        if self.force_matrix is None: self.create_force_matrix()
        if self.force_matrix.size == 0 or not tooth_cell_definitions:
            logging.warning("Force matrix or layout undefined for COF."); self.cof_trajectory=[]; return
        self.cof_trajectory = []
        grid_dim = int(np.sqrt(num_sensor_points_per_cell_layout)); grid_dim=max(1,grid_dim)

        # Create a map from actual_tooth_id to its layout properties (center, width, height)
        layout_map = {props['actual_id']: props for props in tooth_cell_definitions.values()}

        for ts_idx, timestamp in enumerate(self.timestamps):
            sum_fx, sum_fy, total_f_step = 0.0, 0.0, 0.0
            current_pairs, current_forces = self.get_all_forces_at_time(timestamp)
            force_map_ts = {pair: force for pair, force in zip(current_pairs, current_forces)}

            for tooth_id, cell_prop in layout_map.items(): # Iterate through teeth defined in layout
                cell_cx, cell_cy = cell_prop['center']; cell_w, cell_h = cell_prop['width'], cell_prop['height']
                sub_w, sub_h = cell_w/grid_dim, cell_h/grid_dim
                
                actual_sp_ids = sorted([spid for tid,spid in self.ordered_tooth_sensor_pairs if tid==tooth_id])

                for r_idx in range(grid_dim):
                    for c_idx in range(grid_dim):
                        sp_layout_order = r_idx * grid_dim + c_idx
                        if sp_layout_order < len(actual_sp_ids):
                            sp_id = actual_sp_ids[sp_layout_order]
                            force = force_map_ts.get((tooth_id, sp_id), 0.0)
                            if force > 1e-3:
                                sp_x = cell_cx - cell_w/2 + sub_w/2 + c_idx * sub_w
                                sp_y = cell_cy + cell_h/2 - sub_h/2 - r_idx * sub_h # r=0 is top row
                                sum_fx += force*sp_x; sum_fy += force*sp_y; total_f_step += force
            if total_f_step > 1e-3: self.cof_trajectory.append((timestamp, sum_fx/total_f_step, sum_fy/total_f_step))
        logging.info(f"COF trajectory calculated: {len(self.cof_trajectory)} points.")

    def get_cof_up_to_timestamp(self, current_timestamp):
        if not self.cof_trajectory: return []
        return [(x,y) for ts,x,y in self.cof_trajectory if ts <= current_timestamp + 1e-6]
# --- END OF FILE data_processing.py ---