# This is needed for get_var_bytes
from pathlib import Path
import sys
import re

from copy import deepcopy

# import data_tools
# Basic utilities
import numpy as np
import pandas as pd
# Configuration file functionality
import os
import yaml

# Math functions
import math

# Need these for BMI
from bmipy import Bmi
from enum import Enum

import jinja2
from jinja2 import Environment, BaseLoader, meta

from bmipy_util import GridType, Location, BmiMetadata, VarType, Grid

class Jinja(Bmi):
    _start_time = None
    _end_time = None
    _current_time = None
    _time_step = None
    _time_units = None
    _vars = None
    _recipes = None
    _bmi_type_map = None
    _bmi_grid_type_map = None
    _structured_grid_types = [
        GridType.RECTILINEAR,
        GridType.UNIFORM_RECTILINEAR,
        GridType.STRUCTURED_QUADRILATERAL
    ]
    _convention_variable_prefix = ''
    _has_updated = False

    # Consider making this static in the future, once the naming confvention is firmed up.
    _convention_regexes = None 

    def __init__(self):
        """Create a model that is ready for initialization."""
        super(Jinja, self).__init__()

        self._vars = {
            "math": {
                "pi": np.pi,
                "e": np.e,
                "euler_gamma": np.euler_gamma
            }
        }
        self._grids = {
            0: Grid(0, 0, GridType.SCALAR)
        }
        self._input_var_names = []
        self._output_var_names = []

        self._bmi_type_map = {}
        for t in VarType:
            self._bmi_type_map[t.value] = t
        self._bmi_grid_type_map = {}
        for t in GridType:
            self._bmi_grid_type_map[t.value] = t

        #TODO: change to SandboxedEnvironment once we get a minimal working example.
        self.environment = Environment(loader=BaseLoader, trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False, enable_async=False)

        self.environment.filters["cos"] = math.cos
        self.environment.filters["sin"] = math.sin

        self._convention_regexes = {
            "grid_shape": re.compile(self._convention_variable_prefix + 'grid_(\d+)_shape'),
            "grid_spacing": re.compile(self._convention_variable_prefix + 'grid_(\d+)_spacing'),
            "grid_origin": re.compile(self._convention_variable_prefix + 'grid_(\d+)_origin')
        }

    _current_time = 0

    #---------------------------------------------
    # Variables state/metadata store
    #---------------------------------------------
    _vars = None

    _grids = None

    #---------------------------------------------
    # Input variables
    #---------------------------------------------
    _input_var_names = None

    #---------------------------------------------
    # Output variables
    #---------------------------------------------
    _output_var_names = None

    
    #------------------------------------------------------------ 
    #------------------------------------------------------------ 
    #-- Non-BMI utility functions
    #------------------------------------------------------------ 
    #------------------------------------------------------------ 

    def _parse_config(self, cfg):
        if "library" in cfg:
            #TODO some validation here, probably.
            self._vars["constants"] = cfg["library"].get("constants", {})

            self._recipes = cfg["library"].get("recipes", {})

        for key, val in cfg.get("grids", {}).items():
            grid_cfg = val
            if "rank" not in grid_cfg:
                raise Exception("jinjabmi: The key ""rank"" is required for any defined grid.")
            grid_rank = grid_cfg["rank"]
            if not(isinstance(grid_rank, int)) or grid_rank < 0:
                raise Exception(f"jinjabmi: Invalid value for \"rank\": {grid_rank}")
            grid_type = grid_cfg.get("type", GridType.UNIFORM_RECTILINEAR.value)
            if grid_type not in self._bmi_grid_type_map:
                raise Exception(f"jinjabmi: Invalid value for \"type\": {grid_type}")
            grid_type = self._bmi_grid_type_map[grid_type]
            #print(f"Grid {key} type is {grid_type.value}")
            grid = Grid(key, grid_rank, grid_type)
            if "shape" in grid_cfg:
                grid_shape = grid_cfg["shape"]
                if len(grid_shape) != grid.rank:
                    raise Exception(f"jinjabmi: Grid shape array must be of the length \"rank\" which is {grid.rank} (got {len(grid_shape)})")
                grid.shape = grid_shape
            else:
                grid.shape = [1]*grid_rank
            if "spacing" in grid_cfg:
                grid_spacing = grid_cfg["spacing"]
                if len(grid_spacing) != grid.rank:
                    raise Exception(f"jinjabmi: Grid spacing array must be of the length \"rank\" which is {grid.rank} (got {len(grid_spacing)})")
                grid.spacing = grid_spacing
            else:
                grid.spacing = [1]*grid_rank
            if "origin" in grid_cfg:
                grid_origin = grid_cfg["origin"]
                if len(grid_origin) != grid.rank:
                    raise Exception(f"jinjabmi: Grid origin array must be of the length \"rank\" which is {grid.rank} (got {len(grid_origin)})")
                grid.origin = grid_origin
            else:
                grid.origin = [0]*grid_rank
            self._grids[key] = grid

        for key, val in cfg["variables"].items():
            var_cfg = val
            if "recipe" in var_cfg and "name" in var_cfg["recipe"]:
                recipe_name = var_cfg["recipe"]["name"]
                if recipe_name not in self._recipes:
                    raise KeyError(f"jinjabmi: Unknown recipe \"{recipe_name}\" in variable \"{key}\"")
                recipe_copy = deepcopy(self._recipes[recipe_name]) #deep just to be safe
                recipe_copy.update(var_cfg) #update is shallow, granted.
                var_cfg = recipe_copy
            var = {}
            self._vars[key] = var
            bmi_meta = BmiMetadata(
                is_input = var_cfg.get("input", False),
                is_output = var_cfg.get("output", False),
                vartype = self._bmi_type_map.get(var_cfg.get("type", "float64"), VarType.DOUBLE),
                units = var_cfg.get("units", '1'),
                grid = var_cfg.get("grid", 0)
            )
            var["bmi_meta"] = bmi_meta
            if "init" in var_cfg:
                var["init"] = var_cfg["init"]
            if bmi_meta.is_input :
                self._input_var_names.append(key)
            if bmi_meta.is_output :
                self._output_var_names.append(key)
            #var["template"] = self.environment.from_string(var_cfg["template"]) if "template" in var_cfg else None
            if "expression" in var_cfg:
                expr = var_cfg["expression"]
                subs = var_cfg.get("recipe", {}).get("substitutions", {}).items()
                for sub_k,sub_v in subs:
                    expr = re.sub('\{~\s*' + re.escape(sub_k) + '\s*~\}', re.escape(sub_v), expr)
                var["expression"] = expr
                var["expression_callable"] = self.environment.compile_expression(expr, undefined_to_none=False)
            else:
                #print(f"initializing var {key}")
                self._initialize_var(var)

        # Once initial parse of variables is complete, make a pass to determine dependencies...
        for var_name, var_data in self._vars.items():
            if "expression" not in var_data:
                continue
            ast = self.environment.parse("{{" + var_data["expression"] + "}}")
            deps = jinja2.meta.find_undeclared_variables(ast)
            var_data["depends"] = set()
            for dep in deps:
                if dep == var_name:
                    continue
                dep_data = self._vars.get(dep, None)
                #TODO: Throw something if missing?
                if dep_data is None or "expression" not in dep_data:
                    continue
                var_data["depends"].add(dep)
        # Now make a pass to check for cycles...
        #from pprint import pprint
        depbags = { k:v["depends"].copy() for k,v in self._vars.items() if "depends" in v }
        nonend = { k:v for k,v in depbags.items() if len(v) > 0}
        #pprint(nonend)
        while True:
            lastnonendcount = len(nonend)
            for k in nonend:
                nonend[k] = { d for d in nonend[k] if d in nonend }
                #print(f"    {k} : {nonend[k]=}")
            nonend = { k:v for k,v in nonend.items() if len(v) > 0}
            #pprint(nonend)
            #print(f" {len(nonend)=} {lastnonendcount=}")
            if len(nonend) == 0 or lastnonendcount == len(nonend):
                break
        if len(nonend) > 0:
            raise Exception(f"jinjabmi: Cycle in dependencies for one or more variables!")


        self._start_time = cfg.get("start_time", 0)
        self._end_time = cfg.get("end_time", sys.maxsize)
        self._time_step = cfg.get("time_step", 3600)
        self._time_units = cfg.get("time_units", "s")
        return cfg

    def _get_var_data(self, var_name):
        if var_name not in self._vars:
            raise KeyError(f"jinjabmi: Variable {var_name} is not configured.")
        var_data = self._vars[var_name]

        # Lazy initialization...
        if "value" not in var_data:
            self._initialize_var(var_data)

        return var_data
    
    def _initialize_var(self, var_data):
        typ = var_data["bmi_meta"].vartype.get_numpy_dtype()
        var_grid = self._get_grid_data(var_data["bmi_meta"].grid)
        if var_grid.type == GridType.SCALAR:
            var_data["value"] = np.zeros((1), typ)
        else:
            #FIXME: This is currently assumed to work under unstructured grids also, though
            # get_grid_shape is technically only for structured grids...?
            var_data["value"] = np.zeros(var_grid.shape, typ)

        if "init" in var_data:
            init_data = var_data["init"]
            v = var_data["value"]
            if isinstance(init_data, int) or isinstance(init_data, float):
                v[:] = init_data
                return
                
            #TODO: This section could use some more validation/user feedback.
            shp = v.shape
            v = v.flatten()
            rng = np.random.default_rng(seed=init_data.get("seed", None))

            range = init_data.get("range", None)
            stddev = init_data.get("stddev", None)
            if range is not None:
                try: 
                    range = np.array(range, dtype=float)
                except:
                    raise Exception("jinjabmi: init.range was specified but was not an array of two numbers (got \"{range}\")")

            if stddev is not None:
                # Doing a very naive truncated normal distribution. Better options out there.
                stddev = float(init_data["stddev"])
                maxiter = 100
                i = 0
                indices = (np.array([i[0] for i in np.ndindex(v.shape)]),) #can probably initialize to "all" smarter?
                while len(indices[0]) > 0 and i <= maxiter:
                    i += 1
                    mean = float(var_data["init"].get("mean", 0 if range is None else (range[0]+range[1])/2 ))
                    v[indices] = rng.normal(mean, stddev, size=indices[0].shape)
                    if range is not None:
                        indices = np.where(np.logical_or(v < range[0], v > range[1]))
            else:
                v[:] = rng.uniform(low=range[0], high=range[1], size=v.shape)
            var_data["value"] = v.reshape(shp)

    def _get_var_grid_data(self, var_name):
        var_data = self._get_var_data(var_name)
        return self._grids[var_data["bmi_meta"].grid]

    def _get_grid_data(self, grid_id):
        if grid_id not in self._grids:
            raise KeyError(f"jinjabmi: Grid ID {grid_id} is not configured.") 
        return self._grids[grid_id]

    def _evaluate_var_expression(self, var_name):
        var_data = self._get_var_data(var_name)
        expr = var_data["expression_callable"]
        context = self._vars

        # ensure evaluate dependencies... (note: cycle checking should have already been done!)
        for dep_name in var_data["depends"]:
            self._evaluate_var_expression(dep_name)

        #TODO: Avoid evaluating the expression more than once per timestep?


        var_data["value"][:] = expr(context)
        return var_data["value"]

    def _handle_set_convention_variable(self, var_name: str, value) -> bool:
        if self._has_updated:
            # Don't handle any convention variables after update or update_until is called the first time.
            return False

        m = re.fullmatch(self._convention_regexes["grid_shape"], var_name)
        if m is not None:
            grid_id = int(m.group(1))
            grid = self._get_grid_data(grid_id)
            if len(value) != grid.rank:
                raise Exception(f"jinjabmi: Grid shape array must be of the length \"rank\" which is {grid.rank} (got {len(value)})")
            shp_a = np.array(value)
            shp_t = tuple(value)
            grid.shape = shp_a
            # Go through all the variables that use this grid and re-init them!
            for key, var_data in self._vars.items():
                if "bmi_meta" not in var_data:
                    continue
                if var_data["bmi_meta"].grid == grid_id:
                    #if "value" in var_data:
                        #TODO: do something with any existing value?
                        #var_data["value"] = values.copy()
                    self._initialize_var(var_data)
            return True

        m = re.fullmatch(self._convention_regexes["grid_origin"], var_name)
        if m is not None:
            grid_id = int(m.group(1))
            grid = self._get_grid_data(grid_id)
            if len(value) != grid.rank:
                raise Exception(f"jinjabmi: Grid origin array must be of the length \"rank\" which is {grid.rank} (got {len(value)})")
            grid.origin = value
            return True

        m = re.fullmatch(self._convention_regexes["grid_spacing"], var_name)
        if m is not None:
            grid_id = int(m.group(1))
            grid = self._get_grid_data(grid_id)
            if len(value) != grid.rank:
                raise Exception(f"jinjabmi: Grid spacing array must be of the length \"rank\" which is {grid.rank} (got {len(value)})")
            grid.spacing = value
            return True
        
        return False
    

    #------------------------------------------------------------
    #------------------------------------------------------------
    # BMI: Model Control Functions
    #------------------------------------------------------------ 
    #------------------------------------------------------------

    #-------------------------------------------------------------------
    def initialize( self, bmi_cfg_file_name: str ):
        # -------------- A default value if left unconfigured ------------------#
        
        # -------------- Read in the config YAML -------------------------#
        if not isinstance(bmi_cfg_file_name, str) or len(bmi_cfg_file_name) == 0:
            raise RuntimeError("No BMI initialize configuration provided, nothing to do...")

        bmi_cfg_file = Path(bmi_cfg_file_name).resolve()
        if not bmi_cfg_file.is_file():
            raise RuntimeError("No configuration provided, nothing to do...")

        with bmi_cfg_file.open('r') as fp:
            #cfg_yaml = yaml.safe_load(fp)
            
            #adapted from https://stackoverflow.com/a/9577670/489116 and elsewhere
            #cfg_yaml = yaml.load(fp, Loader=IncludeLoader)
            #loader = IncludeLoader(fp)
            #cfg_yaml = loader
            loader_class = yaml.SafeLoader
            def yaml_include(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode):
                root = os.path.split(fp.name)[0]
                filename = os.path.join(root, loader.construct_scalar(node))
                with open(filename, 'r') as f:
                    return yaml.load(f, loader_class)
            loader_class.add_constructor('!include', yaml_include)
            cfg_yaml = yaml.load(fp, Loader=loader_class)
            
        self.cfg = self._parse_config(cfg_yaml)

        self._current_time = self._start_time
        self._vars["time"] = {
            "current_time": self._start_time,
            "last_update_delta": 0
        }


    #------------------------------------------------------------ 
    def update(self):
        """
        Update/advance the model by one time step.
        """
        self.update_until(self._current_time + self._time_step)
    
    #------------------------------------------------------------ 
    def update_until(self, future_time: float):
        """
        Update the model to a particular time

        Parameters
        ----------
        future_time : float
            The future time to when the model should be advanced.
        """
        self._has_updated = True
        update_delta_t = future_time - self._current_time 
        self._vars["time"]["current_time"] = future_time
        self._vars["time"]["last_update_delta"] = update_delta_t

    #------------------------------------------------------------    
    def finalize( self ):
        """Finalize model."""
        pass

    #------------------------------------------------------------    
    def get_input_var_names(self):

        return self._input_var_names

    #------------------------------------------------------------    
    def get_output_var_names(self):
 
        return self._output_var_names

    #------------------------------------------------------------ 
    def get_component_name(self):
        """Name of the component."""
        return "Jinja"

    #------------------------------------------------------------ 
    def get_input_item_count(self):
        """Get names of input variables."""
        return len(self._input_var_names)

    #------------------------------------------------------------ 
    def get_output_item_count(self):
        """Get names of output variables."""
        return len(self._output_var_names)

    #---------------------------------------- -------------------- 
    def get_value(self, var_name: str, dest: np.ndarray) -> np.ndarray:
        """Copy of values.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.
        Returns
        -------
        array_like
            Copy of values.
        """
        dest[:] = self.get_value_ptr(var_name).flatten()
        return

    #-------------------------------------------------------------------
    def get_value_ptr(self, var_name: str) -> np.ndarray:
        """Reference to values.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        array_like
            Value array.
        """
        var_data = self._get_var_data(var_name)
        #with open("jinjabmi.log", "a") as f:
        #    f.write(f"  get_value_ptr({var_name}) -> {var_data['value']}\n")
        
        if "expression" in var_data:
            # the below call mutates var_data["value"]...
            self._evaluate_var_expression(var_name)
        
        #return self._vars[var_name]["value"]
        return var_data["value"]
    
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    # BMI: Variable Information Functions
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------

    def get_var_units(self, var_name):
        var_data = self._get_var_data(var_name)
        return var_data["bmi_meta"].units
                                                             
    #-------------------------------------------------------------------
    def get_var_type(self, var_name: str) -> str:
        """Data type of variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        str
            Data type.
        """
        var_data = self._get_var_data(var_name)
        return str(var_data["bmi_meta"].vartype.value)
    
    #------------------------------------------------------------ 
    def get_var_grid(self, var_name):
        var_data = self._get_var_data(var_name)
        return var_data["bmi_meta"].grid

    #------------------------------------------------------------ 
    def get_var_itemsize(self, var_name):
        var_data = self._get_var_data(var_name)
        return np.dtype(var_data["bmi_meta"].vartype.get_numpy_dtype()).itemsize

    #------------------------------------------------------------ 
    def get_var_location(self, var_name):
        var_data = self._get_var_data(var_name)
        return var_data["bmi_meta"].location.value

    #-------------------------------------------------------------------
    def get_start_time( self ):
        return self._start_time 

    #-------------------------------------------------------------------
    def get_end_time( self ) -> float:
        return self._end_time 

    #-------------------------------------------------------------------
    def get_current_time( self ):
        return self._current_time

    #-------------------------------------------------------------------
    def get_time_step( self ):
        return self._time_step

    #-------------------------------------------------------------------
    def get_time_units( self ):
        return self._time_units
       
    #-------------------------------------------------------------------
    def set_value(self, var_name, values: np.ndarray):
        """
        Set model values for the provided BMI variable.

        Parameters
        ----------
        var_name : str
            Name of model variable for which to set values.
        values : np.ndarray
              Array of new values.
        """ 
        if self._handle_set_convention_variable(var_name, values):
            return

        var_data = self._get_var_data(var_name)
        if "value" not in var_data:
            #var_data["value"] = values.copy()
            grid_data = self._get_grid_data(var_data["bmi_meta"].grid)
            var_data["value"] = np.zeros(grid_data.shape)
        #else:
        var_data["value"][:] = values

    #------------------------------------------------------------ 
    def set_value_at_indices(self, var_name: str, indices: np.ndarray, src: np.ndarray):
        """
        Set model values for the provided BMI variable at particular indices.

        Parameters
        ----------
        var_name : str
            Name of model variable for which to set values.
        indices : array_like
            Array of indices of the variable into which analogous provided values should be set.
        src : array_like
            Array of new values.
        """
        #TOOD: Validate in-range? Others? Catch errors from NumPy?
        var_data = self._get_var_data(var_name)
        v = var_data["value"]
        v[np.unravel_index(indices, v.shape)] = src

    #------------------------------------------------------------ 
    def get_var_nbytes(self, var_name) -> int:
        """
        Get the number of bytes required for a variable.
        Parameters
        ----------
        var_name : str
            Name of variable.
        Returns
        -------
        int
            Size of data array in bytes.
        """
        var_data = self._get_var_data(var_name)
        v = var_data["value"]
        g = self._get_grid_data(var_data["bmi_meta"].grid)
        return var_data["value"].nbytes

    #------------------------------------------------------------ 
    def get_value_at_indices(self, var_name: str, dest: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Get values at particular indices.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        dest : np.ndarray
            A numpy array into which to place the values.
        indices : np.ndarray
            Array of indices.
        Returns
        -------
        np.ndarray
            Values at indices.
        """
        v = self.get_value_ptr(var_name)
        dest[:] = v[np.unravel_index(indices, v.shape)]

        return


    #------------------------------------------------------------ 
    def get_grid_origin(self, grid_id, origin):
        grid_data = self._get_grid_data(grid_id)
        origin[:] = grid_data.origin

    #------------------------------------------------------------ 
    def get_grid_rank(self, grid_id):
        grid_data = self._get_grid_data(grid_id)
        return grid_data.rank

    #------------------------------------------------------------ 
    def get_grid_shape(self, grid_id, shape):
        grid_data = self._get_grid_data(grid_id)
        shape[:] = grid_data.shape

    #------------------------------------------------------------ 
    def get_grid_size(self, grid_id):
        grid_data = self._get_grid_data(grid_id)
        if grid_data.type == GridType.SCALAR:
            return 1
        if grid_data.type in self._structured_grid_types:
            return np.prod(grid_data.shape)
        # else...
        raise NotImplementedError(f"jinjabmi: get_grid_size not implemented for type {grid_data.type.value}.")

    #------------------------------------------------------------ 
    def get_grid_spacing(self, grid_id, spacing):
        grid_data = self._get_grid_data(grid_id)
        if grid_data.type == GridType.UNIFORM_RECTILINEAR:
            grid_data = self._get_grid_data(grid_id)
            spacing[:] = grid_data.spacing
        else:
            raise Exception(f"jinjabmi: Call to get_grid_spacing is not valid for type {grid_data.type.value}.")

    #------------------------------------------------------------ 
    def get_grid_type(self, grid_id):
        grid_data = self._get_grid_data(grid_id)
        return grid_data.type.value
    
    #------------------------------------------------------------ 
    def get_grid_x(self, grid_id, x_out):
        grid_data = self._get_grid_data(grid_id)
        x_out[:] = grid_data.grid_x

    #------------------------------------------------------------ 
    def get_grid_y(self, grid_id, y_out):
        grid_data = self._get_grid_data(grid_id)
        y_out[:] = grid_data.grid_y

    #------------------------------------------------------------ 
    def get_grid_z(self, grid_id, z_out):
        grid_data = self._get_grid_data(grid_id)
        z_out[:] = grid_data.grid_z

    # Remaining grid funcs do not apply for scalar or structured grids
    #   Yet all functions in the BMI must be implemented 
    #   See https://bmi.readthedocs.io/en/latest/bmi.best_practices.html          
    #------------------------------------------------------------ 
    def get_grid_edge_count(self, grid_id):
        #grid_data = self._get_grid_data(grid_id)
        #return(grid_data.?)
        raise NotImplementedError("get_grid_edge_count")

    #------------------------------------------------------------ 
    def get_grid_edge_nodes(self, grid_id, edge_nodes):
        raise NotImplementedError("get_grid_edge_nodes")

    #------------------------------------------------------------ 
    def get_grid_face_count(self, grid_id):
        raise NotImplementedError("get_grid_face_count")
    
    #------------------------------------------------------------ 
    def get_grid_face_edges(self, grid_id, face_edges):
        raise NotImplementedError("get_grid_face_edges")

    #------------------------------------------------------------ 
    def get_grid_face_nodes(self, grid_id, face_nodes):
        raise NotImplementedError("get_grid_face_nodes")
    
    #------------------------------------------------------------
    def get_grid_node_count(self, grid_id):
        raise NotImplementedError("get_grid_node_count")

    #------------------------------------------------------------ 
    def get_grid_nodes_per_face(self, grid_id, nodes_per_face):
        raise NotImplementedError("get_grid_nodes_per_face") 
    

