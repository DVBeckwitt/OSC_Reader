# __init__.py

# Import everything you want to expose from OSC_Reader
from .OSC_Reader import read_osc, convert_to_asc, ShapeError

# Import everything you want to expose from view_osc
from .OSC_Viewer import visualize_osc_data

# Define what is publicly accessible via the package
__all__ = ["read_osc", "convert_to_asc", "ShapeError", "visualize_osc_data"]
