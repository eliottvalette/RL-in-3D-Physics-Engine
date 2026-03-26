# quadruped_points.py
import numpy as np

BODY_LENGTH = 4.0
BODY_WIDTH = 6.0
BODY_HEIGHT = 1.0
UPPER_LEG_LENGTH = 0.8
UPPER_LEG_WIDTH = 0.8
UPPER_LEG_HEIGHT = 1.5
LOWER_LEG_LENGTH = 0.6
LOWER_LEG_WIDTH = 0.6
LOWER_LEG_HEIGHT = 2.0
Y_OFFSET = -2.5

def create_body(length, width, height, y_offset):
    """
    Creates the body vertices with configurable dimensions
    Args:
        length: length along X axis
        width: width along Z axis  
        height: height along Y axis
        y_offset: Y position of the body
    Returns:
        List of 8 body vertices
    """
    half_length = length / 2
    half_width = width / 2
    
    body_vertices = [
        # Bottom face
        np.array([-half_length, y_offset, -half_width]),           # 0: bottom front left
        np.array([-half_length, y_offset,  half_width]),           # 1: bottom front right
        np.array([ half_length, y_offset, -half_width]),           # 2: bottom back left
        np.array([ half_length, y_offset,  half_width]),           # 3: bottom back right
        # Top face
        np.array([-half_length, y_offset + height, -half_width]),  # 4: top front left
        np.array([-half_length, y_offset + height,  half_width]),  # 5: top front right
        np.array([ half_length, y_offset + height, -half_width]),  # 6: top back left
        np.array([ half_length, y_offset + height,  half_width])   # 7: top back right
    ]
    
    return body_vertices

def create_upper_legs(body_vertices, leg_length, leg_width, leg_height):
    """
    Creates the upper leg vertices positioned relative to the body
    Args:
        body_vertices: list of body vertices
        leg_length: length of each leg along X axis
        leg_width: width of each leg along Z axis
        leg_height: height of each leg along Y axis
    Returns:
        List of 4 upper leg vertex lists (one for each leg)
    """
    # Extract body dimensions and position from body vertices
    body_min_x = min(v[0] for v in body_vertices)
    body_max_x = max(v[0] for v in body_vertices)
    body_min_z = min(v[2] for v in body_vertices)
    body_max_z = max(v[2] for v in body_vertices)
    body_y = body_vertices[4][1]  # Top Y position
    
    # Calculate leg positions relative to body corners
    leg_positions = [
        # Front Right (positive X, positive Z)
        (body_max_x, body_max_z),
        # Front Left (positive X, negative Z)  
        (body_max_x, body_min_z),
        # Back Right (negative X, positive Z)
        (body_min_x, body_max_z),
        # Back Left (negative X, negative Z)
        (body_min_x, body_min_z)
    ]
    
    upper_legs = []
    
    for i, (x_pos, z_pos) in enumerate(leg_positions):
        # Determine leg orientation based on position
        if x_pos > 0:  # Front legs
            leg_x_start = x_pos
            leg_x_end = x_pos + leg_length
        else:  # Back legs
            leg_x_start = x_pos - leg_length
            leg_x_end = x_pos
            
        # Z position for leg attachment
        if z_pos > 0:  # Right legs
            leg_z_start = z_pos - leg_width/2
            leg_z_end = z_pos + leg_width/2
        else:  # Left legs
            leg_z_start = z_pos - leg_width/2
            leg_z_end = z_pos + leg_width/2
        
        # Y positions
        leg_y_top = body_y
        leg_y_bottom = body_y - leg_height
        
        # Create 8 vertices for this leg
        leg_vertices = [
            # Bottom face
            np.array([leg_x_start, leg_y_bottom, leg_z_start]),     # 0: bottom front left
            np.array([leg_x_start, leg_y_bottom, leg_z_end]),       # 1: bottom front right
            np.array([leg_x_end,   leg_y_bottom, leg_z_start]),     # 2: bottom back left
            np.array([leg_x_end,   leg_y_bottom, leg_z_end]),       # 3: bottom back right
            # Top face
            np.array([leg_x_start, leg_y_top,    leg_z_start]),     # 4: top front left
            np.array([leg_x_start, leg_y_top,    leg_z_end]),       # 5: top front right
            np.array([leg_x_end,   leg_y_top,    leg_z_start]),     # 6: top back left
            np.array([leg_x_end,   leg_y_top,    leg_z_end])        # 7: top back right
        ]
        
        upper_legs.append(leg_vertices)
    
    return upper_legs

def calculate_shoulder_positions(upper_legs):
    """
    Calculate shoulder positions at the middle of the inner face of each upper leg
    Args:
        upper_legs: list of upper leg vertex lists
    Returns:
        List of 4 shoulder positions (one for each leg)
    """
    shoulder_positions = []
    
    for i, leg_vertices in enumerate(upper_legs):
        # Determine which face is the inner face based on leg position
        # Front Right (i=0): inner face is the left face (negative Z)
        # Front Left (i=1): inner face is the right face (positive Z)
        # Back Right (i=2): inner face is the left face (negative Z)
        # Back Left (i=3): inner face is the right face (positive Z)
        
        if i in [0, 2]:  # Right legs - inner face is left face (lower Z values)
            # Use vertices 0, 2, 4, 6 (left face)
            inner_vertices = [leg_vertices[0], leg_vertices[2], leg_vertices[4], leg_vertices[6]]
        else:  # Left legs - inner face is right face (higher Z values)
            # Use vertices 1, 3, 5, 7 (right face)
            inner_vertices = [leg_vertices[1], leg_vertices[3], leg_vertices[5], leg_vertices[7]]
        
        # Calculate center of the inner face
        center_x = sum(v[0] for v in inner_vertices) / 4
        center_y = sum(v[1] for v in inner_vertices) / 4
        center_z = sum(v[2] for v in inner_vertices) / 4
        
        shoulder_positions.append(np.array([center_x, center_y, center_z]))
    
    return shoulder_positions

def calculate_elbow_positions(upper_legs):
    """
    Calculate elbow positions at the center of the lower face of each upper leg
    Args:
        upper_legs: list of upper leg vertex lists
    Returns:
        List of 4 elbow positions (one for each leg)
    """
    elbow_positions = []
    
    for leg_vertices in upper_legs:
        # Lower face vertices are 0, 1, 2, 3
        lower_vertices = [leg_vertices[0], leg_vertices[1], leg_vertices[2], leg_vertices[3]]
        
        # Calculate center of the lower face
        center_x = sum(v[0] for v in lower_vertices) / 4
        center_y = sum(v[1] for v in lower_vertices) / 4
        center_z = sum(v[2] for v in lower_vertices) / 4
        
        elbow_positions.append(np.array([center_x, center_y, center_z]))
    
    return elbow_positions

def create_lower_legs(upper_legs, leg_length, leg_width, leg_height):
    """
    Creates the lower leg vertices positioned below the upper legs
    Args:
        upper_legs: list of upper leg vertex lists
        leg_length: length of each leg along X axis
        leg_width: width of each leg along Z axis (slightly smaller than upper)
        leg_height: height of each leg along Y axis
    Returns:
        List of 4 lower leg vertex lists (one for each leg)
    """
    lower_legs = []
    
    for upper_leg in upper_legs:
        # Get the bottom face of the upper leg
        upper_bottom_center_x = sum(v[0] for v in upper_leg[:4]) / 4
        upper_bottom_center_z = sum(v[2] for v in upper_leg[:4]) / 4
        upper_bottom_y = upper_leg[0][1]  # Bottom Y of upper leg
        
        # Position lower leg below upper leg
        leg_y_bottom = upper_bottom_y - leg_height
        leg_y_top = upper_bottom_y
        
        # Center the lower leg on the upper leg
        leg_x_start = upper_bottom_center_x - leg_length/2
        leg_x_end = upper_bottom_center_x + leg_length/2
        leg_z_start = upper_bottom_center_z - leg_width/2
        leg_z_end = upper_bottom_center_z + leg_width/2
        
        # Create 8 vertices for this lower leg
        leg_vertices = [
            # Bottom face
            np.array([leg_x_start, leg_y_bottom, leg_z_start]),     # 0: bottom front left
            np.array([leg_x_start, leg_y_bottom, leg_z_end]),       # 1: bottom front right
            np.array([leg_x_end,   leg_y_bottom, leg_z_start]),     # 2: bottom back left
            np.array([leg_x_end,   leg_y_bottom, leg_z_end]),       # 3: bottom back right
            # Top face
            np.array([leg_x_start, leg_y_top,    leg_z_start]),     # 4: top front left
            np.array([leg_x_start, leg_y_top,    leg_z_end]),       # 5: top front right
            np.array([leg_x_end,   leg_y_top,    leg_z_start]),     # 6: top back left
            np.array([leg_x_end,   leg_y_top,    leg_z_end])        # 7: top back right
        ]
        
        lower_legs.append(leg_vertices)
    
    return lower_legs

def create_quadruped_vertices(body_length=BODY_LENGTH, body_width=BODY_WIDTH, body_height=BODY_HEIGHT,
                             upper_leg_length=UPPER_LEG_LENGTH, upper_leg_width=UPPER_LEG_WIDTH, upper_leg_height=UPPER_LEG_HEIGHT,
                             lower_leg_length=LOWER_LEG_LENGTH, lower_leg_width=LOWER_LEG_WIDTH, lower_leg_height=LOWER_LEG_HEIGHT,
                             y_offset=Y_OFFSET):
    """
    Creates the complete quadruped with configurable dimensions
    Args:
        body_length, body_width, body_height: body dimensions
        upper_leg_length, upper_leg_width, upper_leg_height: upper leg dimensions
        lower_leg_length, lower_leg_width, lower_leg_height: lower leg dimensions
        y_offset: Y position of the body
    Returns:
        Dictionary with all parts organized and shoulder positions
    """
    
    # Create body
    body_vertices = create_body(body_length, body_width, body_height, y_offset)
    
    # Create upper legs positioned relative to body
    upper_legs = create_upper_legs(body_vertices, upper_leg_length, upper_leg_width, upper_leg_height)
    
    # Calculate shoulder positions at the middle of inner faces
    shoulder_positions = calculate_shoulder_positions(upper_legs)
    
    # Calculate elbow positions at the center of the top face of each upper leg
    elbow_positions = calculate_elbow_positions(upper_legs)
    
    # Create lower legs positioned below upper legs
    lower_legs = create_lower_legs(upper_legs, lower_leg_length, lower_leg_width, lower_leg_height)
    
    # Group all parts
    return {
        'body': body_vertices,
        'upper_legs': upper_legs,
        'lower_legs': lower_legs,
        'shoulder_positions': shoulder_positions,
        'elbow_positions': elbow_positions,
        'all_parts': [body_vertices] + upper_legs + lower_legs
    }

def get_quadruped_vertices():
    """Returns all quadruped vertices as a flat list using default dimensions"""
    vertices_dict = create_quadruped_vertices()
    all_vertices = []
    
    for part in vertices_dict['all_parts']:
        all_vertices.extend(part)
    
    return all_vertices
