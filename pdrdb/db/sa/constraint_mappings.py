"""
This file contains mappings between database constraints and schema fields
"""


CONSTRAINT_NAME_MAPPING = {
    'devices_serial_number_uniq': {'column_name': 'serial_number'},
    'cities_name_uniq': {'column_name': 'name'},
    'users_lower_idx': {'column_name': 'email'},
    'users_lower_idx1': {'column_name': 'username'},
}
