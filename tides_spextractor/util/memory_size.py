'''
Author: Harry Addison
Created: 18/06/2025
'''


def estimate_table_size(table):
    """
    Estimate the memory footprint of an Astropy Table in bytes, KB, MB, GB.

    Parameters:
        table (astropy.table.Table or QTable): The table to estimate.

    Returns:
        dict: Dictionary containing size in bytes, KB, MB, GB.
    """
    total_bytes = 0

    for col in table.columns:
        data = table[col]
        # For columns stored as NumPy arrays:
        if hasattr(data, 'nbytes'):
            total_bytes += data.nbytes
        else:
            # Fallback: estimate size using sys.getsizeof
            import sys
            total_bytes += sys.getsizeof(data)

    return total_bytes
