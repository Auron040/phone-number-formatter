import pandas as pd
import re
from typing import List, Dict, Tuple, Optional, Union, Set
from file_processor import FileProcessor

class CSVComparer:
    """Class for comparing and merging CSV files based on a key column"""
    
    def __init__(self, file_processor: FileProcessor, log_callback=None):
        self.file_processor = file_processor
        self.log_callback = log_callback
        self.file1_df = None
        self.file2_df = None
        self.file1_name = None
        self.file2_name = None
        self.key_column = None
        self.matches = None
        self.conflicts = None
        self.phone_processor = None
        self.file1_df_comp = None
        self.file2_df_comp = None
        
    def log(self, message: str):
        """Log a message using the callback if available"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)
    
    def load_files(self, file1_df: pd.DataFrame, file2_df: pd.DataFrame, 
                  file1_name: str, file2_name: str) -> Tuple[bool, Optional[str]]:
        """Load the two files to compare"""
        self.file1_name = file1_name
        self.file2_name = file2_name
        self.file1_df = file1_df
        self.file2_df = file2_df
        
        return True, None
    
    def prepare_for_comparison(self, key_column: str, normalize_phone: bool = True, 
                              country_code: str = "+49") -> Tuple[bool, Optional[str]]:
        """Prepare the files for comparison by normalizing the key column if needed"""
        self.key_column = key_column
        
        # Check if key column exists in both files
        if key_column not in self.file1_df.columns:
            return False, f"Key column '{key_column}' not found in first file"
        
        if key_column not in self.file2_df.columns:
            return False, f"Key column '{key_column}' not found in second file"
        
        # Create copies to avoid modifying the original DataFrames
        df1_comp = self.file1_df.copy()
        df2_comp = self.file2_df.copy()
        
        # Convert key columns to string type to ensure proper comparison
        df1_comp[key_column] = df1_comp[key_column].astype(str)
        df2_comp[key_column] = df2_comp[key_column].astype(str)
        
        # Normalize phone numbers if requested
        if normalize_phone:
            # Create a function to extract the local part of a phone number (without country code)
            def extract_local_part(phone_number):
                if pd.isna(phone_number) or phone_number == 'nan':
                    return phone_number
                
                # Convert to string and clean up
                phone_str = str(phone_number).strip()
                
                # Remove all non-digit characters except '+'
                digits_only = re.sub(r'[^\d+]', '', phone_str)
                
                # For UK numbers with or without + prefix
                if digits_only.startswith('+44') or digits_only.startswith('44'):
                    # Remove +44/44 prefix
                    local_part = digits_only[3:] if digits_only.startswith('+') else digits_only[2:]
                    # Remove leading 0 if present
                    if local_part.startswith('0'):
                        local_part = local_part[1:]
                    return local_part
                
                # For German numbers with or without + prefix
                if digits_only.startswith('+49') or digits_only.startswith('49'):
                    # Remove +49/49 prefix
                    local_part = digits_only[3:] if digits_only.startswith('+') else digits_only[2:]
                    # Remove leading 0 if present
                    if local_part.startswith('0'):
                        local_part = local_part[1:]
                    return local_part
                
                # For international format with 00
                if digits_only.startswith('00'):
                    # Find the country code (usually 1-3 digits after 00)
                    match = re.match(r'^00(\d{1,3})', digits_only)
                    if match:
                        country_code_len = len(match.group(1))
                        # Remove 00 + country code
                        local_part = digits_only[2 + country_code_len:]
                        # Remove leading 0 if present
                        if local_part.startswith('0'):
                            local_part = local_part[1:]
                        return local_part
                
                # For local format (starting with 0)
                if digits_only.startswith('0'):
                    return digits_only[1:]
                
                # If no country code is detected, return as is
                return digits_only
            
            # Apply the function to extract local parts for comparison
            df1_comp[key_column] = df1_comp[key_column].apply(extract_local_part)
            df2_comp[key_column] = df2_comp[key_column].apply(extract_local_part)
        
        # Store the comparison DataFrames
        self.file1_df_comp = df1_comp
        self.file2_df_comp = df2_comp
        
        return True, None
    
    def compare_files(self) -> Dict:
        """Compare the files and identify matches and conflicts"""
        if self.file1_df_comp is None or self.file2_df_comp is None:
            return {"error": "Files not prepared for comparison"}
        
        # Get unique values from key column in both files
        keys1 = set(self.file1_df_comp[self.key_column].dropna())
        keys2 = set(self.file2_df_comp[self.key_column].dropna())
        
        # Find matches and unique keys
        matching_keys = keys1.intersection(keys2)
        only_in_file1 = keys1 - keys2
        only_in_file2 = keys2 - keys1
        
        # Sort the keys for consistent display
        matching_keys_sorted = sorted(list(matching_keys))
        only_in_file1_sorted = sorted(list(only_in_file1))
        only_in_file2_sorted = sorted(list(only_in_file2))
        
        # Create a dictionary to store match information
        self.matches = {
            "total_matches": len(matching_keys),
            "only_in_file1": len(only_in_file1),
            "only_in_file2": len(only_in_file2),
            "matching_keys": matching_keys_sorted,
            "file1_only_keys": only_in_file1_sorted,
            "file2_only_keys": only_in_file2_sorted
        }
        
        # Identify conflicts in matching records
        conflicts = []
        
        # Get all columns except the key column
        columns1 = [col for col in self.file1_df.columns if col != self.key_column]
        columns2 = [col for col in self.file2_df.columns if col != self.key_column]
        
        # Find common columns
        common_columns = set(columns1).intersection(set(columns2))
        
        # Check each matching key for conflicts
        for key in matching_keys:
            # Get the rows with this key
            rows1 = self.file1_df_comp[self.file1_df_comp[self.key_column] == key]
            rows2 = self.file2_df_comp[self.file2_df_comp[self.key_column] == key]
            
            # If multiple rows match in either file, this is a complex case
            if len(rows1) > 1 or len(rows2) > 1:
                conflicts.append({
                    "key": key,
                    "type": "multiple_matches",
                    "file1_rows": len(rows1),
                    "file2_rows": len(rows2)
                })
                continue
            
            # Get the original rows from the unmodified DataFrames
            orig_idx1 = rows1.index[0]
            orig_idx2 = rows2.index[0]
            
            orig_row1 = self.file1_df.iloc[orig_idx1]
            orig_row2 = self.file2_df.iloc[orig_idx2]
            
            # Check for conflicts in common columns
            column_conflicts = []
            for col in common_columns:
                val1 = orig_row1[col]
                val2 = orig_row2[col]
                
                # Check if values are different (accounting for NaN values)
                if pd.isna(val1) and pd.isna(val2):
                    continue  # Both are NaN, no conflict
                elif pd.isna(val1) or pd.isna(val2):
                    column_conflicts.append({
                        "column": col,
                        "file1_value": val1,
                        "file2_value": val2
                    })
                elif val1 != val2:
                    column_conflicts.append({
                        "column": col,
                        "file1_value": val1,
                        "file2_value": val2
                    })
            
            if column_conflicts:
                conflicts.append({
                    "key": key,
                    "type": "value_conflicts",
                    "conflicts": column_conflicts,
                    "file1_index": int(orig_idx1),
                    "file2_index": int(orig_idx2)
                })
        
        self.conflicts = conflicts
        
        # Return comparison results
        return {
            "matches": self.matches,
            "conflicts": self.conflicts,
            "file1_columns": list(self.file1_df.columns),
            "file2_columns": list(self.file2_df.columns),
            "common_columns": list(common_columns),
            "only_in_file1_columns": list(set(columns1) - common_columns),
            "only_in_file2_columns": list(set(columns2) - common_columns)
        }
    
    def get_preview_data(self, max_rows: int = 100) -> Dict:
        """Get preview data for the comparison"""
        if self.matches is None:
            return {"error": "No comparison results available"}
        
        preview = {
            "matches": [],
            "conflicts": [],
            "only_in_file1": [],
            "only_in_file2": []
        }
        
        # Get preview of matches with conflicts
        if self.conflicts:
            for conflict in self.conflicts[:max_rows]:
                if conflict["type"] == "value_conflicts":
                    key = conflict["key"]
                    file1_idx = conflict["file1_index"]
                    file2_idx = conflict["file2_index"]
                    
                    preview["conflicts"].append({
                        "key": key,
                        "file1_data": self.file1_df.iloc[file1_idx].to_dict(),
                        "file2_data": self.file2_df.iloc[file2_idx].to_dict(),
                        "conflicts": conflict["conflicts"]
                    })
        
        # Get preview of matches without conflicts
        conflict_keys = [c["key"] for c in self.conflicts]
        non_conflict_keys = [k for k in self.matches["matching_keys"] if k not in conflict_keys]
        
        for key in non_conflict_keys[:max_rows]:
            file1_row = self.file1_df_comp[self.file1_df_comp[self.key_column] == key].iloc[0]
            file2_row = self.file2_df_comp[self.file2_df_comp[self.key_column] == key].iloc[0]
            
            preview["matches"].append({
                "key": key,
                "file1_data": self.file1_df.iloc[file1_row.name].to_dict(),
                "file2_data": self.file2_df.iloc[file2_row.name].to_dict()
            })
        
        # Get preview of records only in file 1
        for key in self.matches["file1_only_keys"][:max_rows]:
            # Make sure we're getting rows that actually match the key
            matching_rows = self.file1_df_comp[self.file1_df_comp[self.key_column] == key]
            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                preview["only_in_file1"].append({
                    "key": key,
                    "data": self.file1_df.iloc[row.name].to_dict()
                })
        
        # Get preview of records only in file 2
        for key in self.matches["file2_only_keys"][:max_rows]:
            # Make sure we're getting rows that actually match the key
            matching_rows = self.file2_df_comp[self.file2_df_comp[self.key_column] == key]
            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                preview["only_in_file2"].append({
                    "key": key,
                    "data": self.file2_df.iloc[row.name].to_dict()
                })
        
        return preview
    
    def merge_files(self, conflict_resolution: str = "file1", 
                   include_only_in_file1: bool = True,
                   include_only_in_file2: bool = True,
                   selected_columns: Optional[List[str]] = None,
                   per_conflict_resolutions: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Merge the files based on the comparison results
        
        Parameters:
        - conflict_resolution: 'file1' or 'file2' to determine which file takes precedence for conflicts
        - include_only_in_file1: Whether to include records only found in file 1
        - include_only_in_file2: Whether to include records only found in file 2
        - selected_columns: List of columns to include in the merged file (None for all columns)
        - per_conflict_resolutions: Dictionary mapping keys to resolution choices ('file1' or 'file2')
        
        Returns:
        - Merged DataFrame and statistics
        """
        if self.matches is None:
            return None, {"error": "No comparison results available"}
        
        # Determine columns to include in the merged file
        all_columns = list(set(self.file1_df.columns) | set(self.file2_df.columns))
        
        if selected_columns is None:
            # Include all columns
            columns_to_include = all_columns
        else:
            # Include only selected columns
            columns_to_include = [col for col in selected_columns if col in all_columns]
            
            # Always include the key column
            if self.key_column not in columns_to_include:
                columns_to_include.append(self.key_column)
        
        # Create an empty DataFrame for the merged data
        merged_df = pd.DataFrame(columns=columns_to_include)
        
        # Sort columns for consistent display
        columns_to_include.sort()
        
        # Move key column to the beginning
        if self.key_column in columns_to_include:
            columns_to_include.remove(self.key_column)
            columns_to_include.insert(0, self.key_column)
        
        # Statistics
        stats = {
            "total_records": 0,
            "matched_records": 0,
            "conflict_records": 0,
            "only_in_file1_records": 0,
            "only_in_file2_records": 0
        }
        
        # Process matching records
        for key in self.matches["matching_keys"]:
            # Get the rows with this key
            rows1 = self.file1_df_comp[self.file1_df_comp[self.key_column] == key]
            rows2 = self.file2_df_comp[self.file2_df_comp[self.key_column] == key]
            
            # Skip complex cases with multiple matches
            if len(rows1) > 1 or len(rows2) > 1:
                continue
            
            # Get the original rows
            orig_idx1 = rows1.index[0]
            orig_idx2 = rows2.index[0]
            
            orig_row1 = self.file1_df.iloc[orig_idx1]
            orig_row2 = self.file2_df.iloc[orig_idx2]
            
            # Check if this key has conflicts
            has_conflict = False
            for conflict in self.conflicts:
                if conflict["key"] == key and conflict["type"] == "value_conflicts":
                    has_conflict = True
                    break
            
            # Create a new row for the merged data
            new_row = {}
            
            # Determine which file to use for each column
            for col in columns_to_include:
                # If column exists only in one file, use that file's value
                if col not in self.file1_df.columns:
                    new_row[col] = orig_row2[col] if col in self.file2_df.columns else None
                elif col not in self.file2_df.columns:
                    new_row[col] = orig_row1[col]
                else:
                    # Column exists in both files
                    val1 = orig_row1[col]
                    val2 = orig_row2[col]
                    
                    # Check if values are different
                    if pd.isna(val1) and pd.isna(val2):
                        new_row[col] = None  # Both are NaN
                    elif pd.isna(val1):
                        new_row[col] = val2  # File 1 has NaN, use file 2
                    elif pd.isna(val2):
                        new_row[col] = val1  # File 2 has NaN, use file 1
                    elif val1 == val2:
                        new_row[col] = val1  # Values are the same
                    else:
                        # Values are different - conflict
                        # Check if there's a per-conflict resolution for this key
                        if per_conflict_resolutions and key in per_conflict_resolutions:
                            resolution = per_conflict_resolutions[key]
                        else:
                            resolution = conflict_resolution
                            
                        if resolution == "file1":
                            new_row[col] = val1
                        else:
                            new_row[col] = val2
            
            # Add the row to the merged DataFrame
            merged_df = pd.concat([merged_df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Update statistics
            stats["total_records"] += 1
            if has_conflict:
                stats["conflict_records"] += 1
            else:
                stats["matched_records"] += 1
        
        # Add records only in file 1 if requested
        if include_only_in_file1:
            for key in self.matches["file1_only_keys"]:
                row = self.file1_df_comp[self.file1_df_comp[self.key_column] == key]
                if len(row) == 1:
                    orig_idx = row.index[0]
                    orig_row = self.file1_df.iloc[orig_idx]
                    
                    # Create a new row with only the columns from file 1 that are in columns_to_include
                    new_row = {}
                    for col in columns_to_include:
                        if col in self.file1_df.columns:
                            new_row[col] = orig_row[col]
                        else:
                            new_row[col] = None
                    
                    # Add the row to the merged DataFrame
                    merged_df = pd.concat([merged_df, pd.DataFrame([new_row])], ignore_index=True)
                    
                    # Update statistics
                    stats["total_records"] += 1
                    stats["only_in_file1_records"] += 1
        
        # Add records only in file 2 if requested
        if include_only_in_file2:
            for key in self.matches["file2_only_keys"]:
                row = self.file2_df_comp[self.file2_df_comp[self.key_column] == key]
                if len(row) == 1:
                    orig_idx = row.index[0]
                    orig_row = self.file2_df.iloc[orig_idx]
                    
                    # Create a new row with only the columns from file 2 that are in columns_to_include
                    new_row = {}
                    for col in columns_to_include:
                        if col in self.file2_df.columns:
                            new_row[col] = orig_row[col]
                        else:
                            new_row[col] = None
                    
                    # Add the row to the merged DataFrame
                    merged_df = pd.concat([merged_df, pd.DataFrame([new_row])], ignore_index=True)
                    
                    # Update statistics
                    stats["total_records"] += 1
                    stats["only_in_file2_records"] += 1
        
        return merged_df, stats
