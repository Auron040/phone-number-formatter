import os
import sys
import pandas as pd
import phonenumbers
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import customtkinter as ctk
from typing import List, Dict, Tuple, Optional, Union, Set
import threading
import re
from datetime import datetime
import numpy as np

# Set appearance mode and default color theme
ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

# Constants
COUNTRY_CODES = {
    "Germany": "+49",
    "Austria": "+43", 
    "Switzerland": "+41",
    "France": "+33",
    "Italy": "+39",
    "United Kingdom": "+44",
    "United States": "+1",
    "Spain": "+34",
    "Netherlands": "+31",
    "Belgium": "+32",
    "Denmark": "+45",
    "Sweden": "+46",
    "Norway": "+47",
    "Finland": "+358",
    "Poland": "+48",
    "Czech Republic": "+420",
    "Hungary": "+36",
    "Other": "Custom"
}

class PhoneNumberProcessor:
    """Class for processing phone numbers"""
    
    def __init__(self, country_code: str):
        self.country_code = country_code
        # Remove '+' for internal processing
        self.country_code_digits = country_code.replace('+', '')
    
    def add_mode(self, phone_number: str) -> str:
        """Simply add country code to the number without modifying it"""
        if not phone_number or pd.isna(phone_number):
            return phone_number
        
        # Convert to string if it's not already
        phone_str = str(phone_number).strip()
        
        # If empty after stripping, return as is
        if not phone_str:
            return phone_str
            
        # If already has the country code, return as is
        if phone_str.startswith('+'):
            return phone_str
            
        # If starts with 00, replace with +
        if phone_str.startswith('00'):
            return '+' + phone_str[2:]
            
        # Add country code
        return self.country_code + phone_str
    
    def clean_mode(self, phone_number: str) -> str:
        """Clean and standardize the phone number"""
        if not phone_number or pd.isna(phone_number):
            return phone_number
            
        # Convert to string if it's not already
        phone_str = str(phone_number).strip()
        
        # If empty after stripping, return as is
        if not phone_str:
            return phone_str
            
        # Remove all non-digit characters except '+'
        digits_only = re.sub(r'[^\d+]', '', phone_str)
        
        # If already has the correct country code, just standardize format
        if digits_only.startswith('+' + self.country_code_digits):
            return digits_only
            
        # If starts with +, it has a different country code - keep as is
        if digits_only.startswith('+'):
            return digits_only
            
        # If starts with 00, replace with +
        if digits_only.startswith('00'):
            digits_only = '+' + digits_only[2:]
            return digits_only
            
        # Remove leading 0
        if digits_only.startswith('0'):
            digits_only = digits_only[1:]
            
        # Add country code
        return self.country_code + digits_only
    
    def validate_number(self, phone_number: str, country_region: str = None) -> Dict:
        """Validate phone number using the phonenumbers library"""
        if not phone_number or pd.isna(phone_number):
            return {"valid": False, "reason": "Empty number", "type": None}
            
        # Convert to string if it's not already
        phone_str = str(phone_number).strip()
        
        # If empty after stripping, return as invalid
        if not phone_str:
            return {"valid": False, "reason": "Empty number", "type": None}
            
        try:
            # Parse the number
            parsed_number = phonenumbers.parse(phone_str, country_region)
            
            # Check if the number is valid
            is_valid = phonenumbers.is_valid_number(parsed_number)
            
            # Get the number type
            number_type = phonenumbers.number_type(parsed_number)
            type_str = self._get_number_type_str(number_type)
            
            if is_valid:
                return {"valid": True, "reason": "Valid number", "type": type_str}
            else:
                return {"valid": False, "reason": "Invalid number format", "type": type_str}
                
        except Exception as e:
            return {"valid": False, "reason": str(e), "type": None}
    
    def _get_number_type_str(self, number_type: int) -> str:
        """Convert phonenumbers number type to string representation"""
        type_map = {
            phonenumbers.PhoneNumberType.MOBILE: "Mobile",
            phonenumbers.PhoneNumberType.FIXED_LINE: "Fixed Line",
            phonenumbers.PhoneNumberType.FIXED_LINE_OR_MOBILE: "Fixed Line or Mobile",
            phonenumbers.PhoneNumberType.TOLL_FREE: "Toll Free",
            phonenumbers.PhoneNumberType.PREMIUM_RATE: "Premium Rate",
            phonenumbers.PhoneNumberType.SHARED_COST: "Shared Cost",
            phonenumbers.PhoneNumberType.VOIP: "VoIP",
            phonenumbers.PhoneNumberType.PERSONAL_NUMBER: "Personal Number",
            phonenumbers.PhoneNumberType.PAGER: "Pager",
            phonenumbers.PhoneNumberType.UAN: "UAN",
            phonenumbers.PhoneNumberType.VOICEMAIL: "Voicemail",
            phonenumbers.PhoneNumberType.UNKNOWN: "Unknown"
        }
        return type_map.get(number_type, "Unknown")


class FileProcessor:
    """Class for handling file operations"""
    
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        
    def log(self, message: str):
        """Log a message using the callback if available"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)
    
    def read_file(self, file_path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Read a CSV or Excel file and return a DataFrame"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.csv':
                # Try different encodings and delimiters
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(file_path, encoding='latin1')
                    except:
                        df = pd.read_csv(file_path, encoding='cp1252')
                except pd.errors.EmptyDataError:
                    return None, "The file is empty"
                
                self.log(f"Successfully read CSV file: {file_path}")
                return df, None
                
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
                self.log(f"Successfully read Excel file: {file_path}")
                return df, None
                
            else:
                return None, f"Unsupported file format: {file_ext}"
                
        except Exception as e:
            error_msg = f"Error reading file: {str(e)}"
            self.log(error_msg)
            return None, error_msg
    
    def save_file(self, df: pd.DataFrame, original_path: str, new_file: bool = False) -> Tuple[bool, Optional[str]]:
        """Save DataFrame back to file"""
        try:
            file_ext = os.path.splitext(original_path)[1].lower()
            
            # Determine output path
            if new_file:
                directory = os.path.dirname(original_path)
                filename = os.path.basename(original_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(directory, f"{name}_processed{ext}")
            else:
                output_path = original_path
            
            # Save based on file extension
            if file_ext == '.csv':
                df.to_csv(output_path, index=False)
                self.log(f"Successfully saved to CSV file: {output_path}")
                
            elif file_ext in ['.xlsx', '.xls']:
                df.to_excel(output_path, index=False)
                self.log(f"Successfully saved to Excel file: {output_path}")
                
            else:
                return False, f"Unsupported file format for saving: {file_ext}"
                
            return True, output_path
            
        except Exception as e:
            error_msg = f"Error saving file: {str(e)}"
            self.log(error_msg)
            return False, error_msg
    
    def process_phone_numbers(self, df: pd.DataFrame, column: str, processor: PhoneNumberProcessor, 
                             mode: str, validate: bool = False) -> Tuple[pd.DataFrame, Dict]:
        """Process phone numbers in the specified column"""
        if column not in df.columns:
            return df, {"error": f"Column '{column}' not found in the file"}
        
        # Create a copy of the DataFrame to avoid modifying the original
        processed_df = df.copy()
        
        # Statistics
        stats = {
            "total": len(df),
            "processed": 0,
            "valid": 0,
            "invalid": 0,
            "empty": 0,
            "types": {}
        }
        
        # Process each phone number
        for idx, value in enumerate(df[column]):
            # Skip empty values
            if pd.isna(value) or str(value).strip() == '':
                stats["empty"] += 1
                continue
                
            # Apply the selected processing mode
            if mode == "add":
                processed_number = processor.add_mode(value)
            else:  # clean mode
                processed_number = processor.clean_mode(value)
                
            # Update the DataFrame
            processed_df.at[idx, column] = processed_number
            stats["processed"] += 1
            
            # Validate if requested
            if validate:
                validation_result = processor.validate_number(processed_number)
                
                if validation_result["valid"]:
                    stats["valid"] += 1
                else:
                    stats["invalid"] += 1
                    
                # Track number types
                number_type = validation_result["type"] or "Unknown"
                if number_type in stats["types"]:
                    stats["types"][number_type] += 1
                else:
                    stats["types"][number_type] = 1
        
        return processed_df, stats


class CSVComparer:
    """Class for comparing and merging CSV files based on a key column"""
    
    def __init__(self, file_processor: FileProcessor, log_callback=None):
        self.file_processor = file_processor
        self.log_callback = log_callback
        self.file1_df = None
        self.file2_df = None
        self.file1_path = None
        self.file2_path = None
        self.key_column = None
        self.matches = None
        self.conflicts = None
        self.phone_processor = None
        
    def log(self, message: str):
        """Log a message using the callback if available"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)
    
    def load_files(self, file1_path: str, file2_path: str) -> Tuple[bool, Optional[str]]:
        """Load the two files to compare"""
        self.file1_path = file1_path
        self.file2_path = file2_path
        
        # Load first file
        df1, error1 = self.file_processor.read_file(file1_path)
        if error1:
            return False, f"Error loading first file: {error1}"
        
        # Load second file
        df2, error2 = self.file_processor.read_file(file2_path)
        if error2:
            return False, f"Error loading second file: {error2}"
        
        self.file1_df = df1
        self.file2_df = df2
        
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
        
        # Create a dictionary to store match information
        self.matches = {
            "total_matches": len(matching_keys),
            "only_in_file1": len(only_in_file1),
            "only_in_file2": len(only_in_file2),
            "matching_keys": list(matching_keys),
            "file1_only_keys": list(only_in_file1),
            "file2_only_keys": list(only_in_file2)
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
            row = self.file1_df_comp[self.file1_df_comp[self.key_column] == key].iloc[0]
            preview["only_in_file1"].append({
                "key": key,
                "data": self.file1_df.iloc[row.name].to_dict()
            })
        
        # Get preview of records only in file 2
        for key in self.matches["file2_only_keys"][:max_rows]:
            row = self.file2_df_comp[self.file2_df_comp[self.key_column] == key].iloc[0]
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
    
    def save_merged_file(self, merged_df: pd.DataFrame, output_path: str) -> Tuple[bool, Optional[str]]:
        """Save the merged DataFrame to a file"""
        try:
            file_ext = os.path.splitext(output_path)[1].lower()
            
            if file_ext == '.csv':
                merged_df.to_csv(output_path, index=False)
                self.log(f"Successfully saved merged data to CSV file: {output_path}")
            elif file_ext in ['.xlsx', '.xls']:
                merged_df.to_excel(output_path, index=False)
                self.log(f"Successfully saved merged data to Excel file: {output_path}")
            else:
                return False, f"Unsupported file format for saving: {file_ext}"
                
            return True, output_path
            
        except Exception as e:
            error_msg = f"Error saving merged file: {str(e)}"
            self.log(error_msg)
            return False, error_msg


class BatchProcessor:
    """Class for processing multiple files in batch"""
    
    def __init__(self, file_processor: FileProcessor, log_callback=None):
        self.file_processor = file_processor
        self.log_callback = log_callback
        
    def log(self, message: str):
        """Log a message using the callback if available"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)
    
    def process_batch(self, file_paths: List[str], column: str, processor: PhoneNumberProcessor, 
                     mode: str, validate: bool = False, new_file: bool = False) -> Dict:
        """Process multiple files in batch"""
        batch_stats = {
            "total_files": len(file_paths),
            "successful_files": 0,
            "failed_files": 0,
            "total_records": 0,
            "processed_records": 0,
            "valid_records": 0,
            "invalid_records": 0,
            "empty_records": 0,
            "file_results": {}
        }
        
        for file_path in file_paths:
            self.log(f"Processing file: {file_path}")
            
            # Read the file
            df, error = self.file_processor.read_file(file_path)
            
            if error:
                self.log(f"Error reading file {file_path}: {error}")
                batch_stats["failed_files"] += 1
                batch_stats["file_results"][file_path] = {"status": "failed", "error": error}
                continue
                
            # Process the file
            processed_df, stats = self.file_processor.process_phone_numbers(
                df, column, processor, mode, validate
            )
            
            # Save the processed file
            success, result = self.file_processor.save_file(processed_df, file_path, new_file)
            
            if success:
                self.log(f"Successfully processed and saved file: {result}")
                batch_stats["successful_files"] += 1
                batch_stats["file_results"][file_path] = {
                    "status": "success", 
                    "output_path": result,
                    "stats": stats
                }
                
                # Update batch statistics
                batch_stats["total_records"] += stats["total"]
                batch_stats["processed_records"] += stats["processed"]
                batch_stats["valid_records"] += stats.get("valid", 0)
                batch_stats["invalid_records"] += stats.get("invalid", 0)
                batch_stats["empty_records"] += stats["empty"]
            else:
                self.log(f"Error saving file {file_path}: {result}")
                batch_stats["failed_files"] += 1
                batch_stats["file_results"][file_path] = {"status": "failed", "error": result}
        
        return batch_stats


class PhoneFormatterApp(ctk.CTk):
    """Main application class"""
    
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("Phone Number Formatter")
        self.geometry("1000x700")
        self.minsize(800, 600)
        
        # Initialize variables
        self.selected_files = []
        self.current_df = None
        self.available_columns = []
        self.processing_thread = None
        self.country_code_var = ctk.StringVar(value="+49")  # Default to Germany
        self.custom_code_var = ctk.StringVar(value="")
        self.mode_var = ctk.StringVar(value="clean")  # Default to clean mode
        self.validate_var = ctk.BooleanVar(value=True)  # Default to validate
        self.new_file_var = ctk.BooleanVar(value=True)  # Default to create new file
        self.selected_column_var = ctk.StringVar(value="")
        
        # Initialize variables for CSV comparison
        self.file1_path = None
        self.file2_path = None
        self.file1_df = None
        self.file2_df = None
        self.file1_columns = []
        self.file2_columns = []
        self.key_column1_var = ctk.StringVar(value="")
        self.key_column2_var = ctk.StringVar(value="")
        self.normalize_phone_var = ctk.BooleanVar(value=True)
        self.conflict_resolution_var = ctk.StringVar(value="file1")
        self.include_only_in_file1_var = ctk.BooleanVar(value=True)
        self.include_only_in_file2_var = ctk.BooleanVar(value=True)
        self.comparison_thread = None
        self.comparison_results = None
        self.selected_columns = []
        self.per_conflict_resolutions = {}
        
        # Create UI components
        self.create_ui()
        
        # Initialize file processor
        self.file_processor = FileProcessor(log_callback=self.log)
        self.batch_processor = BatchProcessor(self.file_processor, log_callback=self.log)
        self.csv_comparer = CSVComparer(self.file_processor, log_callback=self.log)
    
    def create_ui(self):
        """Create the user interface"""
        # Create main frame with padding
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create top frame for file selection
        self.create_file_selection_frame()
        
        # Create middle frame for options
        self.create_options_frame()
        
        # Create bottom frame for results and logging
        self.create_results_frame()
    
    def create_file_selection_frame(self):
        """Create the file selection frame"""
        file_frame = ctk.CTkFrame(self.main_frame)
        file_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Title
        title_label = ctk.CTkLabel(file_frame, text="File Selection", font=ctk.CTkFont(size=16, weight="bold"))
        title_label.pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        # File selection buttons
        button_frame = ctk.CTkFrame(file_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        select_file_btn = ctk.CTkButton(button_frame, text="Select File", command=self.select_file)
        select_file_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        select_files_btn = ctk.CTkButton(button_frame, text="Select Multiple Files", command=self.select_multiple_files)
        select_files_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = ctk.CTkButton(button_frame, text="Clear Selection", command=self.clear_selection)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Selected files display
        self.files_display = ctk.CTkTextbox(file_frame, height=80, wrap=tk.WORD)
        self.files_display.pack(fill=tk.X, padx=10, pady=5)
        self.files_display.configure(state=tk.DISABLED)
        
        # Column selection
        column_frame = ctk.CTkFrame(file_frame)
        column_frame.pack(fill=tk.X, padx=10, pady=5)
        
        column_label = ctk.CTkLabel(column_frame, text="Phone Number Column:")
        column_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.column_dropdown = ctk.CTkOptionMenu(column_frame, variable=self.selected_column_var, 
                                                values=["No columns available"], 
                                                command=self.on_column_selected)
        self.column_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.column_dropdown.configure(state=tk.DISABLED)
    
    def create_options_frame(self):
        """Create the options frame"""
        options_frame = ctk.CTkFrame(self.main_frame)
        options_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Title
        title_label = ctk.CTkLabel(options_frame, text="Processing Options", font=ctk.CTkFont(size=16, weight="bold"))
        title_label.pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        # Country code selection
        country_frame = ctk.CTkFrame(options_frame)
        country_frame.pack(fill=tk.X, padx=10, pady=5)
        
        country_label = ctk.CTkLabel(country_frame, text="Country Code:")
        country_label.pack(side=tk.LEFT, padx=(0, 5))
        
        # Create a list of country names with codes
        country_options = [f"{country} ({code})" for country, code in COUNTRY_CODES.items()]
        
        self.country_dropdown = ctk.CTkOptionMenu(country_frame, values=country_options, 
                                                 command=self.on_country_selected)
        self.country_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Custom country code entry (initially hidden)
        self.custom_code_entry = ctk.CTkEntry(country_frame, textvariable=self.custom_code_var, 
                                             placeholder_text="Enter custom code (e.g. +49)")
        
        # Processing mode
        mode_frame = ctk.CTkFrame(options_frame)
        mode_frame.pack(fill=tk.X, padx=10, pady=5)
        
        mode_label = ctk.CTkLabel(mode_frame, text="Processing Mode:")
        mode_label.pack(side=tk.LEFT, padx=(0, 5))
        
        add_radio = ctk.CTkRadioButton(mode_frame, text="Add Mode", variable=self.mode_var, value="add")
        add_radio.pack(side=tk.LEFT, padx=(0, 10))
        
        clean_radio = ctk.CTkRadioButton(mode_frame, text="Clean Mode", variable=self.mode_var, value="clean")
        clean_radio.pack(side=tk.LEFT, padx=(0, 10))
        
        # Additional options
        options_subframe = ctk.CTkFrame(options_frame)
        options_subframe.pack(fill=tk.X, padx=10, pady=5)
        
        validate_check = ctk.CTkCheckBox(options_subframe, text="Validate Phone Numbers", 
                                        variable=self.validate_var)
        validate_check.pack(side=tk.LEFT, padx=(0, 10))
        
        new_file_check = ctk.CTkCheckBox(options_subframe, text="Create New File", 
                                        variable=self.new_file_var)
        new_file_check.pack(side=tk.LEFT, padx=(0, 10))
        
        # Process button
        process_frame = ctk.CTkFrame(options_frame)
        process_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        self.process_btn = ctk.CTkButton(process_frame, text="Process Files", 
                                        command=self.process_files,
                                        fg_color="#28a745", hover_color="#218838")
        self.process_btn.pack(fill=tk.X, pady=5)
    
    def create_results_frame(self):
        """Create the results and logging frame"""
        results_frame = ctk.CTkFrame(self.main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabview for results and logs
        self.tabview = ctk.CTkTabview(results_frame)
        self.tabview.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add tabs
        self.tabview.add("Log")
        self.tabview.add("Statistics")
        self.tabview.add("Preview")
        self.tabview.add("CSV Comparison")
        
        # Log tab
        self.log_text = ctk.CTkTextbox(self.tabview.tab("Log"), wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Statistics tab
        self.stats_text = ctk.CTkTextbox(self.tabview.tab("Statistics"), wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Preview tab
        preview_frame = ctk.CTkFrame(self.tabview.tab("Preview"))
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.preview_text = ctk.CTkTextbox(preview_frame, wrap=tk.WORD)
        self.preview_text.pack(fill=tk.BOTH, expand=True)
        
        # CSV Comparison tab
        self.create_csv_comparison_tab()
    
    def create_csv_comparison_tab(self):
        """Create the CSV comparison tab"""
        comparison_frame = ctk.CTkFrame(self.tabview.tab("CSV Comparison"))
        comparison_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a scrollable frame for the comparison UI
        comparison_scroll = ctk.CTkScrollableFrame(comparison_frame)
        comparison_scroll.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # File selection section
        file_section = ctk.CTkFrame(comparison_scroll)
        file_section.pack(fill=tk.X, padx=5, pady=5)
        
        # Title
        title_label = ctk.CTkLabel(file_section, text="CSV Files to Compare", 
                                  font=ctk.CTkFont(size=16, weight="bold"))
        title_label.pack(anchor=tk.W, padx=5, pady=5)
        
        # File 1 selection
        file1_frame = ctk.CTkFrame(file_section)
        file1_frame.pack(fill=tk.X, padx=5, pady=5)
        
        file1_label = ctk.CTkLabel(file1_frame, text="File 1:")
        file1_label.pack(side=tk.LEFT, padx=5)
        
        self.file1_path_var = ctk.StringVar(value="No file selected")
        file1_path_label = ctk.CTkLabel(file1_frame, textvariable=self.file1_path_var, 
                                       width=400, anchor="w")
        file1_path_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        file1_btn = ctk.CTkButton(file1_frame, text="Select File", 
                                 command=self.select_comparison_file1)
        file1_btn.pack(side=tk.LEFT, padx=5)
        
        # File 2 selection
        file2_frame = ctk.CTkFrame(file_section)
        file2_frame.pack(fill=tk.X, padx=5, pady=5)
        
        file2_label = ctk.CTkLabel(file2_frame, text="File 2:")
        file2_label.pack(side=tk.LEFT, padx=5)
        
        self.file2_path_var = ctk.StringVar(value="No file selected")
        file2_path_label = ctk.CTkLabel(file2_frame, textvariable=self.file2_path_var, 
                                       width=400, anchor="w")
        file2_path_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        file2_btn = ctk.CTkButton(file2_frame, text="Select File", 
                                 command=self.select_comparison_file2)
        file2_btn.pack(side=tk.LEFT, padx=5)
        
        # Key column selection
        key_section = ctk.CTkFrame(comparison_scroll)
        key_section.pack(fill=tk.X, padx=5, pady=5)
        
        key_title = ctk.CTkLabel(key_section, text="Key Column Selection", 
                                font=ctk.CTkFont(size=14, weight="bold"))
        key_title.pack(anchor=tk.W, padx=5, pady=5)
        
        # File 1 key column
        key1_frame = ctk.CTkFrame(key_section)
        key1_frame.pack(fill=tk.X, padx=5, pady=5)
        
        key1_label = ctk.CTkLabel(key1_frame, text="File 1 Key Column:")
        key1_label.pack(side=tk.LEFT, padx=5)
        
        self.key1_dropdown = ctk.CTkOptionMenu(key1_frame, variable=self.key_column1_var, 
                                              values=["No columns available"])
        self.key1_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.key1_dropdown.configure(state=tk.DISABLED)
        
        # File 2 key column
        key2_frame = ctk.CTkFrame(key_section)
        key2_frame.pack(fill=tk.X, padx=5, pady=5)
        
        key2_label = ctk.CTkLabel(key2_frame, text="File 2 Key Column:")
        key2_label.pack(side=tk.LEFT, padx=5)
        
        self.key2_dropdown = ctk.CTkOptionMenu(key2_frame, variable=self.key_column2_var, 
                                              values=["No columns available"])
        self.key2_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.key2_dropdown.configure(state=tk.DISABLED)
        
        # Normalize phone numbers option
        normalize_frame = ctk.CTkFrame(key_section)
        normalize_frame.pack(fill=tk.X, padx=5, pady=5)
        
        normalize_check = ctk.CTkCheckBox(normalize_frame, text="Normalize Phone Numbers for Comparison", 
                                         variable=self.normalize_phone_var)
        normalize_check.pack(side=tk.LEFT, padx=5)
        
        # Comparison options section
        options_section = ctk.CTkFrame(comparison_scroll)
        options_section.pack(fill=tk.X, padx=5, pady=5)
        
        options_title = ctk.CTkLabel(options_section, text="Merge Options", 
                                    font=ctk.CTkFont(size=14, weight="bold"))
        options_title.pack(anchor=tk.W, padx=5, pady=5)
        
        # Conflict resolution
        conflict_frame = ctk.CTkFrame(options_section)
        conflict_frame.pack(fill=tk.X, padx=5, pady=5)
        
        conflict_label = ctk.CTkLabel(conflict_frame, text="Default Conflict Resolution:")
        conflict_label.pack(side=tk.LEFT, padx=5)
        
        file1_radio = ctk.CTkRadioButton(conflict_frame, text="File 1 Takes Precedence", 
                                        variable=self.conflict_resolution_var, value="file1")
        file1_radio.pack(side=tk.LEFT, padx=5)
        
        file2_radio = ctk.CTkRadioButton(conflict_frame, text="File 2 Takes Precedence", 
                                        variable=self.conflict_resolution_var, value="file2")
        file2_radio.pack(side=tk.LEFT, padx=5)
        
        # Include options
        include_frame = ctk.CTkFrame(options_section)
        include_frame.pack(fill=tk.X, padx=5, pady=5)
        
        include_label = ctk.CTkLabel(include_frame, text="Include Records:")
        include_label.pack(side=tk.LEFT, padx=5)
        
        include1_check = ctk.CTkCheckBox(include_frame, text="Only in File 1", 
                                        variable=self.include_only_in_file1_var)
        include1_check.pack(side=tk.LEFT, padx=5)
        
        include2_check = ctk.CTkCheckBox(include_frame, text="Only in File 2", 
                                        variable=self.include_only_in_file2_var)
        include2_check.pack(side=tk.LEFT, padx=5)
        
        # Column selection section (will be populated after comparison)
        self.column_selection_frame = ctk.CTkFrame(comparison_scroll)
        
        # Buttons section
        button_section = ctk.CTkFrame(comparison_scroll)
        button_section.pack(fill=tk.X, padx=5, pady=5)
        
        # Compare button
        self.compare_btn = ctk.CTkButton(button_section, text="Compare Files", 
                                        command=self.compare_files,
                                        fg_color="#007bff", hover_color="#0069d9")
        self.compare_btn.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        # Merge button (initially disabled)
        self.merge_btn = ctk.CTkButton(button_section, text="Merge Files", 
                                      command=self.merge_files,
                                      fg_color="#28a745", hover_color="#218838",
                                      state=tk.DISABLED)
        self.merge_btn.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        # Preview section
        preview_section = ctk.CTkFrame(comparison_frame)
        preview_section.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a notebook for the preview tabs
        self.preview_notebook = ttk.Notebook(preview_section)
        self.preview_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create preview tabs
        self.matches_frame = ttk.Frame(self.preview_notebook)
        self.conflicts_frame = ttk.Frame(self.preview_notebook)
        self.only_in_file1_frame = ttk.Frame(self.preview_notebook)
        self.only_in_file2_frame = ttk.Frame(self.preview_notebook)
        
        self.preview_notebook.add(self.matches_frame, text="Matches")
        self.preview_notebook.add(self.conflicts_frame, text="Conflicts")
        self.preview_notebook.add(self.only_in_file1_frame, text="Only in File 1")
        self.preview_notebook.add(self.only_in_file2_frame, text="Only in File 2")
        
        # Create treeviews for each tab
        self.matches_tree = ttk.Treeview(self.matches_frame)
        self.conflicts_tree = ttk.Treeview(self.conflicts_frame)
        self.only_in_file1_tree = ttk.Treeview(self.only_in_file1_frame)
        self.only_in_file2_tree = ttk.Treeview(self.only_in_file2_frame)
        
        # Add scrollbars to each treeview
        for tree in [self.matches_tree, self.conflicts_tree, self.only_in_file1_tree, self.only_in_file2_tree]:
            vsb = ttk.Scrollbar(tree.master, orient="vertical", command=tree.yview)
            hsb = ttk.Scrollbar(tree.master, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
            
            vsb.pack(side=tk.RIGHT, fill=tk.Y)
            hsb.pack(side=tk.BOTTOM, fill=tk.X)
            tree.pack(fill=tk.BOTH, expand=True)
    
    def select_file(self):
        """Select a single file"""
        file_path = filedialog.askopenfilename(
            title="Select File",
            filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx;*.xls"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.selected_files = [file_path]
            self.update_files_display()
            self.load_file_preview(file_path)
    
    def select_multiple_files(self):
        """Select multiple files"""
        file_paths = filedialog.askopenfilenames(
            title="Select Files",
            filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx;*.xls"), ("All Files", "*.*")]
        )
        
        if file_paths:
            self.selected_files = list(file_paths)
            self.update_files_display()
            
            # Load preview of the first file
            if self.selected_files:
                self.load_file_preview(self.selected_files[0])
    
    def clear_selection(self):
        """Clear the file selection"""
        self.selected_files = []
        self.current_df = None
        self.available_columns = []
        self.update_files_display()
        self.update_column_dropdown()
        
        # Clear preview
        self.preview_text.configure(state=tk.NORMAL)
        self.preview_text.delete(1.0, tk.END)
        self.preview_text.configure(state=tk.DISABLED)
    
    def update_files_display(self):
        """Update the files display textbox"""
        self.files_display.configure(state=tk.NORMAL)
        self.files_display.delete(1.0, tk.END)
        
        if self.selected_files:
            for file_path in self.selected_files:
                self.files_display.insert(tk.END, f"{file_path}\n")
        else:
            self.files_display.insert(tk.END, "No files selected")
            
        self.files_display.configure(state=tk.DISABLED)
    
    def load_file_preview(self, file_path: str):
        """Load a file preview and update the column dropdown"""
        self.log(f"Loading preview for: {file_path}")
        
        # Read the file
        df, error = self.file_processor.read_file(file_path)
        
        if error:
            self.log(f"Error loading preview: {error}")
            messagebox.showerror("Error", f"Failed to load file: {error}")
            return
            
        self.current_df = df
        self.available_columns = list(df.columns)
        
        # Update column dropdown
        self.update_column_dropdown()
        
        # Show preview
        self.preview_text.configure(state=tk.NORMAL)
        self.preview_text.delete(1.0, tk.END)
        
        # Display first 10 rows
        preview_rows = min(10, len(df))
        self.preview_text.insert(tk.END, f"Preview of first {preview_rows} rows:\n\n")
        self.preview_text.insert(tk.END, df.head(preview_rows).to_string())
        
        self.preview_text.configure(state=tk.DISABLED)
    
    def update_column_dropdown(self):
        """Update the column dropdown with available columns"""
        if self.available_columns:
            self.column_dropdown.configure(state=tk.NORMAL)
            self.column_dropdown.configure(values=self.available_columns)
            
            # Select first column by default
            self.selected_column_var.set(self.available_columns[0])
        else:
            self.column_dropdown.configure(values=["No columns available"])
            self.column_dropdown.configure(state=tk.DISABLED)
            self.selected_column_var.set("")
    
    def on_column_selected(self, column: str):
        """Handle column selection"""
        self.log(f"Selected column: {column}")
    
    def on_country_selected(self, selection: str):
        """Handle country selection"""
        # Extract country name from selection (format: "Country (Code)")
        country = selection.split(" (")[0]
        
        if country == "Other":
            # Show custom code entry
            self.custom_code_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.country_code_var.set("")
        else:
            # Hide custom code entry if visible
            if self.custom_code_entry.winfo_ismapped():
                self.custom_code_entry.pack_forget()
                
            # Set the country code
            self.country_code_var.set(COUNTRY_CODES[country])
            
        self.log(f"Selected country code: {self.country_code_var.get()}")
    
    def get_country_code(self) -> str:
        """Get the selected country code"""
        if self.country_code_var.get():
            return self.country_code_var.get()
        elif self.custom_code_var.get():
            # Ensure custom code starts with +
            custom_code = self.custom_code_var.get().strip()
            if not custom_code.startswith('+'):
                custom_code = '+' + custom_code
            return custom_code
        else:
            return "+49"  # Default to Germany
    
    def process_files(self):
        """Process the selected files"""
        if not self.selected_files:
            messagebox.showwarning("Warning", "No files selected")
            return
            
        if not self.selected_column_var.get():
            messagebox.showwarning("Warning", "No column selected")
            return
            
        # Get processing options
        country_code = self.get_country_code()
        mode = self.mode_var.get()
        validate = self.validate_var.get()
        new_file = self.new_file_var.get()
        column = self.selected_column_var.get()
        
        # Disable process button during processing
        self.process_btn.configure(state=tk.DISABLED, text="Processing...")
        
        # Clear statistics
        self.stats_text.configure(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.configure(state=tk.DISABLED)
        
        # Log processing start
        self.log(f"Starting processing with options: Country Code={country_code}, Mode={mode}, "
                f"Validate={validate}, New File={new_file}, Column={column}")
        
        # Create processor
        processor = PhoneNumberProcessor(country_code)
        
        # Start processing in a separate thread
        self.processing_thread = threading.Thread(
            target=self._process_files_thread,
            args=(processor, mode, validate, new_file, column)
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _process_files_thread(self, processor, mode, validate, new_file, column):
        """Thread function for processing files"""
        try:
            # Process files
            if len(self.selected_files) == 1:
                # Single file processing
                file_path = self.selected_files[0]
                
                # Read the file
                df, error = self.file_processor.read_file(file_path)
                
                if error:
                    self.log(f"Error reading file: {error}")
                    messagebox.showerror("Error", f"Failed to process file: {error}")
                    return
                    
                # Process the file
                processed_df, stats = self.file_processor.process_phone_numbers(
                    df, column, processor, mode, validate
                )
                
                # Save the processed file
                success, result = self.file_processor.save_file(processed_df, file_path, new_file)
                
                if success:
                    self.log(f"Successfully processed and saved file: {result}")
                    self.display_statistics(stats)
                else:
                    self.log(f"Error saving file: {result}")
                    messagebox.showerror("Error", f"Failed to save file: {result}")
            else:
                # Batch processing
                batch_stats = self.batch_processor.process_batch(
                    self.selected_files, column, processor, mode, validate, new_file
                )
                
                self.log(f"Batch processing completed: {batch_stats['successful_files']} successful, "
                        f"{batch_stats['failed_files']} failed")
                        
                self.display_batch_statistics(batch_stats)
                
        except Exception as e:
            self.log(f"Error during processing: {str(e)}")
            messagebox.showerror("Error", f"An error occurred during processing: {str(e)}")
            
        finally:
            # Re-enable process button
            self.process_btn.configure(state=tk.NORMAL, text="Process Files")
    
    def log(self, message: str):
        """Add a message to the log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        # Update log text
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)  # Scroll to the end
        self.log_text.configure(state=tk.DISABLED)
        
        # Print to console as well
        print(log_message)
    
    def display_statistics(self, stats: Dict):
        """Display processing statistics"""
        self.stats_text.configure(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        
        self.stats_text.insert(tk.END, "Processing Statistics\n")
        self.stats_text.insert(tk.END, "====================\n\n")
        
        self.stats_text.insert(tk.END, f"Total records: {stats['total']}\n")
        self.stats_text.insert(tk.END, f"Processed records: {stats['processed']}\n")
        self.stats_text.insert(tk.END, f"Empty records: {stats['empty']}\n")
        
        if 'valid' in stats:
            self.stats_text.insert(tk.END, f"Valid numbers: {stats['valid']}\n")
            self.stats_text.insert(tk.END, f"Invalid numbers: {stats['invalid']}\n\n")
            
            # Number types
            if 'types' in stats and stats['types']:
                self.stats_text.insert(tk.END, "Number Types:\n")
                for type_name, count in stats['types'].items():
                    self.stats_text.insert(tk.END, f"  - {type_name}: {count}\n")
        
        self.stats_text.configure(state=tk.DISABLED)
    
    def display_batch_statistics(self, batch_stats: Dict):
        """Display batch processing statistics"""
        self.stats_text.configure(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        
        self.stats_text.insert(tk.END, "Batch Processing Statistics\n")
        self.stats_text.insert(tk.END, "==========================\n\n")
        
        self.stats_text.insert(tk.END, f"Total files: {batch_stats['total_files']}\n")
        self.stats_text.insert(tk.END, f"Successfully processed files: {batch_stats['successful_files']}\n")
        self.stats_text.insert(tk.END, f"Failed files: {batch_stats['failed_files']}\n\n")
        
        self.stats_text.insert(tk.END, f"Total records: {batch_stats['total_records']}\n")
        self.stats_text.insert(tk.END, f"Processed records: {batch_stats['processed_records']}\n")
        self.stats_text.insert(tk.END, f"Empty records: {batch_stats['empty_records']}\n")
        
        if 'valid_records' in batch_stats:
            self.stats_text.insert(tk.END, f"Valid numbers: {batch_stats['valid_records']}\n")
            self.stats_text.insert(tk.END, f"Invalid numbers: {batch_stats['invalid_records']}\n\n")
        
        # File details
        self.stats_text.insert(tk.END, "File Details:\n")
        for file_path, result in batch_stats['file_results'].items():
            status = result['status']
            if status == 'success':
                self.stats_text.insert(tk.END, f"  - {os.path.basename(file_path)}: Success\n")
                self.stats_text.insert(tk.END, f"    Output: {result['output_path']}\n")
                self.stats_text.insert(tk.END, f"    Records: {result['stats']['processed']}/{result['stats']['total']}\n")
            else:
                self.stats_text.insert(tk.END, f"  - {os.path.basename(file_path)}: Failed - {result['error']}\n")
        
        self.stats_text.configure(state=tk.DISABLED)
    
    # CSV Comparison Methods
    
    def select_comparison_file1(self):
        """Select the first file for comparison"""
        file_path = filedialog.askopenfilename(
            title="Select First File for Comparison",
            filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx;*.xls"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.file1_path = file_path
            self.file1_path_var.set(os.path.basename(file_path))
            self.log(f"Selected file 1 for comparison: {file_path}")
            
            # Load the file and update the column dropdown
            df, error = self.file_processor.read_file(file_path)
            if error:
                self.log(f"Error loading file 1: {error}")
                messagebox.showerror("Error", f"Failed to load file: {error}")
                return
                
            self.file1_df = df
            self.file1_columns = list(df.columns)
            
            # Update the column dropdown
            self.key1_dropdown.configure(state=tk.NORMAL)
            self.key1_dropdown.configure(values=self.file1_columns)
            
            # Select a phone column by default if available
            phone_columns = [col for col in self.file1_columns if 'phone' in col.lower() or 'tel' in col.lower()]
            if phone_columns:
                self.key_column1_var.set(phone_columns[0])
            else:
                self.key_column1_var.set(self.file1_columns[0])
    
    def select_comparison_file2(self):
        """Select the second file for comparison"""
        file_path = filedialog.askopenfilename(
            title="Select Second File for Comparison",
            filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx;*.xls"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.file2_path = file_path
            self.file2_path_var.set(os.path.basename(file_path))
            self.log(f"Selected file 2 for comparison: {file_path}")
            
            # Load the file and update the column dropdown
            df, error = self.file_processor.read_file(file_path)
            if error:
                self.log(f"Error loading file 2: {error}")
                messagebox.showerror("Error", f"Failed to load file: {error}")
                return
                
            self.file2_df = df
            self.file2_columns = list(df.columns)
            
            # Update the column dropdown
            self.key2_dropdown.configure(state=tk.NORMAL)
            self.key2_dropdown.configure(values=self.file2_columns)
            
            # Select a phone column by default if available
            phone_columns = [col for col in self.file2_columns if 'phone' in col.lower() or 'tel' in col.lower()]
            if phone_columns:
                self.key_column2_var.set(phone_columns[0])
            else:
                self.key_column2_var.set(self.file2_columns[0])
    
    def compare_files(self):
        """Compare the selected files"""
        if not self.file1_path or not self.file2_path:
            messagebox.showwarning("Warning", "Please select two files to compare")
            return
            
        if not self.key_column1_var.get() or not self.key_column2_var.get():
            messagebox.showwarning("Warning", "Please select key columns for comparison")
            return
            
        # Disable compare button during comparison
        self.compare_btn.configure(state=tk.DISABLED, text="Comparing...")
        
        # Clear previous comparison results
        self.clear_comparison_results()
        
        # Get comparison options
        key_column1 = self.key_column1_var.get()
        key_column2 = self.key_column2_var.get()
        normalize_phone = self.normalize_phone_var.get()
        country_code = self.get_country_code()
        
        # Log comparison start
        self.log(f"Starting comparison with options: Key Column 1={key_column1}, Key Column 2={key_column2}, "
                f"Normalize Phone={normalize_phone}, Country Code={country_code}")
        
        # Start comparison in a separate thread
        self.comparison_thread = threading.Thread(
            target=self._compare_files_thread,
            args=(key_column1, key_column2, normalize_phone, country_code)
        )
        self.comparison_thread.daemon = True
        self.comparison_thread.start()
    
    def _compare_files_thread(self, key_column1, key_column2, normalize_phone, country_code):
        """Thread function for comparing files"""
        try:
            # Load the files
            success, error = self.csv_comparer.load_files(self.file1_path, self.file2_path)
            if not success:
                self.log(f"Error loading files for comparison: {error}")
                messagebox.showerror("Error", f"Failed to load files: {error}")
                return
                
            # Prepare for comparison
            # For simplicity, we'll use the key column from file 1 as the common key
            success, error = self.csv_comparer.prepare_for_comparison(
                key_column1, normalize_phone, country_code
            )
            if not success:
                self.log(f"Error preparing for comparison: {error}")
                messagebox.showerror("Error", f"Failed to prepare for comparison: {error}")
                return
                
            # Compare the files
            self.comparison_results = self.csv_comparer.compare_files()
            
            # Get preview data
            preview_data = self.csv_comparer.get_preview_data()
            
            # Update the UI with the comparison results
            self.after(100, lambda: self.update_comparison_ui(self.comparison_results, preview_data))
            
            self.log(f"Comparison completed: {self.comparison_results['matches']['total_matches']} matches, "
                    f"{len(self.comparison_results['conflicts'])} conflicts, "
                    f"{self.comparison_results['matches']['only_in_file1']} only in file 1, "
                    f"{self.comparison_results['matches']['only_in_file2']} only in file 2")
                    
        except Exception as e:
            self.log(f"Error during comparison: {str(e)}")
            messagebox.showerror("Error", f"An error occurred during comparison: {str(e)}")
            
        finally:
            # Re-enable compare button
            self.compare_btn.configure(state=tk.NORMAL, text="Compare Files")
    
    def clear_comparison_results(self):
        """Clear previous comparison results"""
        # Clear treeviews
        for tree in [self.matches_tree, self.conflicts_tree, self.only_in_file1_tree, self.only_in_file2_tree]:
            for item in tree.get_children():
                tree.delete(item)
            
            # Clear columns
            for col in tree["columns"]:
                tree["displaycolumns"] = ()
                
        # Hide column selection frame if visible
        if self.column_selection_frame.winfo_ismapped():
            self.column_selection_frame.pack_forget()
            
        # Disable merge button
        self.merge_btn.configure(state=tk.DISABLED)
        
        # Reset selected columns and per-conflict resolutions
        self.selected_columns = []
        self.per_conflict_resolutions = {}
    
    def update_comparison_ui(self, comparison_results, preview_data):
        """Update the UI with the comparison results"""
        # Update column selection frame
        self.update_column_selection_frame(comparison_results)
        
        # Update treeviews with preview data
        self.update_treeviews(comparison_results, preview_data)
        
        # Enable merge button
        self.merge_btn.configure(state=tk.NORMAL)
    
    def update_column_selection_frame(self, comparison_results):
        """Update the column selection frame with available columns"""
        # Clear previous content
        for widget in self.column_selection_frame.winfo_children():
            widget.destroy()
            
        # Add column selection UI
        title_label = ctk.CTkLabel(self.column_selection_frame, text="Column Selection", 
                                  font=ctk.CTkFont(size=14, weight="bold"))
        title_label.pack(anchor=tk.W, padx=5, pady=5)
        
        # Get all columns
        all_columns = list(set(comparison_results["file1_columns"] + comparison_results["file2_columns"]))
        all_columns.sort()
        
        # Create a frame for the column checkboxes
        columns_frame = ctk.CTkFrame(self.column_selection_frame)
        columns_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add "Select All" checkbox
        self.select_all_var = ctk.BooleanVar(value=True)
        select_all_check = ctk.CTkCheckBox(columns_frame, text="Select All Columns", 
                                          variable=self.select_all_var,
                                          command=self.toggle_all_columns)
        select_all_check.pack(anchor=tk.W, padx=5, pady=5)
        
        # Add column checkboxes
        self.column_vars = {}
        
        # Create a scrollable frame for the column checkboxes
        column_scroll = ctk.CTkScrollableFrame(columns_frame, height=150)
        column_scroll.pack(fill=tk.X, padx=5, pady=5)
        
        # Create a grid of checkboxes (3 columns)
        for i, column in enumerate(all_columns):
            row = i // 3
            col = i % 3
            
            var = ctk.BooleanVar(value=True)
            self.column_vars[column] = var
            
            # Determine which file(s) this column is in
            in_file1 = column in comparison_results["file1_columns"]
            in_file2 = column in comparison_results["file2_columns"]
            
            # Create label text with file indicators
            if in_file1 and in_file2:
                label_text = f"{column} (Both)"
            elif in_file1:
                label_text = f"{column} (File 1)"
            else:
                label_text = f"{column} (File 2)"
            
            check = ctk.CTkCheckBox(column_scroll, text=label_text, variable=var)
            check.grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
        
        # Update selected columns
        self.update_selected_columns()
        
        # Show the frame
        self.column_selection_frame.pack(fill=tk.X, padx=5, pady=5)
    
    def toggle_all_columns(self):
        """Toggle all column checkboxes"""
        value = self.select_all_var.get()
        for var in self.column_vars.values():
            var.set(value)
        
        # Update selected columns
        self.update_selected_columns()
    
    def update_selected_columns(self):
        """Update the list of selected columns"""
        self.selected_columns = [col for col, var in self.column_vars.items() if var.get()]
    
    def update_treeviews(self, comparison_results, preview_data):
        """Update the treeviews with preview data"""
        # Update matches treeview
        self.update_matches_treeview(preview_data["matches"])
        
        # Update conflicts treeview
        self.update_conflicts_treeview(preview_data["conflicts"])
        
        # Update only in file 1 treeview
        self.update_only_in_file1_treeview(preview_data["only_in_file1"])
        
        # Update only in file 2 treeview
        self.update_only_in_file2_treeview(preview_data["only_in_file2"])
    
    def update_matches_treeview(self, matches):
        """Update the matches treeview"""
        if not matches:
            return
            
        # Clear previous content
        for item in self.matches_tree.get_children():
            self.matches_tree.delete(item)
            
        # Get all columns from the first match
        if matches:
            file1_columns = list(matches[0]["file1_data"].keys())
            file2_columns = list(matches[0]["file2_data"].keys())
            
            # Configure columns
            self.matches_tree["columns"] = ("key",) + tuple(file1_columns) + tuple(f"file2_{col}" for col in file2_columns)
            self.matches_tree.column("#0", width=0, stretch=tk.NO)
            self.matches_tree.column("key", width=150, anchor=tk.W)
            
            for col in file1_columns:
                self.matches_tree.column(col, width=100, anchor=tk.W)
                
            for col in file2_columns:
                col_name = f"file2_{col}"
                self.matches_tree.column(col_name, width=100, anchor=tk.W)
            
            # Configure headings
            self.matches_tree.heading("#0", text="")
            self.matches_tree.heading("key", text="Key")
            
            for col in file1_columns:
                self.matches_tree.heading(col, text=f"File 1: {col}")
                
            for col in file2_columns:
                col_name = f"file2_{col}"
                self.matches_tree.heading(col_name, text=f"File 2: {col}")
            
            # Add data
            for i, match in enumerate(matches):
                key = match["key"]
                file1_data = match["file1_data"]
                file2_data = match["file2_data"]
                
                values = [key]
                for col in file1_columns:
                    values.append(file1_data.get(col, ""))
                    
                for col in file2_columns:
                    values.append(file2_data.get(col, ""))
                
                self.matches_tree.insert("", tk.END, text="", values=values, tags=("match",))
            
            # Configure tag for matches
            self.matches_tree.tag_configure("match", background="#e6ffe6")
    
    def update_conflicts_treeview(self, conflicts):
        """Update the conflicts treeview"""
        if not conflicts:
            return
            
        # Clear previous content
        for item in self.conflicts_tree.get_children():
            self.conflicts_tree.delete(item)
            
        # Get all columns from the first conflict
        if conflicts:
            file1_columns = list(conflicts[0]["file1_data"].keys())
            file2_columns = list(conflicts[0]["file2_data"].keys())
            
            # Configure columns
            self.conflicts_tree["columns"] = ("key", "resolution") + tuple(file1_columns) + tuple(f"file2_{col}" for col in file2_columns)
            self.conflicts_tree.column("#0", width=0, stretch=tk.NO)
            self.conflicts_tree.column("key", width=150, anchor=tk.W)
            self.conflicts_tree.column("resolution", width=100, anchor=tk.W)
            
            for col in file1_columns:
                self.conflicts_tree.column(col, width=100, anchor=tk.W)
                
            for col in file2_columns:
                col_name = f"file2_{col}"
                self.conflicts_tree.column(col_name, width=100, anchor=tk.W)
            
            # Configure headings
            self.conflicts_tree.heading("#0", text="")
            self.conflicts_tree.heading("key", text="Key")
            self.conflicts_tree.heading("resolution", text="Resolution")
            
            for col in file1_columns:
                self.conflicts_tree.heading(col, text=f"File 1: {col}")
                
            for col in file2_columns:
                col_name = f"file2_{col}"
                self.conflicts_tree.heading(col_name, text=f"File 2: {col}")
            
            # Add data
            for i, conflict in enumerate(conflicts):
                key = conflict["key"]
                file1_data = conflict["file1_data"]
                file2_data = conflict["file2_data"]
                
                # Add resolution dropdown
                resolution_var = tk.StringVar(value="file1")
                self.per_conflict_resolutions[key] = "file1"
                
                values = [key, "File 1"]
                for col in file1_columns:
                    values.append(file1_data.get(col, ""))
                    
                for col in file2_columns:
                    values.append(file2_data.get(col, ""))
                
                item_id = self.conflicts_tree.insert("", tk.END, text="", values=values, tags=("conflict",))
                
                # Add resolution dropdown
                resolution_frame = tk.Frame(self.conflicts_tree)
                resolution_dropdown = ttk.Combobox(resolution_frame, textvariable=resolution_var, 
                                                 values=["File 1", "File 2"], width=10)
                resolution_dropdown.pack(fill=tk.X, expand=True)
                
                # Bind dropdown change
                resolution_dropdown.bind("<<ComboboxSelected>>", 
                                       lambda event, key=key, var=resolution_var: 
                                       self.update_conflict_resolution(key, var.get()))
                
                # Place the dropdown in the tree
                self.conflicts_tree.item(item_id, values=values)
            
            # Configure tag for conflicts
            self.conflicts_tree.tag_configure("conflict", background="#ffe6e6")
    
    def update_conflict_resolution(self, key, resolution):
        """Update the resolution for a specific conflict"""
        if resolution == "File 1":
            self.per_conflict_resolutions[key] = "file1"
        else:
            self.per_conflict_resolutions[key] = "file2"
    
    def update_only_in_file1_treeview(self, only_in_file1):
        """Update the only in file 1 treeview"""
        if not only_in_file1:
            return
            
        # Clear previous content
        for item in self.only_in_file1_tree.get_children():
            self.only_in_file1_tree.delete(item)
            
        # Get all columns from the first item
        if only_in_file1:
            columns = list(only_in_file1[0]["data"].keys())
            
            # Configure columns
            self.only_in_file1_tree["columns"] = ("key",) + tuple(columns)
            self.only_in_file1_tree.column("#0", width=0, stretch=tk.NO)
            self.only_in_file1_tree.column("key", width=150, anchor=tk.W)
            
            for col in columns:
                self.only_in_file1_tree.column(col, width=100, anchor=tk.W)
            
            # Configure headings
            self.only_in_file1_tree.heading("#0", text="")
            self.only_in_file1_tree.heading("key", text="Key")
            
            for col in columns:
                self.only_in_file1_tree.heading(col, text=col)
            
            # Add data
            for i, item in enumerate(only_in_file1):
                key = item["key"]
                data = item["data"]
                
                values = [key]
                for col in columns:
                    values.append(data.get(col, ""))
                
                self.only_in_file1_tree.insert("", tk.END, text="", values=values, tags=("file1",))
            
            # Configure tag for file 1 only
            self.only_in_file1_tree.tag_configure("file1", background="#e6f2ff")
    
    def update_only_in_file2_treeview(self, only_in_file2):
        """Update the only in file 2 treeview"""
        if not only_in_file2:
            return
            
        # Clear previous content
        for item in self.only_in_file2_tree.get_children():
            self.only_in_file2_tree.delete(item)
            
        # Get all columns from the first item
        if only_in_file2:
            columns = list(only_in_file2[0]["data"].keys())
            
            # Configure columns
            self.only_in_file2_tree["columns"] = ("key",) + tuple(columns)
            self.only_in_file2_tree.column("#0", width=0, stretch=tk.NO)
            self.only_in_file2_tree.column("key", width=150, anchor=tk.W)
            
            for col in columns:
                self.only_in_file2_tree.column(col, width=100, anchor=tk.W)
            
            # Configure headings
            self.only_in_file2_tree.heading("#0", text="")
            self.only_in_file2_tree.heading("key", text="Key")
            
            for col in columns:
                self.only_in_file2_tree.heading(col, text=col)
            
            # Add data
            for i, item in enumerate(only_in_file2):
                key = item["key"]
                data = item["data"]
                
                values = [key]
                for col in columns:
                    values.append(data.get(col, ""))
                
                self.only_in_file2_tree.insert("", tk.END, text="", values=values, tags=("file2",))
            
            # Configure tag for file 2 only
            self.only_in_file2_tree.tag_configure("file2", background="#fff2e6")
    
    def merge_files(self):
        """Merge the files based on the comparison results"""
        if not self.comparison_results:
            messagebox.showwarning("Warning", "Please compare files first")
            return
            
        # Get merge options
        conflict_resolution = self.conflict_resolution_var.get()
        include_only_in_file1 = self.include_only_in_file1_var.get()
        include_only_in_file2 = self.include_only_in_file2_var.get()
        
        # Update selected columns
        self.update_selected_columns()
        
        # Ask for output file
        output_path = filedialog.asksaveasfilename(
            title="Save Merged File",
            filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx;*.xls")],
            defaultextension=".csv"
        )
        
        if not output_path:
            return
            
        # Disable merge button during merging
        self.merge_btn.configure(state=tk.DISABLED, text="Merging...")
        
        # Log merge start
        self.log(f"Starting merge with options: Conflict Resolution={conflict_resolution}, "
                f"Include Only in File 1={include_only_in_file1}, Include Only in File 2={include_only_in_file2}")
        
        # Start merging in a separate thread
        self.merge_thread = threading.Thread(
            target=self._merge_files_thread,
            args=(conflict_resolution, include_only_in_file1, include_only_in_file2, 
                 self.selected_columns, self.per_conflict_resolutions, output_path)
        )
        self.merge_thread.daemon = True
        self.merge_thread.start()
    
    def _merge_files_thread(self, conflict_resolution, include_only_in_file1, include_only_in_file2, 
                           selected_columns, per_conflict_resolutions, output_path):
        """Thread function for merging files"""
        try:
            # Merge the files
            merged_df, stats = self.csv_comparer.merge_files(
                conflict_resolution, include_only_in_file1, include_only_in_file2,
                selected_columns, per_conflict_resolutions
            )
            
            # Save the merged file
            success, result = self.csv_comparer.save_merged_file(merged_df, output_path)
            
            if success:
                self.log(f"Successfully merged and saved file: {result}")
                self.display_merge_statistics(stats)
                messagebox.showinfo("Success", f"Files successfully merged and saved to: {result}")
            else:
                self.log(f"Error saving merged file: {result}")
                messagebox.showerror("Error", f"Failed to save merged file: {result}")
                
        except Exception as e:
            self.log(f"Error during merging: {str(e)}")
            messagebox.showerror("Error", f"An error occurred during merging: {str(e)}")
            
        finally:
            # Re-enable merge button
            self.merge_btn.configure(state=tk.NORMAL, text="Merge Files")
    
    def display_merge_statistics(self, stats: Dict):
        """Display merge statistics"""
        self.stats_text.configure(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        
        self.stats_text.insert(tk.END, "Merge Statistics\n")
        self.stats_text.insert(tk.END, "===============\n\n")
        
        self.stats_text.insert(tk.END, f"Total records: {stats['total_records']}\n")
        self.stats_text.insert(tk.END, f"Matched records: {stats['matched_records']}\n")
        self.stats_text.insert(tk.END, f"Conflict records: {stats['conflict_records']}\n")
        self.stats_text.insert(tk.END, f"Records only in file 1: {stats['only_in_file1_records']}\n")
        self.stats_text.insert(tk.END, f"Records only in file 2: {stats['only_in_file2_records']}\n")
        
        self.stats_text.configure(state=tk.DISABLED)


def main():
    app = PhoneFormatterApp()
    app.mainloop()


if __name__ == "__main__":
    main()
