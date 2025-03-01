import os
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Set
from phone_processor import PhoneNumberProcessor

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
    
    def read_file(self, file_data, file_name: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Read a CSV or Excel file and return a DataFrame"""
        try:
            file_ext = os.path.splitext(file_name)[1].lower()
            
            if file_ext == '.csv':
                # Try different encodings
                try:
                    df = pd.read_csv(file_data, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(file_data, encoding='latin1')
                    except:
                        df = pd.read_csv(file_data, encoding='cp1252')
                except pd.errors.EmptyDataError:
                    return None, "The file is empty"
                
                self.log(f"Successfully read CSV file: {file_name}")
                return df, None
                
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_data)
                self.log(f"Successfully read Excel file: {file_name}")
                return df, None
                
            else:
                return None, f"Unsupported file format: {file_ext}"
                
        except Exception as e:
            error_msg = f"Error reading file: {str(e)}"
            self.log(error_msg)
            return None, error_msg
    
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
