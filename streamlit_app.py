import os
import sys
import pandas as pd
import streamlit as st
import numpy as np
import io

from phone_processor import PhoneNumberProcessor, COUNTRY_CODES
from file_processor import FileProcessor
from csv_comparer import CSVComparer
from utils import get_download_link, log_message, display_log, init_session_state

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Phone Number Formatter",
        page_icon="📱",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # App title and description
    st.title("Phone Number Formatter")
    st.markdown("A tool for processing and standardizing phone numbers in CSV and Excel files.")
    
    # Create tabs for different functionality
    tabs = st.tabs(["Phone Formatter", "CSV Comparison", "Log"])
    
    # Sidebar for common options
    with st.sidebar:
        st.header("Options")
        
        # Country code selection
        st.subheader("Country Code")
        country_options = list(COUNTRY_CODES.keys())
        selected_country = st.selectbox("Select Country", country_options, index=0)
        
        # Handle custom country code
        if selected_country == "Other":
            custom_code = st.text_input("Enter Custom Code", value="+", help="Enter a custom country code (e.g., +49)")
            country_code = custom_code
        else:
            country_code = COUNTRY_CODES[selected_country]
            st.info(f"Selected country code: {country_code}")
        
        # Processing mode
        st.subheader("Processing Mode")
        mode = st.radio(
            "Select Mode",
            ["add", "clean"],
            format_func=lambda x: "Add Mode" if x == "add" else "Clean Mode",
            help="Add Mode: Simply add country code without modifying the number\nClean Mode: Standardize numbers by removing formatting variations and adding country code"
        )
        
        # Additional options
        st.subheader("Additional Options")
        validate = st.checkbox("Validate Phone Numbers", value=True, 
                              help="Check if the processed numbers are valid according to international standards")
        new_file = st.checkbox("Create New File", value=True,
                              help="Create a new file with '_processed' suffix instead of overwriting the original")
    
    # Phone Formatter tab
    with tabs[0]:
        st.header("Phone Number Formatter")
        
        # File upload section
        st.subheader("Upload Files")
        uploaded_files = st.file_uploader("Choose CSV or Excel files", 
                                         type=["csv", "xlsx", "xls"], 
                                         accept_multiple_files=True)
        
        if uploaded_files:
            # Display uploaded files
            st.write(f"Uploaded {len(uploaded_files)} file(s):")
            for file in uploaded_files:
                st.write(f"- {file.name}")
            
            # Column selection
            if len(uploaded_files) > 0:
                # Read the first file to get columns
                file_processor = FileProcessor(log_callback=log_message)
                file = uploaded_files[0]
                # Reset file position to the beginning
                file.seek(0)
                df, error = file_processor.read_file(file, file.name)
                
                if error:
                    st.error(f"Error reading file: {error}")
                else:
                    # Column selection
                    st.subheader("Column Selection")
                    column = st.selectbox("Select Phone Number Column", df.columns)
                    
                    # Preview data
                    st.subheader("Data Preview")
                    st.dataframe(df.head(5))
                    
                    # Process button
                    if st.button("Process Files"):
                        # Create processor
                        processor = PhoneNumberProcessor(country_code)
                        
                        # Process each file
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, file in enumerate(uploaded_files):
                            # Update progress
                            progress = (i / len(uploaded_files))
                            progress_bar.progress(progress)
                            status_text.text(f"Processing {file.name}...")
                            
                            # Read file - need to reset the file position to the beginning
                            file.seek(0)
                            df, error = file_processor.read_file(file, file.name)
                            
                            if error:
                                st.error(f"Error reading file {file.name}: {error}")
                                continue
                            
                            # Process file
                            processed_df, stats = file_processor.process_phone_numbers(
                                df, column, processor, mode, validate
                            )
                            
                            # Store results in session state
                            file_key = f"{file.name}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
                            st.session_state.processed_dfs[file_key] = processed_df
                            st.session_state.processing_stats[file_key] = stats
                            
                            # Log results
                            log_message(f"Processed {file.name}: {stats['processed']} numbers processed")
                        
                        # Complete progress
                        progress_bar.progress(1.0)
                        status_text.text("Processing complete!")
                        
                        # Display results
                        st.subheader("Processing Results")
                        
                        # Create tabs for each processed file
                        if st.session_state.processed_dfs:
                            file_tabs = st.tabs([key.split('_')[0] for key in st.session_state.processed_dfs.keys()])
                            
                            for i, (file_key, df) in enumerate(st.session_state.processed_dfs.items()):
                                with file_tabs[i]:
                                    # Display statistics
                                    stats = st.session_state.processing_stats[file_key]
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.subheader("Statistics")
                                        st.write(f"Total records: {stats['total']}")
                                        st.write(f"Processed records: {stats['processed']}")
                                        st.write(f"Empty records: {stats['empty']}")
                                        
                                        if 'valid' in stats:
                                            st.write(f"Valid numbers: {stats['valid']}")
                                            st.write(f"Invalid numbers: {stats['invalid']}")
                                    
                                    with col2:
                                        if 'types' in stats and stats['types']:
                                            st.subheader("Number Types")
                                            for type_name, count in stats['types'].items():
                                                st.write(f"{type_name}: {count}")
                                    
                                    # Display processed data
                                    st.subheader("Processed Data")
                                    st.dataframe(df)
                                    
                                    # Download link
                                    file_name = file_key.split('_')[0]
                                    name, ext = os.path.splitext(file_name)
                                    # Make sure the extension is .csv
                                    download_name = f"{name}_processed.csv"
                                    
                                    csv = df.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label="Download Processed File",
                                        data=csv,
                                        file_name=download_name,
                                        mime="text/csv",
                                        key=f"download_{file_key}"  # Add unique key
                                    )
        else:
            st.info("Please upload one or more CSV or Excel files to process.")
    
    # CSV Comparison tab
    with tabs[1]:
        st.header("CSV Comparison")
        
        # File upload section for comparison
        st.subheader("Upload Files for Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("File 1")
            file1 = st.file_uploader("Choose first CSV or Excel file", 
                                    type=["csv", "xlsx", "xls"], 
                                    key="comparison_file1")
        
        with col2:
            st.write("File 2")
            file2 = st.file_uploader("Choose second CSV or Excel file", 
                                    type=["csv", "xlsx", "xls"], 
                                    key="comparison_file2")
        
        if file1 and file2:
            # Read files
            file_processor = FileProcessor(log_callback=log_message)
            
            # Reset file positions
            file1.seek(0)
            file2.seek(0)
            
            df1, error1 = file_processor.read_file(file1, file1.name)
            df2, error2 = file_processor.read_file(file2, file2.name)
            
            if error1:
                st.error(f"Error reading file 1: {error1}")
            elif error2:
                st.error(f"Error reading file 2: {error2}")
            else:
                # Key column selection
                st.subheader("Key Column Selection")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    key_column = st.selectbox("Select Key Column", df1.columns)
                
                with col2:
                    normalize_phone = st.checkbox("Normalize Phone Numbers for Comparison", value=True,
                                                help="Normalize phone numbers by removing country codes and formatting for better matching")
                
                # Comparison options
                st.subheader("Comparison Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    conflict_resolution = st.radio(
                        "Default Conflict Resolution",
                        ["file1", "file2"],
                        format_func=lambda x: "File 1 Takes Precedence" if x == "file1" else "File 2 Takes Precedence"
                    )
                
                with col2:
                    include_only_in_file1 = st.checkbox("Include Records Only in File 1", value=True)
                    include_only_in_file2 = st.checkbox("Include Records Only in File 2", value=True)
                
                # Compare button
                if st.button("Compare Files"):
                    # Create CSV comparer
                    csv_comparer = CSVComparer(file_processor, log_callback=log_message)
                    
                    # Load files
                    success, error = csv_comparer.load_files(df1, df2, file1.name, file2.name)
                    
                    if not success:
                        st.error(f"Error loading files: {error}")
                    else:
                        # Prepare for comparison
                        success, error = csv_comparer.prepare_for_comparison(
                            key_column, normalize_phone, country_code
                        )
                        
                        if not success:
                            st.error(f"Error preparing for comparison: {error}")
                        else:
                            # Compare files
                            comparison_results = csv_comparer.compare_files()
                            
                            # Store results in session state
                            st.session_state.comparison_results = comparison_results
                            st.session_state.csv_comparer = csv_comparer
                            
                            # Reset per-conflict resolutions and selected columns
                            st.session_state.per_conflict_resolutions = {}
                            st.session_state.selected_columns = list(set(df1.columns) | set(df2.columns))
                            
                            # Log results
                            log_message(f"Comparison completed: {comparison_results['matches']['total_matches']} matches, "
                                      f"{len(comparison_results['conflicts'])} conflicts")
                            
                            # Rerun to show results
                            st.rerun()
                
                # Display comparison results if available
                if 'comparison_results' in st.session_state and st.session_state.comparison_results:
                    comparison_results = st.session_state.comparison_results
                    csv_comparer = st.session_state.csv_comparer
                    
                    # Display summary
                    st.subheader("Comparison Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Matching Records", comparison_results['matches']['total_matches'])
                    
                    with col2:
                        st.metric("Only in File 1", comparison_results['matches']['only_in_file1'])
                    
                    with col3:
                        st.metric("Only in File 2", comparison_results['matches']['only_in_file2'])
                    
                    # Column selection for merge
                    st.subheader("Column Selection for Merge")
                    
                    # Get all columns
                    all_columns = list(set(comparison_results["file1_columns"]) | set(comparison_results["file2_columns"]))
                    
                    # Create a multiselect for column selection
                    selected_columns = st.multiselect(
                        "Select Columns to Include in Merged File",
                        all_columns,
                        default=all_columns
                    )
                    
                    # Store selected columns in session state
                    st.session_state.selected_columns = selected_columns
                    
                    # Display preview data
                    preview_data = csv_comparer.get_preview_data()
                    
                    # Create tabs for preview data
                    preview_tabs = st.tabs(["Matches", "Conflicts", "Only in File 1", "Only in File 2"])
                    
                    # Matches tab
                    with preview_tabs[0]:
                        if preview_data["matches"]:
                            st.write(f"Showing {len(preview_data['matches'])} of {comparison_results['matches']['total_matches']} matches")
                            
                            for i, match in enumerate(preview_data["matches"]):
                                with st.expander(f"Match {i+1}: {match['key']}"):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write("File 1 Data")
                                        st.json(match["file1_data"])
                                    
                                    with col2:
                                        st.write("File 2 Data")
                                        st.json(match["file2_data"])
                        else:
                            st.info("No matches found")
                    
                    # Conflicts tab
                    with preview_tabs[1]:
                        if preview_data["conflicts"]:
                            st.write(f"Showing {len(preview_data['conflicts'])} of {len(comparison_results['conflicts'])} conflicts")
                            
                            for i, conflict in enumerate(preview_data["conflicts"]):
                                with st.expander(f"Conflict {i+1}: {conflict['key']}"):
                                    # Resolution selection
                                    resolution = st.radio(
                                        f"Resolution for {conflict['key']}",
                                        ["file1", "file2"],
                                        format_func=lambda x: "Use File 1" if x == "file1" else "Use File 2",
                                        key=f"conflict_{i}"
                                    )
                                    
                                    # Store resolution in session state
                                    st.session_state.per_conflict_resolutions[conflict['key']] = resolution
                                    
                                    # Display conflict details
                                    st.write("Conflicting Fields:")
                                    for field_conflict in conflict["conflicts"]:
                                        st.write(f"- {field_conflict['column']}: '{field_conflict['file1_value']}' vs '{field_conflict['file2_value']}'")
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write("File 1 Data")
                                        st.json(conflict["file1_data"])
                                    
                                    with col2:
                                        st.write("File 2 Data")
                                        st.json(conflict["file2_data"])
                        else:
                            st.info("No conflicts found")
                    
                    # Only in File 1 tab
                    with preview_tabs[2]:
                        if preview_data["only_in_file1"]:
                            st.write(f"Showing {len(preview_data['only_in_file1'])} of {comparison_results['matches']['only_in_file1']} records only in File 1")
                            
                            for i, record in enumerate(preview_data["only_in_file1"]):
                                with st.expander(f"Record {i+1}: {record['key']}"):
                                    st.json(record["data"])
                        else:
                            st.info("No records found only in File 1")
                    
                    # Only in File 2 tab
                    with preview_tabs[3]:
                        if preview_data["only_in_file2"]:
                            st.write(f"Showing {len(preview_data['only_in_file2'])} of {comparison_results['matches']['only_in_file2']} records only in File 2")
                            
                            for i, record in enumerate(preview_data["only_in_file2"]):
                                with st.expander(f"Record {i+1}: {record['key']}"):
                                    st.json(record["data"])
                        else:
                            st.info("No records found only in File 2")
                    
                    # Merge button
                    if st.button("Merge Files"):
                        # Merge files
                        merged_df, merge_stats = csv_comparer.merge_files(
                            conflict_resolution,
                            include_only_in_file1,
                            include_only_in_file2,
                            selected_columns,
                            st.session_state.per_conflict_resolutions
                        )
                        
                        # Store results in session state
                        st.session_state.merged_df = merged_df
                        st.session_state.merge_stats = merge_stats
                        
                        # Log results
                        log_message(f"Merge completed: {merge_stats['total_records']} total records")
                        
                        # Rerun to show results
                        st.rerun()
                    
                    # Display merge results if available
                    if 'merged_df' in st.session_state and st.session_state.merged_df is not None:
                        merged_df = st.session_state.merged_df
                        merge_stats = st.session_state.merge_stats
                        
                        st.subheader("Merge Results")
                        
                        # Display statistics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"Total records: {merge_stats['total_records']}")
                            st.write(f"Matched records: {merge_stats['matched_records']}")
                            st.write(f"Conflict records: {merge_stats['conflict_records']}")
                        
                        with col2:
                            st.write(f"Records only in file 1: {merge_stats['only_in_file1_records']}")
                            st.write(f"Records only in file 2: {merge_stats['only_in_file2_records']}")
                        
                        # Display merged data
                        st.dataframe(merged_df)
                        
                        # Download button
                        csv = merged_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Merged File",
                            data=csv,
                            file_name="merged_data.csv",
                            mime="text/csv",
                            key="download_merged_file"  # Add unique key
                        )
        else:
            st.info("Please upload two CSV or Excel files to compare.")
    
    # Log tab
    with tabs[2]:
        st.header("Processing Log")
        display_log()


if __name__ == "__main__":
    main()
