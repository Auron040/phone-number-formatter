# Phone Number Formatter

A Streamlit application for processing and standardizing phone numbers in CSV and Excel files.

## Features

- **Phone Number Formatting**: Add country codes and standardize phone number formats
- **CSV Comparison**: Compare two CSV files and identify differences
- **File Merging**: Merge two CSV files with conflict resolution options
- **Validation**: Validate phone numbers against international standards
- **Multiple File Support**: Process multiple files at once
- **Detailed Statistics**: View detailed statistics about processed phone numbers

## Modes

- **Add Mode**: Simply add country code without modifying the number
- **Clean Mode**: Standardize numbers by removing formatting variations and adding country code

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/phone-number-formatter.git
   cd phone-number-formatter
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit application:
```
streamlit run streamlit_app.py
```

## Sample Data

The repository includes a sample CSV file (`sample_data.csv`) with example phone numbers for testing.

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- Phonenumbers
