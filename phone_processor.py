import pandas as pd
import phonenumbers
import re
from typing import Dict

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
            
        # Remove leading 0 if present
        if digits_only.startswith('0'):
            digits_only = digits_only[1:]
            
        # Add country code
        result = self.country_code + digits_only
        return result
    
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
