from langdetect import LangDetectException, detect


def detect_language(text: str) -> str:
    """Detect the language of the text. 
    If it is not possible to detect the language, 
    return an empty string and let the llm handle the problem itself."""
    try:
        return detect(text)
    except LangDetectException:
        return ""

def is_rtl_language(text):
    rtl_languages = {'ar', 'he', 'fa', 'ur'}
    try:
        detected_lang = detect(text)
        return detected_lang in rtl_languages
    except:
        return False
    

TONE_MAPPING = {
    "Keiner": None,
    "Formell": "Formal",
    "Informell": "Informal",
    "Technisch": "Technical",
}

DOMAIN_MAPPING = {
    "Keines": None,
    "Behörden": "Government",
    "Rechtswesen": "Legal",
    "Medizin": "Medical",
    "Technik": "Technical",
    "Finanzen": "Financial",
    "Wissenschaft": "Scientific",
    "Marketing": "Marketing",
    "Literatur": "Literary",
    "Bildung": "Educational",
    "Gastgewerbe und Tourismus": "Hospitality and Tourism",
    "Informationstechnologie": "Information Technology",
    "Landwirtschaft": "Agriculture",
    "Energie": "Energy",
    "Immobilien": "Real Estate",
    "Personalwesen": "Human Resources",
    "Pharmazie": "Pharmaceutical",
    "Kunst und Kultur": "Art and Culture",
    "Logistik und Transport": "Logistics and Transportation",
}

LANGUAGE_MAPPING = {
    "Automatisch erkennen": "Auto-detect",
    "Deutsch": "German",
    "Englisch": "English",
    "Französisch": "French",
    "Italienisch": "Italian",
    "Spanisch": "Spanish",
    # "Albanisch": "Albanian",
    # "Arabisch": "Arabic",
    # "Armenisch": "Armenian",
    # "Bosnisch": "Bosnian",
    # "Bulgarisch": "Bulgarian",
    # "Chinesisch (Mandarin)": "Chinese (Mandarin)",
    # "Dänisch": "Danish",
    # "Finnisch": "Finnish",
    # "Griechisch": "Greek",
    "Hindi": "Hindi",
    # "Indonesisch": "Indonesian",
    # "Japanisch": "Japanese",
    # "Koreanisch": "Korean",
    # "Kroatisch": "Croatian",
    # "Mazedonisch": "Macedonian",
    # "Malay": "Malay",
    # "Niederländisch": "Dutch",
    # "Norwegisch": "Norwegian",
    # "Polnisch": "Polish",
    "Portugiesisch": "Portuguese",
    # "Rumänisch": "Romanian",
    # "Russisch": "Russian",
    # "Schwedisch": "Swedish",
    # "Serbisch": "Serbian",
    "Thailändisch": "Thai",
    # "Türkisch": "Turkish",
    # "Ungarisch": "Hungarian",
    # "Vietnamesisch": "Vietnamese",
    # "Hebräisch": "Hebrew"
}
