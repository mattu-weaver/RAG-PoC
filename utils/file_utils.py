import os

def count_pdf_files(folder_path: str) -> int:
    """
    Count the number of PDF files in the given folder.
    :param folder_path: The path to the folder.
    :return: The number of PDF files in the folder.
    """
    if not os.path.exists(folder_path):
        return 0
    
    return len([f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')])

def path_exists(folder_path: str) -> bool:
    """
    Check if the given path exists.
    :param folder_path: The path to check.
    :return: True if the path exists, False otherwise.
    """
    return os.path.exists(folder_path)
