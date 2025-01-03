import os # manipulate the files and directories
import zipfile # unzipp data.zip
import pandas as pd # for manipulating the dataset

def extract_zip(zip_path: str, extract_to: str) -> None:
    """
    Extracts a zip file to the specified directory.

    Args:
        zip_path (str): The path to the zip file.
        extract_to (str): The directory where the zip contents will be extracted.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def load_csv_from_zip(extracted_dir: str, filename: str) -> pd.DataFrame:
    """
    Loads a CSV file from the extracted directory.

    Args:
        extracted_dir (str): The directory where the zip contents were extracted.
        filename (str): The name of the CSV file to load.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    file_path = os.path.join(extracted_dir, filename)
    
    try:
        # Specify dtype for column 7 as integer
        df = pd.read_csv(
            file_path,
            index_col=0,
            dtype={7: int},  # Replace 7 with the actual column name or index
            low_memory=False
        )
        
        return df
    
    except ValueError as e:
        print("ValueError encountered while loading the CSV. Attempting to fix...")
        
        # Load without specifying dtype, then clean the column
        df = pd.read_csv(file_path, index_col=0, low_memory=False)
        df.iloc[:, 7] = pd.to_numeric(df.iloc[:, 7], errors='coerce').fillna(0).astype(int)
        
        return df
    
def load_data(zip_path: str, filename: str, extract_to) -> pd.DataFrame:
    """
    Orchestrates the extraction and loading of data from a zip file.

    Args:
        zip_path (str): The path to the zip file.
        filename (str): The name of the CSV file to load.

    Returns:
        pd.DataFrame: The processed data as a pandas DataFrame.
    """
    try:
        
        extract_zip(zip_path, extract_to)
    
        df = load_csv_from_zip(extract_to, filename)
    
        return df
    
    except Exception as e:
        raise RuntimeError(f'Error loading data: {str(e)}') 