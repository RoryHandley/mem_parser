#!/usr/bin/evn python3
# -*- coding: utf-8 -*-
"""
File name: Mem_parser.py
Author: Rory Handley
Contact: 
Created: 25/09/2024
Version: 2.0
Description: Script to help troubleshoot High Memory Usage Tickets
"""

# Import Standard Libraries 
import os
import tarfile as tf
import gzip
from datetime import datetime
import logging # Todo
import external_libraries
from typing import Optional

# Import External Libraries
import matplotlib.pyplot as plt
import pandas as pd
import configparser

# Define Global Constant(s)
CURRENT_PATH = os.getcwd() # Revisit
CURRENT_YEAR = (str(datetime.now().year)) + '-' # Had to add the hyphen because this let to unintentional matches on PIDs that started with current year. Need to find better solution. 

# Define my global dictionaries/lists. Poor naming convention currently
memstats_dict = {} 
topstats_dict = {}
data_list = []

# Declare our user-changeable global config
file_names = []
attribute = ""
num_lines_per_date = int()
topstat_columns = []

def set_config():
    "Function to set user-changable config based on config.ini file"
    
    # Instantiating the ConfigParser class to create an instance called config. 
    config = configparser.ConfigParser()

    # Invoking the read method with .ini file name as argument. Function will return a list of successfully read files, in this case config.ini.
    print(config.read("config.ini"))
    
    # Declare these variables as Global so we can access them outside of the function. Probably bad practice. 
    global file_names
    global attribute
    global num_lines_per_date
    global topstat_columns

    # Assign values to our global user-changeable variables by reading them from the config.ini file. ini contains [SectionName] and key value pairs
    file_names = config['FILES']['file_names'].split(',')
    attribute = config['MEMORY']['attribute']
    num_lines_per_date = int(config['MEMORY']['num_lines_per_date'])
    topstat_columns = config['TOPSTAT']['topstat_columns'].split(',') # % is a special character in configparser so need to escape it by using double %


def check_file_exists() -> Optional[str]: 
    "Return str of cwd if cwd conatins any files with .tar.gz extensions"
    for original_fname in os.listdir(CURRENT_PATH):
        if original_fname.endswith(".tar.gz"):
            print(f"File found: {original_fname}")
            return original_fname
    else:
        print(f"No .tar.gz files found in {CURRENT_PATH}")
        return None

def extract_tar_files(original_fname, file_path_extracted:str) -> bool:
    "Extract files from .tar.gz archive and put them into file_path_extracted directory"
      
    try:
        # Using 'with' so object is automatically closed after we're done with it.
        with tf.TarFile.open(os.path.abspath(original_fname), "r:gz") as tar_file_object:
            tar_file_object.extractall(file_path_extracted)
            print(f"Action complete. Extracted files in {file_path_extracted}. ")
        return True
    except:
        # Not sure what error we could hit here, but will come back and add specific errors
        print("Extraction failed!")
        return False

def open_file(file: str) -> object:
    "Function to open file depending on if it's zipped or not"
    if str(file).endswith(".gz"): # If it's zipped we need to open with gzip
        print(f"Found .gz file {file}")
        file = gzip.open(file, 'rt')
    else:
        print(f"Found non-gz file {file}")
        file = open(file, "r")

    return file # Returns the File object


def get_file_dict() -> dict:
    "Creates a dictionary where key = file name, value = list of files that match key"

    # Declare our dictionary
    filtered_files_dict = {}

    """ 
    - Note I previously used os.walk() for below code but I dropped it because of it's recursive nature (I only care about the files in current directory)
    - os.walk() returns a tuple for each directory it visits. The tuple consists of 1) The current directory path (a string), 2) A list of directories in the current directory and 3) A list of files in the current directory. 
    - In the below code, we are using multiple assignment to unpack the tuple values to a variable.
    - '_' indictaes the variable is temporary or insignificant i.e. we only care about the list of files below.
    
    **Previous Code**
    for _, _, files in os.walk(os.getcwd()): 
        filtered_files_dict[file_names[0]] = [file for file in files if file.startswith(file_names[0])]
        filtered_files_dict[file_names[1]] = [file for file in files if file.startswith(file_names[1])]
        break # This is bad code. The reason it's here is because I had an issue with above for loop - it correctly updated my dicitonary but then it went back and overwrote it so values were blank. I need to go back and better understand what's happening here. 
    """

    filtered_files_dict[file_names[0]] = [file for file in os.listdir(os.getcwd()) if file.startswith(file_names[0])]
    filtered_files_dict[file_names[1]] = [file for file in os.listdir(os.getcwd()) if file.startswith(file_names[1])]

    return filtered_files_dict


def parse_memstats(memstat_file):
    "Parse the memstats file and update global dicitionary"
    
    # I don't want to return a new dict every time this function is run so I'd rather configure a global dictionary although this is bad practice.
    global memstats_dict 

    """
    Note I originally had list comprehension here but I thought it was inefficient. E.g. we had to go through the entire doc twice - Once to check for dates and add to list and twice to check for data and add to list. I couldn't figure out a way to do that with list comprehsnion so I used below for loop. 
    """

    for row in memstat_file:
        # Remove the whitespace and newline characters
        row = str(row).strip()

        # Check if row starts with current year
        if row.startswith(CURRENT_YEAR): # regex 
            date = pd.to_datetime(row) # Convert to datetime object
            memstats_dict[date] = None # Initialize the key for MemFree value

        # Check if row starts with "MemFree"
        elif row.startswith("MemFree"):
            # Extract memory value by creating a list with : as delimiter and give me the string at index 1.
            mem_free_value = int(row.split(":")[1].removesuffix(" kB"))/1024 # Are we stripping whitespace?

            # Check memstats_dict1 is truthy i.e. not empty in this case.
            if memstats_dict: 
                # Assign the value to the latest date key (value should be none)
                last_date = max(memstats_dict.keys())
                memstats_dict[last_date] = mem_free_value

    # One way to improve this would be to jump straight to next date and look for MemFree from there instead of having to go through all the intermediary rows. 

    memstat_file.close()

def analyse_memstat_data(memstats_dict: dict): 
    "Analyse the data and produce a graph"
    
    """
    Instantiatie Series class as myvar using memstats_dict as an arguement. 
    Panda series is one-dimensional array i.e. a column. In this case, we will use the keys of the dictionary as the index and the values will be memfree values.
    """

    myvar = pd.Series(memstats_dict)
    
    # Smooth the data using a rolling window (you can adjust the window size)
    window_size = 10 

    # rolling allows us to perform a calculation on a window size of data, in this case calculate the mean of every 10 values, allowing us to smooth our data out.  
    smoothed_myvar = myvar.rolling(window=window_size).mean()

    # Plot the scatter plot (scatter(x axis, y axis, marker size))

    plt.scatter(smoothed_myvar.index, smoothed_myvar, s=0.1)

    # Customize the plot using Matplotlib
    plt.xticks([])  # Remove x-axis ticks to avoid clutter
    plt.xlabel(f"Time Range: {smoothed_myvar.index.min().strftime('%Y-%m-%d %H:%M:%S')} to {smoothed_myvar.index.max().strftime('%Y-%m-%d %H:%M:%S')}")  # Specify time range. Remember index is our times. strftime just formats. 
    plt.ylabel("MemFree (MB)")
    plt.title("MemFree over Time")

    # Display the plot created in the previous steps in a new window
    plt.show()


def parse_topstats(topstat_file:str):
    "Parse the topstats file and update global list"

    # Global variables - Poorly named
    global topstats_dict
    global data_list
    
    # Intermidary lists - Shouldn't be necessary
    topstats_date_list = []
    topstats_column_list = []
    topstats_data_list = []

    # Summary list
    topstats_summary_list = []
    
    # Remove the leading and trailing whitespace and newline characters. Best way I could think to do it without duplicating .split although we are now processing rows we don't care about.
    rows = [str(row).strip() for row in topstat_file]

    topstat_file.close() # Should close this as soon as possible.

    """ 
    - We only care about the first list of columns since they will never change. Don't want to waste processing.
    - next function takes a generator expression as an argument. A generator expression is similar to list comprehension but doesn't generate the entire list at once, instead it generates items one by one so it's more memory effcient. 
    - If there are no rows that match the condition, topstats_column_list will be set to None. We currently aren't set up to handle that. 
    """
    topstats_column_list = next(((row.split()) for row in rows if row.startswith("PID")), None)

    """
    **PREVIOUS CODE**
    # Contruct a list of columns. 
    for row in rows:
        if row.startswith("PID"):
            # Split into a list of elements separated by space
            topstats_column_list = row.split()

            break # Note I only want to do this once since the columns will not change and I don't want to waste processing. This seems dumb - do I need a for loop?
    """

    # Contruct a list of dates
    topstats_date_list = [pd.to_datetime(row) 
                          for row in rows 
                          if row.startswith(CURRENT_YEAR)]

    # Contruct a data list if the row isn't null and the first character is a digit. This is now broken because it's picking up dates when it shouldnt. Slapdash fix.
    # This doesn't seem to be working right now. 
    topstats_data_list = [row.split()
                          for row in rows 
                          if row and row[0].isdigit() and not row.startswith(CURRENT_YEAR)]
   
    

    # We now have 3 separate lists. 1) with the dates, 2) with the columns and 3) with the actual data. Now need to combine this so we can build the dataframe in the next function. 

    for row in topstats_data_list:
        topstats_dict1 = {} # Need to reset the dicitonary each time so we can add new values for our column keys. 
        for column, data in zip(topstats_column_list, row):
            if column in topstat_columns: # Filter out columns we don't care about
                topstats_dict1[column] = data # Should really be converting %MEM columns to floats
        # Note topstats_summary list is referencing the values of the dicitonary not the dictionary itself.
        topstats_summary_list.append(topstats_dict1) # We now have a list of dictionaries where each dictionary has a column = key and data = value e.g. PID:138099, USER:root etc. 

    # Each date should be clumped together with the next 16 rows of data which we can do with below list comprehension [expression iterable condition (condition is optional)]

    grouped_process_data = [
        topstats_summary_list[i:i + num_lines_per_date] # Expression - slice list from i to i + 16
        for i in range(0, len(topstats_summary_list), num_lines_per_date) # Iterable - range(start, stop, step) so 0, 16, 32 etc
    ]

    # So now we have a list grouped_process_data where each element in the outer list is a list of 16 dictionaries. We now want to clump together the list of 16 dictionaries with their corresponding date. So date = key, list of 16 dictionaries = value.  
    
    for date, inner in zip(topstats_date_list, grouped_process_data): 
        topstats_dict[date] = inner

    

    # We now need to flatten out the data, although it feels like it should have been done beforehand. 

    for timestamp, processes in topstats_dict.items(): # Where processes is the list of 16 dictionaries associated with each timestamp
        for process in processes: # For each individual dictionary of the 16
            process['timestamp'] = timestamp  # Add a timestamp key with value = to the timestamp. 
            data_list.append(process)  # Append each process to the list. We now have a list where every entry is a dictionary corresponding to one row on the table and with a timestamp recorded.



def analyse_topstat_data(data_list: list):
    "Analyse the data and produce a graph"

    df = pd.DataFrame(data_list)

    # Ensure our columns are of the correct datatype
    df[topstat_columns[0]] = df[topstat_columns[0]].astype(float)  # %MEM
    df[topstat_columns[1]] = df[topstat_columns[1]].astype(str)    # Process name
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Ensure timestamp is datetime

    # Group by process and timestamp, summing the %MEM
    grouped_df = df.groupby([topstat_columns[1], 'timestamp'], as_index=False)[topstat_columns[0]].sum()

    # Find the top 5 processes by total memory usage
    total_mem_by_process = grouped_df.groupby(topstat_columns[1])[topstat_columns[0]].sum().nlargest(5).index.tolist()

    # Filter the grouped DataFrame to only include top 5 processes
    top_5_df = grouped_df[grouped_df[topstat_columns[1]].isin(total_mem_by_process)]

    # Plotting
    plt.figure(figsize=(12, 6))

    for process in total_mem_by_process:
        process_data = top_5_df[top_5_df[topstat_columns[1]] == process]
        plt.scatter(process_data['timestamp'], process_data[topstat_columns[0]], label=process)

    # Add labels and title
    plt.ylabel('Memory Usage (%MEM)')
    plt.xticks([])
    plt.title('Top 5 Memory Users Over Time')
    date_range_label = f"Time Range: {top_5_df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')} to {top_5_df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')}"
    plt.xlabel(date_range_label) 

    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Note each timestamp has two values for each process - one for each core. Need to research which core we care about. 



def main():
    "Main function"
    # Check we have the necessary dependencies and install/import if not
    # external_libraries.dependencies_check()

    # Read the config.ini file to set user-configurable values. Probably unnecessary but cool to try. 
    set_config()

    """
    - check_file_exists is first called which returns a value. 
    - This returned value is then set equal to original_fname using the walrus operator ':='. 
    - 'if original_fname' then checks if the value associated with original_fname is truthy i.e. not 0, not False, not None, not Empty
    """
    if original_fname := check_file_exists():
        file_path_extracted = CURRENT_PATH + "\\" + original_fname.removesuffix(".tar.gz")
        
        if extract_tar_files(original_fname, file_path_extracted):        
            os.chdir(file_path_extracted  + r"\system\statistics")
            
            # Need to account for case where user has removed a file from config.ini

            """
            # memstats
            for file in get_file_dict()[file_names[0]]: 
                memstat_file = open_file(file) # Possible file could be open by another program causing an exception
                parse_memstats(memstat_file)
            analyse_memstat_data(memstats_dict)
            """

            # topstats
            for file in get_file_dict()[file_names[1]]: 
                topstat_file = open_file(file)
                parse_topstats(topstat_file)
            analyse_topstat_data(data_list)


try:
    main()
except KeyboardInterrupt:
    external_libraries.graceful_exit(2)
except PermissionError:
    print("Known Bug hit: Extracted file already present. To be Resolved.")
finally:
    print("Exiting Programme..")



"""
To-Dos V2 
- Account for case where user removes file name from config.ini
- Opening memstats.gz files is very slow. (Maybe just in debugger which is normal)
- Remove reliance on intermediate dictionaries/lists.
- Remove reliance on global variables. 
- Resolve bug hit when extracted file already present. 
- Auto-installation of necessary dependencies.
- Two cores - need validation on topstats file makeup.
"""




























