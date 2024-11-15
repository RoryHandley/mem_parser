#!/usr/bin/evn python3
# -*- coding: utf-8 -*-
"""
File name: Mem_parser.py
Author: Rory Handley
Contact: 
Created: 25/09/2024
Version: 3.0
Description: Script to help troubleshoot High Memory Usage Tickets
"""

# Import Standard Libraries 
import os
import tarfile as tf
import gzip
from datetime import datetime
import logging # Todo
from typing import Optional
import re
from collections import defaultdict

# Import External Libraries
import matplotlib.pyplot as plt
import pandas as pd
import configparser

# Define Global Constant(s)
CURRENT_PATH = os.getcwd() # Revisit
CURRENT_YEAR = (str(datetime.now().year)) + '-' # Had to add the hyphen because this let to unintentional matches on PIDs that started with current year. Need to find better solution. 

# Define my global dictionaries/lists.
memstats_dict = {} 

topstat_global_data_list = []
topstats_file_top_5 = defaultdict(str)

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
        # Compile regex pattern to match paths that start with "topstats" or "memstats"
        pattern = re.compile(r"^(system/statistics/topstats|system/statistics/memstats)")
        with tf.TarFile.open(os.path.abspath(original_fname), "r:gz") as tar_file_object:
            for member in tar_file_object.getmembers():
                if pattern.match(member.name):
                    tar_file_object.extract(member, file_path_extracted)
                    print(f"Extracted {member.name} to {file_path_extracted}")
        print("Extraction complete for specified files.")
        return True
    except (tf.ExtractError, IOError) as e:
        print(f"An extraction error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
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


def parse_topstats(topstat_file:object) -> list:
    "Parse the topstats file and return a data list"

    # Initialize Lists/dictionaries for each function call
    topstat_file_data_list = []
    dates = []
    columns = []
    data_rows = []
    topstats_summary_list = []
    topstats_dict = {}
      
    # Remove the leading and trailing whitespace and newline characters. 
    rows = [str(row).strip() for row in topstat_file]
    # Close out the file as soon as possible.
    topstat_file.close()

    # Construct a list of columns - Columns wont change, so use next function to get first occurence.
    columns = next(((row.split()) for row in rows if row.startswith("PID")), None)

    # Contruct a list of dates
    dates = [pd.to_datetime(row) for row in rows if row.startswith(CURRENT_YEAR)]

    # Contruct a data list 
    data_rows = [row.split() for row in rows if row and row[0].isdigit() and not row.startswith(CURRENT_YEAR)]
   
    # We now have 3 separate lists. 1) with the dates, 2) with the columns and 3) with the actual data.  

    # Create a list of dictionaries where key = column and value = data value 
    # e.g. [{"COMMAND":"python", "%MEM":0.7}, {"COMMAND":"vpsn", "%MEM":0.3}.....]
    for row in data_rows:
        topstats_dict1 = {} # Need to reset the dicitonary each time so we can add new values for our column keys. 
        for column, data in zip(columns, row):
            if column in topstat_columns: # Filter out columns we don't care about
                if column == "%MEM":
                    topstats_dict1[column] = float(data)
                elif column == "COMMAND":
                    topstats_dict1[column] = data
        # Note topstats_summary list is referencing the values of the dicitonary not the dictionary itself.
        topstats_summary_list.append(topstats_dict1) # We now have a list of dictionaries where each dictionary has a column = key and data = value e.g. PID:138099, USER:root etc. 

    
    # Each date is associated with the next 16 rows of data, so let's create a new list with clumps of 16 rows together.
    grouped_data = [
        topstats_summary_list[i:i + num_lines_per_date] # Expression - slice list from i to i + 16
        for i in range(0, len(topstats_summary_list), num_lines_per_date) # Iterable - range(start, stop, step) so 0, 16, 32 etc
    ]


    # Now we need to make sure that there is only one process per clump of data. 
    # If there are multiple instances of the same process, then we should keep only the highest one. 

    # Create a hash map to count occurences of specific command in each inner dict
    for inner_list in grouped_data:
        # Create a new hash map for each list of dicitonaries
        command_hash_map = {}
        for inner_dict in inner_list:
            if inner_dict["COMMAND"] in command_hash_map:
                command_hash_map[inner_dict["COMMAND"]] += 1
            else: 
                # Deals with first case
                command_hash_map[inner_dict["COMMAND"]] = 1 


        for command in command_hash_map:
            # Only care about commands that occur more than once
            if command_hash_map[command] > 1:
                # This will be a value which we can use to compare
                highest = None
                # This will refer to the specifc dictionary that has the highest value
                highest_entry = None 
                
                for inner_dict in inner_list:
                    if (inner_dict["COMMAND"] == command): 
                        if highest is None or inner_dict["%MEM"] > highest:
                            highest = inner_dict["%MEM"]
                            highest_entry = inner_dict  # Update highest_entry to point at specific dictinoary
                
                # We cant remove elements from a list while iterating through it as it will cause indexing errors. 
                # By writing inner_list[:] = [...], you’re telling Python to replace the entire content of inner_list with the new list [ ... ], without creating a new list object. This differs from inner_list = [...], which would reassign inner_list to a new object, potentially breaking any references to the original list.
                inner_list[:] = [
                        inner_dict for inner_dict in inner_list
                        if (inner_dict["COMMAND"] != command) or (inner_dict is highest_entry)
                    ]

    # So now each of our inner lists contains only unique processes. If there was more than one process key, we kept the one with the highest %MEM value.
    # We now want to connect our dates with our list of dictionaries, which we can do by setting key = date and value = list of dictionaries
    for date, inner in zip(dates, grouped_data): 
        # where inner = list of dictionaries
        topstats_dict[date] = inner

    

    # We now need to flatten out the data, so every dictionary in the list has a date value
    for timestamp, processes in topstats_dict.items(): # Where processes is the list of dictionaries associated with each timestamp
        for process in processes: # For each individual dictionary
            process['timestamp'] = timestamp  # Add a timestamp key with value = to the timestamp. 
            topstat_file_data_list.append(process)  # Append each process to the list. 
    
    print(f"Calculated Top 5 Memory users in {topstat_file.name}.")

    # Now we return a list where every entry is a dictionary corresponding to one row in the table, with an associated timestamp for each row. 
    # E.g. [{'%MEM': 0.6, 'COMMAND': 'tuned', 'timestamp': Timestamp('2024-10-08 07:05:40+0000', tz='UTC')}.....]
    return topstat_file_data_list



def topstats_analyser(topstat_file_data_list):
    "Take in a topstat_file_data_list and return the top 5 memory users"

    unique_processes = []
    # Apparently I need to do this if my keys don't exist in the beginning
    total_memory_tracker = defaultdict(float)

    # Create a list of unique processes in our data
    # E.g. ['tuned', 'python3', 'polkitd', 'hasplmd_x86_64', 'systemd', 'top', 'kthreadd', 'rcu_gp']
    for dict in topstat_file_data_list:
        for key in dict:
            if (key == 'COMMAND') and (dict[key] not in unique_processes):
                unique_processes.append(dict[key])
    
    
    for specific_process in unique_processes:
        for entry in topstat_file_data_list:
            if entry.get('COMMAND') == specific_process:
                total_memory_tracker[specific_process] += entry.get("%MEM", 0)  # Add %MEM or default to 0 if missing

    # total_memory_tracker is now equal to a dictionary that looks like below e.g.:
    # {'Python': 6363252, 'VPSI': 546363652, 'SNMP': 525252563........}
    # No we want to sort this dictionary, but the problem is that dictionaries are inherently orderless by nature.  
    # Instead, we can convert our dictionary into a list of tuples and sort the tuples. (Recall tuples are ordered)

    sorted_items = sorted(total_memory_tracker.items(), key=lambda item: item[1], reverse=True)

    # total_memory_tracker.items() - will give us a list of tuples based on our dictionary keys and values e.g. [('Python', 6363252), ('VPSI', 546363652), ('SNMP', 525252563)]
    # key=lambda item: item[1] - key is a function determines what to sort by. In this case, it is a lambda expression where syntax is lambda arguement : expression. 
    # So this key is saying sort by element at index 1 of each tuple i.e. the total %MEM value.
    # Reverse = True is saying sort in descending order, so we are left with a list of tuples that are sorted in descending order of total %MEM usage. 
    # So we'll have [('VPSI', 546363652), ('SNMP', 525252563), ('Python', 6363252)]


    # Now we have these values, we want to display them somehow on a graph to view the memory usage trend.
    # x-axis is specific topstats file (although the label will be the date range covered by all of the files)
    # For each topstats file --> for each of the top 5 processes --> add a mark on the graph. 
    # Then add a trend line or just connect the dots. 
 
    # We only care about the top 5 processes
    sorted_items = sorted_items[0:5]
    
    # We will return a list of top 5 memory users sorted by %MEM for each topstats file. 
    return sorted_items

def create_topstats_graph(data):
    "Take a dictionary and produce a graph"

    # Extract unique processes
    all_processes = set(process for file_data in data.values() for process, _ in file_data)

    # Prepare data for plotting
    file_names = list(data.keys())  # x-axis points
    process_to_values = {process: [] for process in all_processes}

    # Fill in the values for each process, ensuring consistent ordering
    for file_name in file_names:
        file_data = dict(data[file_name])  # Convert list of tuples to a dictionary
        for process in all_processes:
            process_to_values[process].append(file_data.get(process, 0))  # Default to 0 if process is absent

    # Plot
    plt.figure(figsize=(12, 6))
    for process, values in process_to_values.items():
        plt.plot(file_names, values, marker='o', label=process)  # Plot each process as a line

    # Graph Labels and Legends
    plt.title("Top 5 Processes Across Files by %MEM Usage")
    plt.xlabel("Files")
    plt.ylabel("%MEM Usage")
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.legend(title="Processes")
    plt.grid(True)

    # Show the graph
    plt.tight_layout()
    plt.show()




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

            # memstats
            for file in get_file_dict()[file_names[0]]: 
                memstat_file = open_file(file) # Possible file could be open by another program causing an exception
                parse_memstats(memstat_file)
            analyse_memstat_data(memstats_dict)
            
            # topstats
            for file in get_file_dict()[file_names[1]]: 
                topstat_file = open_file(file)
                # Analyse topstat file for EACH topstat file
                topstats_file_top_5[file] = topstats_analyser(parse_topstats(topstat_file))
            
            create_topstats_graph(topstats_file_top_5)
            

            


try:
    main()
except KeyboardInterrupt:
    print("Exit")
    # external_libraries.graceful_exit(2)
except PermissionError:
    print("Known Bug hit: Extracted file already present. To be Resolved.")
finally:
    print("Exiting Programme..")






























