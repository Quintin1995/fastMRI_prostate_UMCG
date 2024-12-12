import os

# Specify the network directory path
network_path = r'Z:\01_quintin\00_tmp'

# List all files and directories in the specified path
try:
    entries = os.listdir(network_path)
    files = [entry for entry in entries if os.path.isfile(os.path.join(network_path, entry))]
    print("Files in '{}':".format(network_path))
    for file in files:
        print(file)
except FileNotFoundError:
    print("The specified path does not exist.")
except PermissionError:
    print("You do not have the necessary permissions to access this path.")
except Exception as e:
    print(f"An error occurred: {e}")
