# Install Logodet dataset locally on this computer 
# Usage: logone/datasets/install_logodet.sh

# Save the current directory 
original_dir=$(pwd) 

# Change to the script's directory 
cd $(dirname $0)/../data

kaggle datasets download lyly99/logodet3k 
unzip logodet3k.zip
rm logodet3k.zip

# Change back to the original directory 
cd $original_dir


