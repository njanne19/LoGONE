# Install Logodet dataset locally on this computer 
# Usage: logone/datasets/install_logodet.sh

# Save the current directory 
original_dir=$(pwd) 

# Change to the script's directory 
cd $(dirname $0)/../data

gdown --fuzzy https://drive.google.com/open\?id\=1p1BWofDJOKXqCtO0JPT5VyuIPOsuxOuj
tar -xvf openlogo.tar   
rm openlogo.tar

# Change back to the original directory 
cd $original_dir


