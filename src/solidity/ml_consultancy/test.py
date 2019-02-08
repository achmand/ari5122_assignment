# Installing technologies 
# Step 1: install solidity compiler on Ubuntu 
    # sudo add-apt-repository ppa:ethereum/ethereum
    # sudo apt-get update
    # sudo apt-get install solc

# importing dependencies 
from solc import compile_files 

contracts = compile_files('MlConsultancy.sol')
#consultancy_contract = contracts.pop("MlConsultancy.sol:mlConsultancy")