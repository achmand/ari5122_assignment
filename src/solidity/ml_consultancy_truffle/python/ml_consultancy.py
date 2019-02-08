""" 
technologies:
    -Node.js: to install npm packages
        -> installation 
            curl -sL https://deb.nodesource.com/setup_11.x | sudo -E bash -
            sudo apt-get install -y nodejs 
    -Truffle: to compile smart contracts 
        NOTE: used truffle to compile instead of py-solc as it does not support solidity versions >= 0.5
        -> installation 
            npm install -g truffle
    -Ganache-cli: to deploy smart contracts in a development environment (command line interface)
        -> installation
            npm install -g ganache-cli
    -Web3.py: python library for interacting with Ethereum, wrapper of web3.js
        -> installation 
            pip install web3
    -OpenZepplin: ready available smart contracts and intefaces; used to inherit Ownable.sol
        -> src: https://github.com/OpenZeppelin/openzeppelin-solidity/blob/master/contracts/ownership/Ownable.sol 

steps: 
    1. Create directory and use terminal command "truffle init"
    2. It will create a new dApp project. Navigate to contracts and create contract. 
    3. Once the contract is finished use terminal to compile "truffle compile".
    4. Start ganache by opening a new terminal tab and inputting "ganache-cli"
    # NOTE: close terminal to close port, there are some issues when stopping ganache-cli 
    5. You can deploy contracts via truffle using "truffle migrate" but in our case we need to deploy from python.
    6. Follow the classes found in ml_consultancy.py
    7. These classes were utilised in the Jupyter Notebook Q5.

issues:
    Since the data is all publicly available you cant really keep secrets on the chain which goes 
    against what we are trying to build since a client needs to pay to access weights. If someone runs 
    a full node they can find ways to view the weights saved since everything is on each node. 
    Once solution to go around this is to encrypt the properties and upload the encrypted data on the 
    chain. After the client pays you may send the keys so that the client can decrypt the data once paid. 
    But this defeats the whole purpose since now the client need to trust a third party. 
    Uptil now there are now current solutions to store secrets on the EVM. 

    Solutions:
        Use Ripple's codius as a decentralized application platform as it allows you to store secrets (early stages).
        Note -> Note blockchain but DLT. 
            https://twitter.com/joelkatz/status/991375966231445504?lang=en
        Wait for further development. 
"""

###### importing dependencies ############################################################################
import json
import decimal
import numpy as np
from web3 import Web3
from datetime import datetime
from collections import OrderedDict

# NOTE: All function which end up with _chain interact with the blockchain in someway. 
###### service ###########################################################################################
class Service: 
    """ A base class for a service, this will be inherited by both the client and opertor. 
        Holds common functions for the learner and to interact with a contract.
    """
    
    # Constants 
    ABI = "abi"
    BYTECODE = "bytecode"
    CONTRACTADDR = "contractAddress"
    TXFROM = "from"
    TXVAL = "value"
    ETHER = "ether"
    
    def __init__(self, account_addr, endpoint_uri="", web3=None, contract_addr="", contract_abi=""):
        """ Constructor for Service.

            Args:
                account_addr (string or int): int; uses account index in the web3.eth.accounts, string; address passed.
                endpoint_uri (string): Required if web3 instance is not passed, URI to the RPC endpoint.
                web3 (Web3): Optional, A web3 instance can be passed. If not passed a new instance is created.
                contract_addr (string): Optional, The contract (MlConsultancy.sol) address to interact with. 
                contract_abi (string): Optional, The contract (MlConsultancy.sol) abi to interact with. 
        """
        
        # initialize model collection
        self.learners = OrderedDict()
        # if no instance of web3 is passed, initialize instance 
        if web3 == None:
            self._web3 = Web3(Web3.HTTPProvider(endpoint_uri=endpoint_uri))
        else: 
            self._web3 = web3

        # set address for this service 
        if isinstance(account_addr, str): # sets to address passed
            self._account_addr = account_addr
        if isinstance(account_addr, int): # sets to index passed in web3.eth.account
            self._account_addr = self._web3.eth.accounts[account_addr]
        else:
            self._account_addr = self._web3.eth.accounts[0] # bad type set to first index in web3.eth.account

        # set contract instance if values passed 
        if contract_addr != "" and contract_abi != "":
            self.set_contract(contract_addr, contract_abi)

    @property
    def account_addr(self):
        """ Getter for web3. 
        """
        return self._account_addr
    
    @property 
    def account_balance(self):
        """ Getter for account balance.
        """
        return self._web3.fromWei((self._web3.eth.getBalance(self._account_addr)), self.ETHER)

    @property
    def web3(self):
        """ Getter for web3. 
        """
        return self._web3
    
    @property
    def contract_addr(self):
        """ Getter for smart contract address (MlConsultancy.sol). 
        """
        return self._contract_addr
    
    @property
    def contract_abi(self):
        """ Getter for smart contract abi (MlConsultancy.sol). 
        """
        return self._contract_abi
    
    def add_learner(self, learner, learner_id=None):
        """ Adds a new learner/model to the dictionary.

            Args:
                learner_id (int): Optional, if not set takes count, the identifier for a specific learner.
                learner (tuple (dataset_name, model_name, sklearn model)): The learner instance with other details. Learner can be already fitted.

            Returns:
                int: The id for the learner saved in the collection.
        """
        
        # adds model to collection
        if learner_id == None:
            learner_id = self.get_count_local()

        self.learners[learner_id] = learner
        return learner_id
    
    def fit_learner(self, learner_id, x, y):
        """ Fits the model corresponding to the key found in the collection. 

            Args:
                learner_id (int): The key to get the learner which will be fitted.
                x (numpy array): The features training set to fit on. 
                y (numpy array): The outcome training set to fit on.
        """
        
        # fit the learner corresponding the the id passed
        # the dict will return a tuple and index 2 is the 
        # instance for the model 
        self.learners[learner_id][2].fit(x, y)

    def set_params_learner(self, learner_id, params):
        """ Change parameters set for a specific learners. 

            Args:
                learner_id (int): The learner id used to get the learner which will update params. 
                **params (kwargs): Parameters to be updated. 
        """
        self.learners[learner_id][2].set_params(**params)

    def get_params_learner(self, learner_id):
        """ Gets the params for a specific learner. 

            Args:
                learner_id (int): The learner id used to get the learner's params. 
        """
        self.learners[learner_id][2].get_params()

    def predict_learner(self, learner_id, x):
        """ Predict x using the model found in the collection with specified key.

            Args:
                learner_id (int): The model id to select the model you want to predict with. 
                x (numpy array): Set of instances you want to predict. 

            Returns:
                numpy array: Array with the predicted outcomes. 
        """
        
        self.learners[learner_id][2].intercept_ = np.array([0.26560617, 1.08542374, -1.21471458])
        self.learners[learner_id][2].classes_ = np.array([0,1,2])
        return self.learners[learner_id][2].predict(x)

    def set_contract(self, contract_addr, contract_abi):
        """ Instantiate and set an instance of the contract found in this service, 
            using the parameters passed.

            Args:
                contract_addr (string): The address for the contract. 
                contract_abi (string): The abi for the contract. 
        """

        # set contract instance 
        self.contract = self._web3.eth.contract(address=contract_addr,
                                                abi=contract_abi)
        
        # set contract address 
        self._contract_addr = contract_addr

        # set contract abi
        self._contract_abi = contract_abi

    def get_cost_chain(self, denomination="ether"):
        """ Get service cost to get weights from the chain.

            Returns:
                float: The cost for the service in ether. 
        """
        return self._web3.fromWei(self.contract.functions.getServiceCost().call(), denomination)

    def get_count_chain(self):
        """ Get the total number of models on the chain.

            Returns:
                int: The total number of models on the chain. 
        """
        return self.contract.functions.getModelCount().call()

    def get_count_local(self):
        """ Get the total number of models in local collections,

            Returns:
                int: The total number of models in local collection.
        """
        return len(self.learners.keys())

    def get_details_local(self):
        """ Returns the details of the model weights available locally.

            Returns:
                list: List of all the model details available (model_id, dataset, model).
        """
        model_list = []
        for key, value in self.learners.items():
            model_list.append((key, value[0], value[1]))

        return model_list


    def get_details_chain(self):
        """ Returns the details of the model weights available on the chain.

            Returns:
                list: List of all the model details available (model_id, dataset, model, outcomes, timestamp).
        """
        model_list = []
        model_count = self.get_count_chain()
        for i in range(model_count):
            model_id, dataset, model, outcomes, timestamp = self.contract.functions.getModelDetails(i).call()
            timestamp = datetime.utcfromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M:%S')
            model_list.append((model_id, dataset, model, outcomes, timestamp))

        return model_list
    
    def get_weights_local(self, learner_id):
        """ Returns the weights, intercept and outcome for a specific learner. 

            Args:
                learner_id (int): The learner id to get the properties from. 

            Returns:
                numpy array: Weights for the fitted learner.
                numpy array: Intercept for the fitted learner. 
                numpy array: Outcomes/Classes for the fitted learner. 
        """
        learner = self.learners[learner_id][2]
        return learner.coef_, learner.intercept_, learner.classes_

    def get_weights_chain(self, learner_id):
        """ Get model weights for the learner id passed. Address requesting the weights must pay for the service first.

            Args:
                learner_id (int): The id for the learner found on the chain. 
            
            Returns:    
                numpy array: Array with the corresponding weights.
        """

        # get weights from chain to the corresponding id 
        outcomes, n_decimals, n_dims, intercept, weights = self.contract.functions.getWeights(learner_id).call({self.TXFROM: self.account_addr})
        
        # transform the intercept back according to the n_decimals returned
        intercept = [round(x * (10**-n_decimals), n_decimals) for x in intercept]

        # transform the weights back according to the n_decimals returned
        weights = [round(x * (10**-n_decimals), n_decimals) for x in weights]
        
        # if outcome is not binary we need to reshape 
        total_outcomes = len(outcomes)
        if total_outcomes > 2:
            weights = np.reshape(weights, (total_outcomes, n_dims))

        return outcomes, intercept, weights

    def set_weights_chain(self, learner_id, chain_learner_id):
        """ Set the weights, intercept and outcome of a learner from data downloaded from the chain.
            The service requesting the data must have paid to access such data. 

            Args:
                learner_id (int): The local learner id to set the properties to. 
                chain_learner_id (int): The chain learner id to set the properties from. 
        """

        outcomes, intercept, weights = self.get_weights_chain(chain_learner_id)
        self.learners[learner_id][2].classes_ = outcomes
        self.learners[learner_id][2].intercept_ = intercept 
        self.learners[learner_id][2].coef_ = weights 

###### operator ##########################################################################################
class Operator(Service):
    """ Operator class inherits from Service. This class handles functions for the operator 
        such as; deploying/interacting with contract, adding model, add and update weights.
    """ 

    def deploy_contract(self, contract_path, wait=True):
        """ Deploys a smart contract to the endpoint/node.

            Args:
                contract_path (string): The src/path for the compiled smart contract (MlConsultancy.sol) to deploy.
        """

        # converts compiled contract to json object 
        with open(contract_path, 'r') as json_file:
            consultancy_contract = json.load(json_file)

        # instantiate and deploy smart contract 
        contract_deploy = self._web3.eth.contract(
            abi=consultancy_contract[self.ABI],
            bytecode=consultancy_contract[self.BYTECODE])
        
        # getting tx hash from the newly deployed contract 
        tx_hash = contract_deploy.constructor().transact(transaction={self.TXFROM: self._account_addr})
        
        # wait for transaction to be mined...
        if wait == True:
            self._web3.eth.waitForTransactionReceipt(tx_hash)

        # getting the tx receipt to get contract address 
        tx_receipt = self._web3.eth.getTransactionReceipt(tx_hash)

        # set the contract instance with the newly-deployed address
        self.set_contract(contract_addr=tx_receipt[self.CONTRACTADDR],
                          contract_abi=consultancy_contract[self.ABI])
    

    def __prep_for_upload(self, learner_id, n_decimals=8):
        """ Prepares the model to be uploaded on the chain 

            Args:
                learner_id (int): The learner id to get the learner which will be uploaded.
                n_decimals (int): Since EVM does not support floats we use this as a reference to the decimal points.

            Returns:
                string: Dataset name. 
                string: Model name. 
                numpy array: Array with possible classes. 
                int: Since EVM does not support floats we use this as a reference to the decimal points.
                int: Number of features
                numpy array: The intercept for the fitted model. 
                numpy array: The weights for the fitted model.  
        """

        # get learner to upload weights for
        learner = self.learners[learner_id]
        
        # get model details 
        dataset_name, model_name = [learner[0], learner[1]]
        
        # get properties for the fitted model 
        weights, intercept, outcomes = self.get_weights_local(learner_id)

        # convert outcome numpy array to list of ints
        outcomes = [int(x) for x in outcomes]

        # number of dimension 
        # if binary take the shape index 0 of the weights 
        n_dims = weights.shape[0]
        # if non binary take the shape index 1 of the weights since its 2D
        if len(outcomes) > 2:
            n_dims = weights.shape[1]

        # remove decimal places for intercepts
        intercept = np.round(intercept, n_decimals)
        intercept = [int(x * (10**n_decimals)) for x in intercept]
    
        # flatten weights array and convert float to int (removing decimal places) 
        weights = np.round(weights.flatten(), n_decimals) 
        weights = [int(x * (10**n_decimals)) for x in weights]
        
        return dataset_name, model_name, outcomes, n_decimals, n_dims, intercept, weights

    def upload_model_chain(self, learner_id, n_decimals=8):
        """ Upload the properties of a local model found in the collection 
            to the blockchain. 

            Args:
                learner_id (int): The local learner id to set the properties from. 
                chain_learner_id (int): The chain learner id to set the properties to.
                n_decimals (int): Since EVM does not support floats we use this as a reference to the decimal points.
        """

        # prepare for upload
        dataset_name, model_name, outcomes, n_decimals, n_dims, intercept, weights = self.__prep_for_upload(learner_id, n_decimals)
       
        # upload on chain 
        self.contract.functions.addModel(dataset_name, 
                                        model_name, 
                                        outcomes, 
                                        n_decimals,
                                        n_dims,
                                        intercept,
                                        weights).transact(transaction={self.TXFROM : self.account_addr})

    def update_model_chain(self, learner_id, chain_learner_id, n_decimals=8):
        """ Update the model found on the chain with the local version. 
             
            Args:
                n_decimals (int): Since EVM does not support floats we use this as a reference to the decimal points.
        """

        # prepare for update/upload
        _, _, _, _, _, intercept, weights = self.__prep_for_upload(learner_id, n_decimals)

        # update model found on chain 
        self.contract.functions.updateModel(chain_learner_id,
                                           intercept, 
                                           weights).transact(transaction={self.TXFROM : self.account_addr})

###### client ############################################################################################
class Client(Service): 
    """ Client class inherits from service. This class handles functions for the client
        such as; interacting with contract, view current models, get weights.
    """

    def pay_service_chain(self, learner_id):
        """ Pays service to get access to the weights saved on chain 

            Args:
                learner_id (int): The id of the model which the client is paying for. After payment can access anytime.
        """
        service_cost = self.get_cost_chain()
        self.contract.functions.payService(learner_id).transact({self.TXFROM: self.account_addr,
                                                                self.TXVAL: self._web3.toWei(service_cost, self.ETHER)})

##########################################################################################################
