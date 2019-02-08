/* 
    - Making something private only prevents other contracts from accessing and modifying the information
    - style/conventions: https://solidity.readthedocs.io/en/v0.5.3/style-guide.html?highlight=style
*/

pragma solidity ^0.5.0;

// Ownable.sol
import "installed_contracts/zeppelin/contracts/ownership/Ownable.sol";

contract MlConsultancy is Ownable {

    /* data about the machine learning model */
    struct Model {
        uint8 id;           // the id for this model 
        string dataset;     // dataset used
        string learner;     // model name
        uint8[] outcomes;   // outcomes/classes
        uint256 timestamp;  // timestamp when model was uploaded
    }

    /* weights for a specific model */
    struct ModelWeights {
        uint8 nDecimals;   // number of decimal places 
        uint8 nDims;       // total number of dimensions for features
        int32[] intercept; // intercept values
        int32[] weights;   // weight/coefficients values
    }

    /* the cost required to access weights */
    uint public SERVICE_COST = 0.01 ether;

    /* maximum number of different models an owner can have */
    uint8 constant maxModels = 255; 

    /* holds different models added by the owner */
    mapping(uint8 => Model) private models;

    /* holds weights for different models */
    mapping(uint8 => ModelWeights) private modelWeights;

    /* hold the rights for an address to view weights */
    mapping(address => mapping (uint8 => bool)) private clientAccess;

    /* holds count of the total models */
    uint8 private modelCount;

    /* this constructor is executed at initialization and sets the owner of the contract */
    constructor() public { 
        modelCount = 0; // set number of models to 0
    }

    /* get service cost */
    function getServiceCost() public view returns (uint) {
        return SERVICE_COST; 
    }

    /* returns the total number of models */
    function getModelCount() public view returns (uint8) {
        return modelCount;
    }
    
    /* returns model details for the id passed */
    function getModelDetails(uint8 modelId) public view returns (
        uint8 id, 
        string memory, 
        string memory, 
        uint8[] memory, 
        uint256
    ) 
    {
        Model memory modelAccess = models[modelId];
        return (
            modelAccess.id, 
            modelAccess.dataset, 
            modelAccess.learner, 
            modelAccess.outcomes, 
            modelAccess.timestamp
        );
    }

    /* adds a new model (can only be called by the owner) */
    function addModel(
        string memory dataset, 
        string memory learner, 
        uint8[] memory outcomes,
        uint8 nDecimals, 
        uint8 nDims, 
        int32[] memory intercept,
        int32[] memory weights
    ) 
        public 
        onlyOwner() 
    {
       
        // check if more models can be added 
        require(modelCount < maxModels, "Cannot add more models.");

        // add new model to dictionary 
        models[modelCount] = Model(modelCount, dataset, learner, outcomes, now);

        // add model weights to dictionary 
        modelWeights[modelCount] = ModelWeights(nDecimals, nDims, intercept, weights);

        // increment model count 
        modelCount += 1;
    }

    /* update model weights and intercept only accessible by owner  */
    function updateModel(
        uint8 modelId, 
        int32[] memory intercept, 
        int32[] memory weights
    ) 
        public 
        onlyOwner()
    {
        modelWeights[modelId].intercept = intercept;
        modelWeights[modelId].weights = weights;
    }

    /* pay for service since if you put a return with payable u just make a call */
    /* and get the weights for free */
    function payService(uint8 modelId) public payable {        
        require(msg.value == SERVICE_COST, "Amount sent is not equal to the service cost.");
        
        // send ammount to the owner 
        address payable ownerWallet = address(uint160(owner())); // cast to payable
        ownerWallet.transfer(address(this).balance); // send payment (throws on failure unlike send)
        
        // give access to weights to the client
        clientAccess[msg.sender][modelId] = true; 
    }

    /* return weights if sender has permission/paid for the service */
    function getWeights(uint8 modelId) public view returns (
        uint8[] memory, 
        uint8, 
        uint8, 
        int32[] memory, 
        int32[] memory
    ) 
    {
        require(clientAccess[msg.sender][modelId] == true, "Sender does not have rights to view properties");

        // get weights if user has permission to 
        ModelWeights memory weightsAccess = modelWeights[modelId];
        Model memory modelAccess = models[modelId];
        
        // return values 
        return (
            modelAccess.outcomes, 
            weightsAccess.nDecimals, 
            weightsAccess.nDims, 
            weightsAccess.intercept, 
            weightsAccess.weights
        );
    }
}