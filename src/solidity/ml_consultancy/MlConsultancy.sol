pragma solidity ^0.5.0;

contract MlConsultancy {

    /* Define variable owner of the type address */
    address owner;

    /* This constructor is executed at initialization and sets the owner of the contract */
    constructor() public { owner = msg.sender; }

    function hi () public pure returns (string memory) {
        return "Hello World";
    }

}