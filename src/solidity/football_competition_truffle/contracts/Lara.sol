pragma solidity >=0.4.22 <0.6.0;

contract TournamentContract {
    
    address organizer;
    uint balance;
    uint bettingEndDate;
    uint tournamentEndDate;
    uint winningTeam;
    
    struct Participant {
        bool betDone;
        uint winningAmount;
    }
    
    struct Bet {
        uint team;
        address participantId;
    }
    
    mapping(address => Participant) public participants;
    Bet[] public bets;
    
    constructor () public{
        balance = 0;
        organizer = msg.sender;
        bettingEndDate = 1548979200; 
        tournamentEndDate = 1551312000;
    }
    
    function enterTournament() public{
        // Participant storage sender = participants[msg.sender];
        
        // sender.betDone = false;
        // sender.winningAmount = 0;
    
        participants[msg.sender] = Participant(false, 0);
    }

    function bet(uint _predictedTeam) public {
        Participant storage sender = participants[msg.sender];
        
        if(now >= bettingEndDate) return;
        require(!sender.betDone, "You have already placed your bet.");
        
        require(_predictedTeam < 1 || _predictedTeam > 10, "Invalid team."); 
        
        bets.push(Bet({
            team: _predictedTeam,
            participantId: msg.sender
        }));
        
        sender.betDone = true;
        balance += 10;
    }
    
    function setWinningTeam(uint _winningTeam) public{
        require(
            msg.sender == organizer,
            "Only the organizer can give betting pass."
        );
        
        if (now <= tournamentEndDate) return;
        
        winningTeam = _winningTeam;
        
        setWinningAmount();
    }

    function getWinningAmount() public view returns (uint winnerValue_) {
        winnerValue_ = participants[msg.sender].winningAmount;
    }
    
    function setWinningAmount() public{
        uint numberOfWinners_ = 0;
        
        for (uint8 b = 0; b < bets.length; b++)
            if (bets[b].team == winningTeam) {
                numberOfWinners_ += 1;
            }
            
        for (uint8 b = 0; b < bets.length; b++)
            if (bets[b].team == winningTeam) {
                Participant storage winner = participants[bets[b].participantId];
                winner.winningAmount = balance/numberOfWinners_;
            }
    }
}