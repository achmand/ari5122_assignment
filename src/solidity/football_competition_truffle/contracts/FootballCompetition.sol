pragma solidity ^0.5.0;

// Ownable.sol
import "installed_contracts/zeppelin/contracts/ownership/Ownable.sol";

/* Contact used for football competitions */
contract FootballCompetition is Ownable {

    /* football competition */
    /* for the purposes of the problem only 255 competitions can be added to this contract */
    struct Competition {
        uint8 id;                   // competition id
        string name;                // name for competition
        address organizer;          // the address organizing the competition
        bool created;               // is competition created
        uint potBalance;            // total balance in competition
        uint256 startTimestamp;     // competition start timestamp
        uint256 winnerTimestamp;    // competition winner announcement timestamp
        bool winnerAnnounced;       // is the winner announced 
        uint8 winningTeam;          // the winning team

        uint8[] teamIds;                               // holds a list of team ids
        uint8 totalTeams;                              // total number of teams
        mapping(uint8 => bool) teams;                  // teams available in competition
        mapping(uint8 => uint) betsOnTeam;             // total bets on a team
        mapping(address => Participant) participants;  // the participants in the competition
    }
    
    /* struct for team */
    struct Team {
        uint8 id;       // team id 
        string name;    // team name
        bool nameSet;   // is team name set
    }
    
    /* struct for participant */
    struct Participant {
        uint8 teamId;       // team betting on
        bool isCompeting;   // is participant competing
    }

    /* holds count of the total competitions */
    uint8 private competitionCount;
   
    /* the maximum competitions which can be added */
    uint8 constant maxCompetitions = 255;

    /* the maximum football teams which can be added */
    uint8 constant maxFootballTeams = 255;
    
    /* holds count of the total teams */
    uint8 private teamCount;

    /* holds football teams which can be found in any competition */
    /* for the purposes of the problem only 255 teams can be added to this contract */
    mapping(uint8 => Team) private footballTeams;

    /* holds a mapping of available competitions */
    /* for the purposes of the problem only 255 competitions can be added to this contract */
    mapping(uint8 => Competition) private competitions;

    /* this constructor is executed at initialization and sets the owner of the contract */
    constructor() public { 
        competitionCount = 0; // set number of competitions to 0
        teamCount = 0;        // set number of teams to 0
    }

    /**
    * @dev Throws if called by any account other than the owner.
    */
    modifier onlyOrganizer(uint8 competitionId) {
        require(isOrganizer(competitionId), "Not Organizer.");
        _;
    }

    /**
    * @return true if `msg.sender` is the organizer of the competition.
    */
    function isOrganizer(uint8 competitionId) public view returns (bool) {
        return msg.sender == competitions[competitionId].organizer;
    }
    
    /**
    * @dev Throws if competition does not exist.
    */
    modifier competitionExist(uint8 competitionId) {
        require(competitions[competitionId].created == true, "Competition does not exist.");
        _;
    }
    
    /**
    * @dev Throws if team does not exist.
    */
    modifier teamExist(uint8 teamId) {
        require(footballTeams[teamId].nameSet == true, "Team does not exist.");
        _;
    }

    /**
    * @dev Throws if called competition already started.
    */
    modifier competitionStarted(uint8 competitionId) {
        require(now < competitions[competitionId].startTimestamp, "Competition already started.");
        _;
    }

    /**
    * @dev Throws if team is not in competition.
    */
    modifier teamInCompetition(uint8 competitionId, uint8 teamId) {
        require(competitions[competitionId].teams[teamId] == true, "Team is not available in competition");
        _;
    }
    
     /* returns the total number of teams */
    function getTeamCount() public view returns (uint8) {
        return teamCount;
    }

    /* returns the total number of competitions */
    function getCompetitionCount() public view returns (uint8) {
        return competitionCount;
    }

    /* add a new team to this contract (only the one who deployed contract can add team) */
    /* these teams can be used in a specific competition */
    /* this insures transparency as it cant be edited and the participant  */
    /* is insured that team with a specific id is the team with a specific name */
    function addTeam(string memory name) public onlyOwner() {

        // check if more teams can be added to this contract
        require(teamCount < maxFootballTeams, "Cannot add more teams.");
        
        // increment before as we dont want id 0 to be a team
        teamCount += 1;
        
        // adds a new team 
        footballTeams[teamCount] = Team(teamCount, name, true);
    }

    // get team name for the specified id 
    function getTeam(uint8 id) public view teamExist(id) returns(
        uint8, 
        string 
        memory
    ) 
    {
        // return team information
        return (footballTeams[id].id, footballTeams[id].name);
    }

    /* add a new competition (anyone can start a competition) */
    function addCompetition(
        string memory name,     // name of competition
        uint256 startTimestamp, // competition starting date
        uint256 winnerTimestamp // competition winner announcement date
    ) 
        public 
    {
        
        // check if more competitions can be added to this contract
        require(competitionCount < maxCompetitions, "Cannot add more competitions.");
        
        // check dates 
        require(now <= startTimestamp, "Invalid start date.");
        require(startTimestamp < winnerTimestamp, "Invalid winner date.");

        // increment before as we dont want id 0 to be a competition
        competitionCount += 1;

        // set values for new competition
        Competition memory newCompetition;
        newCompetition.id = competitionCount;
        newCompetition.name = name; 
        newCompetition.organizer = msg.sender;
        newCompetition.created = true;
        newCompetition.potBalance = 0;
        newCompetition.startTimestamp = startTimestamp;
        newCompetition.winnerTimestamp = winnerTimestamp;
        newCompetition.winnerAnnounced = false;
        newCompetition.teamIds = new uint8[](255);
        newCompetition.totalTeams = 0;

        // add competition
        competitions[competitionCount] = newCompetition;
    }

    /* return details about a competition */
    function getCompetition(uint8 id) public view competitionExist(id) returns (
        uint8,          // competition id 
        string memory,  // competition name
        address,        // organizer 
        uint,           // pot balance
        uint256,        // start ts
        uint256,        // end timestamp
        bool,           // winner announced 
        uint8,          // winning team
        uint8[] memory, // team ids
        uint8           // total teams
    ) 
    {
        Competition memory tmpCompetition = competitions[id];
        return(
            tmpCompetition.id,
            tmpCompetition.name,
            tmpCompetition.organizer,
            tmpCompetition.potBalance,
            tmpCompetition.startTimestamp,
            tmpCompetition.winnerTimestamp,
            tmpCompetition.winnerAnnounced,
            tmpCompetition.winningTeam,
            tmpCompetition.teamIds,
            tmpCompetition.totalTeams
        );
    }

    /* teams to be added in a competition must be entered one by one */
    /* this is because of the problems of evm and unbounded loops */
    /* only accessible by organizer */
    function addTeamToCompetition(
        uint8 competitionId,    // the competition id to add new team
        uint8 teamId            // the team which will be added to the competition
    )
        public 
        onlyOrganizer(competitionId)        // only accessible by organizer
        teamExist(teamId)                   // check if team exist
        competitionExist(competitionId)     // check if competition exist
        competitionStarted(competitionId)   // check if competition started
    {   
        // check if team exists
        require(footballTeams[teamId].nameSet == true, "Team does not exist");

        // check if team is already in competition (cannot override teams!)
        require(competitions[competitionId].teams[teamId] == false, "Team already in competition.");

        // add team to competition
        competitions[competitionId].teams[teamId] = true;

        // increment total number of teams in competition and add team to competition
        competitions[competitionId].teamIds[competitions[competitionId].totalTeams] = teamId;
        competitions[competitionId].totalTeams += 1;
    }

    /* allows participants to join a specific competition */
    function joinCompetition(
        uint8 competitionId,    // competion id which the address will be joining
        uint8 teamId            // the team id which the address is betting on
    ) 
        public 
        competitionExist(competitionId)             // check if competition exist
        competitionStarted(competitionId)           // check if competition started
        teamInCompetition(competitionId, teamId)    // check if team is available in competition
    {
        // check if the one joining is already in competition (one address one bet) 
        require(competitions[competitionId].participants[msg.sender].isCompeting == false, "Already in competition.");

        // set new balance for pot
        competitions[competitionId].potBalance += 10;

        // set team for participant 
        competitions[competitionId].participants[msg.sender].isCompeting = true;
        competitions[competitionId].participants[msg.sender].teamId = teamId;

        // increment the number of bets on that team
        competitions[competitionId].betsOnTeam[teamId] += 1;
    }

    /* set winner for a specific competition only accessible by organizer */
    function setWinningTeam(
        uint8 competitionId,    // the competition id to set the winner for
        uint8 teamId            // the winning team for the competition
    )
        public
        onlyOrganizer(competitionId)                // only accessed by organizer
        competitionExist(competitionId)             // check if competition exist
        teamInCompetition(competitionId, teamId)    // check if team is available in competition
    {
        // cannot override winner check if winner was already announced 
        require(competitions[competitionId].winnerAnnounced == false, "Winner is already set.");

        // can set winner if competition is over 
        require(now >= competitions[competitionId].winnerTimestamp, "Competition not finished yet.");

        // set winning team 
        competitions[competitionId].winnerAnnounced = true;
        competitions[competitionId].winningTeam = teamId;
    }

    /* check winnings for a participant */
    function checkWinnings(uint8 competitionId) public view competitionExist(competitionId) returns(
        bool, 
        uint, 
        uint, 
        uint, 
        uint8
    )
    {   
        
        // check if participant was actually competing in competition
        require(competitions[competitionId].participants[msg.sender].isCompeting == true, "Address was not in competition.");

        // check that the winner was announced 
        require(competitions[competitionId].winnerAnnounced == true, "Winning team not set yet.");

        // get competition
        Competition storage tmpCompetition = competitions[competitionId];
        uint8 winningTeam = tmpCompetition.winningTeam;
        uint8 selectedTeam = tmpCompetition.participants[msg.sender].teamId;
        bool isWinner = selectedTeam == winningTeam;
        uint potBalance = tmpCompetition.potBalance;
        uint totalWinners = tmpCompetition.betsOnTeam[winningTeam];

        // calculate winnings for the one requesting 
        uint winnings = 0; 
        if(isWinner == true){
            if(totalWinners > 1){
                winnings = potBalance / totalWinners;
            }
            else{
                winnings = potBalance;
            }
        }

        // return values
        return(
            isWinner,       // is the address a winner
            potBalance,     // total in pot
            winnings,       // winnings by address
            totalWinners,   // total winners
            selectedTeam    // the selected team
        );
    }
}