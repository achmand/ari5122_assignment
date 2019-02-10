# NOTE: Make sure to run ganache or ganache-cli before running script, also make sure endpoint is correct

# importing dependencies 
import json 
import time
import pytz
import random 
import numpy as np
from web3 import Web3
from datetime import datetime
from datetime import timezone
from datetime import timedelta  

# function which prints a divider 
def divider():
    print("#####################################################################################################")

# specifiy end point 
endpoint_uri = "http://127.0.0.1:8545"

# specifiy path for the compiled contract so one can deploy contract 
compiled_contract = "../build/contracts/FootballCompetition.json"

# instantiate a new web3 instance 
web3 = Web3(Web3.HTTPProvider(endpoint_uri=endpoint_uri))

# set the address for the organizer 
organizer = web3.eth.accounts[0]
print("Organizer address is: {}".format(organizer))

# print divider 
divider()

# organizer will deploy the smart contract 
# firt open json and convert to object 
with open(compiled_contract, "r") as json_file:
    competition_contract = json.load(json_file) # load json
    
    # get contract abi and bytecode 
    contract_abi = competition_contract["abi"]
    contract_bytecode = competition_contract["bytecode"]
    
    # instantiate smart contract 
    contract_deploy = web3.eth.contract(abi=contract_abi,
                                        bytecode=contract_bytecode)
    
    # get tx hash from deployed contract 
    # deployed from organizer, so the organizer is also the owner of this contract
    tx_hash = contract_deploy.constructor().transact(transaction={"from":organizer})

    # wait for the tx to be mined 
    web3.eth.waitForTransactionReceipt(tx_hash)

    # getting the tx receipt 
    tx_receipt = web3.eth.getTransactionReceipt(tx_hash)

    # get deployed contract address 
    contract_addr = tx_receipt["contractAddress"]

    # get the instance of the contract to be able to interact with it
    contract = web3.eth.contract(address= contract_addr,
                                abi=contract_abi)


# print deployed contract address 
print("Contract deployed by: {0}, contract address is: {1}".format(organizer, contract_addr))

# print divider 
divider()

# create a list of participants (we assume that the organizer will be competing too)
participants = []
for i in range(len(web3.eth.accounts)):
    participant_addr = web3.eth.accounts[i]
    participants.append(participant_addr) # append account to list 
    print("Participant {0} has address {1}".format(i+1, participant_addr)) # print participants

# print divider 
divider()

# now we want to add some teams which can be used when creating a new competition 
# only the owner can add teams to the contract but a competition can be organized by anyone 
# the teams which are added can then be used by an organizer to created a competition with the specified teams 

# create synthetic teams
n_teams = 30
for i in range(n_teams):
    team_name = "Team_" + str(i+1)
    # sending from organizer since the organizer is the same address as the one who deployed the contract 
    contract.functions.addTeam(team_name).transact(transaction={"from": organizer})

# lets test that only the owner can add teams to the contract 
# using try and catch clause coz we know it will return an exception
try:
    contract.functions.addTeam("Team_Fail").transact(transaction={"from": participants[1]})
except:
    print("Failed to add a new team from someone who is not the owner of the contract.")

# now that we have teams saved on the chain lets 
# query this teams to make sure everything was written on the chain
# this time anyone can call this function so lets call it from 
# an address which is not the owner 

# first get the total number of teams which are stored in the contract 
total_teams = contract.functions.getTeamCount().call(transaction={"from": participants[1]})
print("Total number of teams which are available to be used in a competition: {}".format(total_teams))
transaction={"from": participants[1]}

# get team with id from the chain
def get_team(team_id, caller):
    return contract.functions.getTeam(team_id).call(transaction={"from": caller})

# now we know that the team ids are incremental as it is shown in the contract 
# lets get the id and name of each team which was added on the chain 
# again this function can be accessed by anyone 
teams_available = [] # holds reference of the ids for the teams available on the chain
print("Teams on the chain are:")
for i in range(total_teams):
    team_id, team_name = get_team(i+1,participants[2])
    teams_available.append(team_id)
    print("\tID: {0} Name: {1}".format(team_id, team_name))

# print divider 
divider()

# convert string to unix timestamp
def convert_unix(date):
    dt = datetime.strptime(date, "%d/%b/%Y %H:%M:%S")
    timestamp = dt.replace(tzinfo=timezone.utc).timestamp()
    return int(timestamp)

def add_competition(comp_name, comp_start, comp_end, comp_organizer):
    # we must convert these dates to unix time since solidity accepts unix timestamps 
    # convert dates to unix timestamps
    start_date_ts = convert_unix(comp_start)
    #print("Converted date {} to unix ts {}".format(start_date, start_date_ts))
    end_date_ts = convert_unix(comp_end)
    #print("Converted date {} to unix ts {}".format(end_date, end_date_ts))

    # let add a new competition on the chain 
    tx_hash = contract.functions.addCompetition(comp_name,
                                                start_date_ts,
                                                end_date_ts).transact(transaction={"from": comp_organizer})
    # wait for the tx to be mined 
    web3.eth.waitForTransactionReceipt(tx_hash)

    print("Address {} created a new competition".format(comp_organizer))

# now that teams are available, we can add a new competition 
# a competition can be created by anyone in our case we will use the address 
# we set for organizer which in this case also happens to be the owner of the contract 
# NOTE: Used March since I added a check which wont allow you to add competitions in the past !
competition_name = "Champions League"
# start_date = "01/Mar/2019 00:00:00"
# end_date = "28/Mar/2019 00:00:00"

# NOTE: We convert local time to UTC since to compare times on the chain UTC is used
local = pytz.timezone ("Europe/Malta")
start_date = (datetime.now() + timedelta(seconds=5)).astimezone(pytz.utc) 
end_date = (start_date + timedelta(seconds=5)).astimezone(pytz.utc)
start_date = start_date.strftime("%d/%b/%Y %H:%M:%S")
end_date = end_date.strftime("%d/%b/%Y %H:%M:%S") 

add_competition(comp_name=competition_name,
                comp_start=start_date,
                comp_end=end_date,
                comp_organizer=organizer)
                
# now that a competition is added lets check how many competitions are available on the chain 
# this function is public and can be accessed by anyone 
total_comps = contract.functions.getCompetitionCount().call(transaction={"from": participants[4]})
print("Total competitions available on the chain {}.".format(total_comps))

# for the sake of it lets create a new competition organized by another address 
competition_name_b = "Brazil League"
start_date_b = "01/Jul/2019 00:00:00"
end_date_b = "28/Jul/2019 00:00:00"
add_competition(comp_name=competition_name_b,
                comp_start=start_date_b,
                comp_end=end_date_b,
                comp_organizer=participants[4])

total_comps = contract.functions.getCompetitionCount().call(transaction={"from": participants[4]})
print("Total competitions available on the chain {}.".format(total_comps))

def convert_dt(unix):
    return datetime.utcfromtimestamp(int(unix)).strftime('%Y-%m-%d %H:%M:%S')

# get competition available on chain 
# this is accessible by anyone 
def get_comp(comp_id, caller):
    c_id, name, organizer, balance, s_ts, e_ts, announced, winner, teams, n_teams = contract.functions.getCompetition(comp_id).call(transaction={"from": caller}) 
    available_teams = np.array(teams)
    available_teams = available_teams[available_teams != 0]
   
    print("\n--------------------")
    print("Competition ID: {}".format(c_id) +
        "\nName: {}".format(name) + 
        "\nOrganizer: {}".format(organizer) +
        "\nPot: {} EU".format(balance)+ 
        "\nStart: {}".format(convert_dt(s_ts)) +
        "\nEnd: {}".format(convert_dt(e_ts)) +
        "\nWinner announced: {}".format(announced) +
        "\nWinner Team: {}".format(winner) +
        "\nAvailable Team IDs: {}".format(available_teams) + 
        "\nTotal Teams: {}".format(n_teams))

    print("--------------------\n")

# get all available competitions 
def get_comps(caller):
    total_comps = contract.functions.getCompetitionCount().call(transaction={"from":caller})
    for i in range(total_comps):
        get_comp(i + 1, caller) # i+1 since we know ids increment and start from 1 
   
# print available competitions found on the chain
divider()
print("Available competitions found on chain:")
get_comps(participants[5])
divider()

# now that we added a competition we need to add teams which will be available to bet on 
# this function can only be accessed by the one who created the competition 
competition_teams_n = 10 

# lets randomly select teams from the available teams found on the chain 
# seed = 42
# random.seed(seed)
selected_teams = random.sample(teams_available, k=competition_teams_n)

# let add these competing teams to a competition found on the chain
# only the one who created the competition can add teams to the competition
# teams cannot be edited and cannot add new teams once competition started  
def add_teams_comp(comp_id, teams, comp_organizer):
    for i in range(len(teams)):
        contract.functions.addTeamToCompetition(comp_id, 
                                                teams[i]).transact(transaction={"from": comp_organizer})

competition_id = 1 # we will add to the first competition that was created by the organizer address 
print("Adding teams to competition: {}".format(competition_id))
add_teams_comp(competition_id, selected_teams, organizer) # add teams to competition 1

# get teams available for a competition from the chain
# any address can call this function
def get_teams_comp(comp_id, caller):
    teams = contract.functions.getCompetition(comp_id).call(transaction={"from": caller})[8]
    available_teams = np.array(teams)
    available_teams = available_teams[available_teams != 0]

    print("Teams available for competition {}".format(comp_id))
    for i in range(len(available_teams)):
        team_id, team_name = get_team(int(available_teams[i]), caller)
        print("\tID: {0} Name: {1}".format(team_id, team_name))

# print available teams for competition from chain
print("Getting available teams from chain.")
get_teams_comp(competition_id, participants[6])

# as the organizer lets try and add an existing team to the competition 
# this function must fail 
try:
    contract.functions.addTeamToCompetition(competition_id,
                                            selected_teams[4]).transact(transaction={"from": organizer})
except:
    print("Cannot add an existing team.")

divider()

# now we have an existing competition with available teams to select from 
# let the participants join the competition 
for i in range(len(participants)):
    participant_addr = participants[i]
    random_team = random.choice(selected_teams)
    print("Participant: {0} with address: {1} bet on team {2} in competition {3}".format(i+1,
                                                                                        participant_addr,
                                                                                        random_team,
                                                                                        competition_id))
    # join competition
    contract.functions.joinCompetition(competition_id,
                                        random_team).transact(transaction={"from": participant_addr})

divider()
# print details of the competition now to check the pot 
print("Competition details after bets are added competition id: {}".format(competition_id))
get_comp(competition_id, participants[7])

# now let set the winner for the competition 
random_winner = random.choice(selected_teams)

# sleep for 15 so competition is over in this test 
time.sleep(15)
contract.functions.setWinningTeam(competition_id,
                                  random_winner).transact(transaction={"from": organizer})


# re-print competition details after winner is set 
print("Winner was set on the chain.")
get_comp(competition_id, participants[7])

# call the check winnings
for i in range(len(participants)):
    participant_addr = participants[i]
    is_winner, pot_balance, winnings, total_winners, selected_team = contract.functions.checkWinnings(competition_id).call(transaction={"from": participant_addr})
    print("Participant {0} with address {1}".format(i+1, participant_addr) +
        "\nIs winner {}".format(is_winner) +
        "\nPot balance {}".format(pot_balance) +
        "\nWinnings {}".format(winnings) +
        "\nTotal winners {}".format(total_winners) + 
        "\nTeam selected {}".format(selected_team))
            
    print("--------------------\n")
