###### importing dependencies ############################################################################
from ml_consultancy import Operator 
from ml_consultancy import Client 
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# load the iris dataset 
X_iris, y_iris = load_iris(return_X_y=True)

# split the data into 80% for training (Operator) and 20% for testing (Client)
X_operator, X_client, y_operator, y_client = train_test_split(X_iris, y_iris, test_size=0.20)

# create an 'Operator' instance which represents the ML consultancy agency 
# ganache end point
endpoint_uri = "http://127.0.0.1:8545" 

# path for the compiled contract
contract_json = "../build/contracts/MlConsultancy.json"

# instantiate Operator and set the account address to index 0 in the provided accounts 
# in the web3 instance 
operator = Operator(account_addr=0, endpoint_uri=endpoint_uri)

# deploy contract as an operator 
operator.deploy_contract(contract_json)

# add a local model to the operator
dataset_name = "Iris Dataset"
model_name = "Logisitic Regression"
o_local_model_id = operator.add_learner((dataset_name, model_name, LogisticRegression()))

# fit the local model which was added to the operator's local collection 
operator.fit_learner(o_local_model_id, X_operator, y_operator)

# upload the weights for the fitted model to the chain 
operator.upload_model_chain(o_local_model_id)

# create a new instance for the client 
client = Client(account_addr=1, 
                web3=operator.web3, 
                contract_addr=operator.contract_addr, 
                contract_abi=operator.contract_abi)

# check the available models found on the chain 
print("Available Models on the chain {}".format(client.get_details_chain()))

# knowing which models are available the client creates a local model 
# which utilises the same model which the client is willing to pay for 
c_local_model_id = client.add_learner((dataset_name, model_name, LogisticRegression()))

# client pays for service 
print("Client balance before paying for the service: {}".format(client.account_balance))
print("Operator balance before the client pays for the service: {}".format(operator.account_balance))
client.pay_service_chain(o_local_model_id)
print("Client balance after paying for the service: {}".format(client.account_balance))
print("Operator balance after the client pays for the service: {}".format(operator.account_balance))

# set the weights for the local model in the client with the values 
# send back from the chain (the model which the client paid for)
client.set_weights_chain(c_local_model_id, o_local_model_id)

# predict using the 20% data found on the client 
y_predicted = client.predict_learner(c_local_model_id, X_client)

# output the accuracy score 
print("Accuracy for the model on the client after updating from the chain: {}%".format(round(accuracy_score(y_client, y_predicted), 2) * 100))

# update the model parameters and re-fit 
operator.set_params_learner(o_local_model_id, {"penalty":"l1"})
operator.fit_learner(o_local_model_id, X_operator, y_operator)
operator.update_model_chain(o_local_model_id, o_local_model_id)

# reupdate the model weights from the chain 
client.set_weights_chain(c_local_model_id, o_local_model_id)

# predict using the 20% data found on the client with updated weights 
y_predicted = client.predict_learner(c_local_model_id, X_client)

# output the accuracy score (updated weights)
print("Accuracy for the model on the client after re-updating from the chain: {}%".format(round(accuracy_score(y_client, y_predicted), 2) * 100))
