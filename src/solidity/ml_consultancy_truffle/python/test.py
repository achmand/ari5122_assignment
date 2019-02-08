from ml_consultancy import Operator 
from ml_consultancy import Client 
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

provider = "http://127.0.0.1:8545"
contract_json = "../build/contracts/MlConsultancy.json"


operator = Operator(account_addr=0, endpoint_uri=provider)
operator.deploy_contract(contract_json)

client = Client(account_addr=1, 
                web3=operator.web3, 
                contract_addr=operator.contract_addr, 
                contract_abi=operator.contract_abi)

client2 = Client(account_addr=4, 
                web3=operator.web3, 
                contract_addr=operator.contract_addr, 
                contract_abi=operator.contract_abi)
# print(client._account_addr)
# print(operator._account_addr)
# print(operator.get_service_cost())
# print(operator.get_total_models())
# print(operator.get_models_detail())

dataset_name = "Iris Dataset"
lr_model_name = "Logisitic Regression"

X, y = load_iris(return_X_y=True)

new_learner = (dataset_name, lr_model_name, LogisticRegression())
learner_id = operator.add_learner(new_learner)
operator.fit_learner(learner_id, X, y)
print(operator.get_weights_local(learner_id))
operator.upload_model_chain(learner_id)

client.pay_service_chain(0)
client.add_learner((dataset_name, lr_model_name, LogisticRegression()))
client.set_weights_chain(0, 0)
print(client.get_weights_local(0))
#print(client2.get_weights_chain(0))

operator.set_params_learner(learner_id, {"penalty":"l1"})
operator.fit_learner(learner_id, X, y)
print(operator.get_weights_local(learner_id))
operator.update_model_chain(0, 0)

print(client.get_weights_local(0))
client.set_weights_chain(0, 0)
print(client.get_weights_local(0))

# print("Updated")
# print(client.get_weights_chain(0))


#print(client.get_weights_chain(0))
# print(operator.account_balance)
# client.pay_service_chain(0)
# print(client.account_balance)
# print(operator.account_balance)



# client_learner_id = client.add_learner((dataset_name, lr_model_name, LogisticRegression()))
# client.set_weights_chain(0,0)
# #print(client.get_weights_local(0))
# predicted_client = client.predict_learner(0, X)
# #print(predicted_client)
# print(client.get_details_local())
#print(accuracy_score(y, predicted_client))
# #print(operator.get_total_models())
# # print(X[0])
# # print(y)

# client.pay_service(0)
# # client.get_weights_chain(99)
# # client.get_weights_chain(0)

# #weights = client.get_weights_chain(0)

# new_client_learner = (dataset_name, lr_model_name, LogisticRegression())

# client_learner_id = client.add_learner(new_client_learner)

# print(client_learner_id)

# client.set_weights_chain(client_learner_id, client_learner_id)

# #print(client.get_weights_local(client_learner_id))

# predicted_client = client.predict_learner(0, X)

# print(predicted_client)