import torch
from Evaluation import evaluation
import sys as sy

# Train the global model using federated learning
def Model_CHAINFL(Train_Data, Train_Target, Test_Data, Test_Target):
    # Define the remote worker nodes (clients)
    hook = sy.TorchHook(torch)
    client1 = sy.VirtualWorker(hook, id="client1")
    client2 = sy.VirtualWorker(hook, id="client2")
    client3 = sy.VirtualWorker(hook, id="client3")
    clients = [client1, client2, client3]

    # Define the global model
    global_model = torch.nn.Linear(10, 1)

    for epoch in range(5):
        for client in clients:
            # Get data from the client
            Train_Data = client._objects[Train_Data.id]
            Train_Target = client._objects[Train_Target.id]

            # Train the global model on client's data
            global_model.train()
            optimizer = torch.optim.SGD(global_model.parameters(), lr=0.1)
            optimizer.zero_grad()
            output = global_model(Train_Data)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output.squeeze(), Train_Target.float())
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            y_pred = torch.sigmoid(global_model(Test_Data)).round()

        Eval = evaluation(y_pred, Test_Target)
        return Eval
