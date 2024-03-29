import torch
import torch.nn as nn

# ref https://blog.paperspace.com/writing-cnns-from-scratch-in-pytorch/
def trainmodel(dataloader, evalloader, model, epochs, criterion = nn.CrossEntropyLoss(), optimizer = None):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for i, (digits, labels) in enumerate(dataloader):
            digits = digits.to(device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            outputs = model(digits)
            loss = criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, total_loss/len(dataloader)))

        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            for i, (digits, labels) in enumerate(evalloader):
                digits = digits.to(device)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(device)
                outputs = model(digits)
                loss = criterion(outputs,labels)
                total += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

            print('Epoch [{}/{}], Eval: {:.4f} %'.format(epoch+1, epochs, correct/total * 100))

