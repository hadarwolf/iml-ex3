import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from models import Ridge_Regression, Logistic_Regression
from helpers import plot_decision_boundaries, read_data_demo
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def load_data(filename):
    df = pd.read_csv(filename)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


def run_ridge_regression():
    #load the data
    print("Loading data...")
    X_train, y_train = load_data('C:\\Users\\hadar\\IML\\EX3\\train.csv')
    X_val, y_val = load_data('C:\\Users\\hadar\\IML\\EX3\\validation.csv')
    X_test, y_test = load_data('C:\\Users\\hadar\\IML\\EX3\\test.csv')

    #setup the experiment
    lambdas = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    train_accs = []
    val_accs = []
    test_accs = []
    best_val_acc = -1
    best_model = None
    best_lambda = -1
    worst_val_acc = 2.0
    worst_model = None
    worst_lambda = -1

    #train the model for each lambda
    for lam in lambdas:
        print(f"Training with Lambda: {lam}")
        model = Ridge_Regression(lambd=lam)
        model.fit(X_train, y_train)
        #predict the labels
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        #calculate the accuracy
        train_acc = np.mean(y_train_pred == y_train)
        val_acc = np.mean(y_val_pred == y_val)
        test_acc = np.mean(y_test_pred == y_test)

        #update the best and worst models
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            best_lambda = lam
        if val_acc < worst_val_acc:
            worst_val_acc = val_acc
            worst_model = model
            worst_lambda = lam
    
    print(f"\nBest Lambda: {best_lambda} (Val Acc: {best_val_acc:.4f})")
    print(f"Test Acc of Best Model: {test_accs[lambdas.index(best_lambda)]:.4f}")

    #plot the results
    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, train_accs, label='Train Accuracy')
    plt.plot(lambdas, val_accs, label='Validation Accuracy')
    plt.plot(lambdas, test_accs, label='Test Accuracy')
    plt.xlabel('Lambda')
    plt.ylabel('Accuracy')
    plt.title('Ridge Regression Results')
    plt.legend()
    plt.grid(True)
    plt.show()


    # Plot 2: Decision Boundaries
    print("Plotting Decision Boundaries...")
    plot_decision_boundaries(best_model, X_train, y_train, title='Best Model Decision Boundaries')
    plot_decision_boundaries(worst_model, X_train, y_train, title='Worst Model Decision Boundaries')
    
    #return best_model, best_lambda, worst_model, worst_lambda, train_accs, val_accs, test_accs

import numpy as np
import matplotlib.pyplot as plt

def run_gradient_descent():
    # 1. Initialize
    x = 0.0
    y = 0.0
    learning_rate = 0.1
    iterations = 1000

    # We need to store the history to plot the path later
    x_history = []
    y_history = []

    # 2. The Loop
    for i in range(iterations):
        # Store current position
        x_history.append(x)
        y_history.append(y)
        
        # Calculate Gradients (Derivatives)
        # d/dx of (x-3)^2  -> 2*(x-3)
        grad_x = 2 * (x - 3)
        
        # d/dy of (y-5)^2  -> 2*(y-5)
        grad_y = 2 * (y - 5)
        
        # Update Rule: New = Old - (Learning Rate * Gradient)
        x = x - (learning_rate * grad_x)
        y = y - (learning_rate * grad_y)

    # 3. Print the Final Result
    print(f"Final Position: ({x:.5f}, {y:.5f})")

    # 4. Plotting
    plt.figure(figsize=(8, 6))
    
    # Scatter plot: x vs y, colored by time (0 to 1000)
    plt.scatter(x_history, y_history, c=range(len(x_history)), cmap='viridis', s=10)
    
    # Mark the start and end
    plt.plot(x_history, y_history, color='gray', alpha=0.3) # Connect the dots
    plt.title("Gradient Descent Path")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="Iteration (Time)")
    plt.grid(True)
    plt.show()


def evaluate(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    total_loss = 0
    total_count = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
            total_count += labels.size(0)
    
    return correct / total_count, total_loss / len(dataloader)


def run_logistic_regression():
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    #load the data
    print("Loading data...")
    # A. Load the raw numpy arrays using your function
    X_train, y_train = load_data('C:\\Users\\hadar\\IML\\EX3\\train.csv')         # Adjusted paths to be relative
    X_val, y_val     = load_data('C:\\Users\\hadar\\IML\\EX3\\validation.csv')
    X_test, y_test   = load_data('C:\\Users\\hadar\\IML\\EX3\\test.csv')

    # B. Convert to PyTorch Tensors
    # Features (X) must be Floats. Labels (y) must be Longs (Integers).
    train_x_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_y_tensor = torch.tensor(y_train, dtype=torch.long)
    
    val_x_tensor = torch.tensor(X_val, dtype=torch.float32)
    val_y_tensor = torch.tensor(y_val, dtype=torch.long)
    
    test_x_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_y_tensor = torch.tensor(y_test, dtype=torch.long)

    # C. Wrap in Datasets
    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_y_tensor)
    val_dataset   = torch.utils.data.TensorDataset(val_x_tensor, val_y_tensor)
    test_dataset  = torch.utils.data.TensorDataset(test_x_tensor, test_y_tensor)

    # D. Create DataLoaders (The Managers)
    # This enables the "for inputs, labels in train_loader" loop to work
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    #setup the experiment
    learning_rates= [0.1, 0.01,0.01]
    history = {}
    criterion = nn.CrossEntropyLoss()
    for lr in learning_rates:
        # Initialize list for this LR
        history[lr] = {'train_acc' : [], 'val_acc' : [], 'test_acc' : [], 
                        'train_loss' : [], 'val_loss' : [], 'test_loss' : []}


    for learning_rate in learning_rates:
        print(f"Training with Learning Rate: {learning_rate}")
        model = Logistic_Regression(input_dim=2, output_dim=2)
        print(list(model.parameters()))
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
        for epoch in range(10):
            model.train()
            for inputs, labels in train_loader:
               # A. Move data to GPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # B. Erase old scratchpad notes (gradients)
                optimizer.zero_grad()

                # C. Take a guess (Forward Pass)
                # The model outputs raw scores (logits)
                outputs = model(inputs)

                # D. Grade the guess (Calculate Loss)
                # criterion is likely nn.CrossEntropyLoss() defined earlier
                loss = criterion(outputs, labels)

                # E. Find out who is to blame (Backward Pass)
                loss.backward()

                # F. Fix the brain cells (Update Weights)
                optimizer.step()


            train_acc, train_loss = evaluate(model, train_loader, criterion, device)
            val_acc, val_loss = evaluate(model, val_loader, criterion, device)
            test_acc, test_loss = evaluate(model, test_loader, criterion, device)

            #record scores
            history[learning_rate]['train_acc'].append(train_acc)
            history[learning_rate]['val_acc'].append(val_acc)
            history[learning_rate]['test_acc'].append(test_acc)
            history[learning_rate]['train_loss'].append(train_loss)
            history[learning_rate]['val_loss'].append(val_loss)
            history[learning_rate]['test_loss'].append(test_loss)

            print(f"Epoch {epoch+1} : Val Acc {val_acc}")

    best_lr = max(learning_rates, key=lambda lr: history[lr]['val_acc'][-1])
    print(f"\nWINNER: Best Learning Rate is {best_lr}")

    # Plotting code
    epochs_range = range(1, 11)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history[best_lr]['train_loss'], label='Training Loss', marker='o')
    plt.plot(epochs_range, history[best_lr]['val_loss'], label='Validation Loss', marker='o')
    plt.plot(epochs_range, history[best_lr]['test_loss'], label='Test Loss', marker='o')
    plt.title(f'Losses for Best Model (LR={best_lr})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 1. Re-build the Winning Model
    print(f"Retraining the best model ({best_lr}) for plotting...")
    best_model = Logistic_Regression(input_dim=2, output_dim=2)
    best_model.to(device)
    optimizer = torch.optim.SGD(best_model.parameters(), lr=best_lr)
    
    # 2. Quick Train (Same 10 epochs)
    for epoch in range(11):
        best_model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = best_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # 3. NOW Plot (Use X_test/y_test which are the numpy arrays from the top)
    plot_decision_boundaries(best_model, X_test, y_test, title=f'Best Model (LR={best_lr}) Predictions')


def multi_class_logistic_regression():
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    #load the data
    print("Loading data...")
    # A. Load the raw numpy arrays using your function
    X_train, y_train = load_data('C:\\Users\\hadar\\IML\\EX3\\train_multiclass.csv')         # Adjusted paths to be relative
    X_val, y_val     = load_data('C:\\Users\\hadar\\IML\\EX3\\validation_multiclass.csv')
    X_test, y_test   = load_data('C:\\Users\\hadar\\IML\\EX3\\test_multiclass.csv')

    # B. Convert to PyTorch Tensors
    # Features (X) must be Floats. Labels (y) must be Longs (Integers).
    train_x_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_y_tensor = torch.tensor(y_train, dtype=torch.long)
    
    val_x_tensor = torch.tensor(X_val, dtype=torch.float32)
    val_y_tensor = torch.tensor(y_val, dtype=torch.long)
    
    test_x_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_y_tensor = torch.tensor(y_test, dtype=torch.long)

    # C. Wrap in Datasets
    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_y_tensor)
    val_dataset   = torch.utils.data.TensorDataset(val_x_tensor, val_y_tensor)
    test_dataset  = torch.utils.data.TensorDataset(test_x_tensor, test_y_tensor)

    # D. Create DataLoaders (The Managers)
    # This enables the "for inputs, labels in train_loader" loop to work
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    #setup the experiment
    learning_rates= [0.01, 0.001,0.0003]
    history = {}
    criterion = nn.CrossEntropyLoss()
    num_classes = len(np.unique(y_train))
    for lr in learning_rates:
        history[lr] = {'train_acc' : [], 'val_acc' : [], 'test_acc' : [], 
                        'train_loss' : [], 'val_loss' : [], 'test_loss' : []}


    for learning_rate in learning_rates:
        print(f"Training with Learning Rate: {learning_rate}")
        model = Logistic_Regression(input_dim=2, output_dim=num_classes)
        # print(list(model.parameters()))
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        # "Decay LR by 0.3 every 5 epochs"
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
        
        for epoch in range(30):
            model.train()
            for inputs, labels in train_loader:
               # A. Move data to GPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # B. Erase old scratchpad notes (gradients)
                optimizer.zero_grad()

                # C. Take a guess (Forward Pass)
                # The model outputs raw scores (logits)
                outputs = model(inputs)

                # D. Grade the guess (Calculate Loss)
                # criterion is likely nn.CrossEntropyLoss() defined earlier
                loss = criterion(outputs, labels)

                # E. Find out who is to blame (Backward Pass)
                loss.backward()

                # F. Fix the brain cells (Update Weights)
                optimizer.step()
            
            #update the learning rate
            scheduler.step()

            train_acc, train_loss = evaluate(model, train_loader, criterion, device)
            val_acc, val_loss = evaluate(model, val_loader, criterion, device)
            test_acc, test_loss = evaluate(model, test_loader, criterion, device)

            #record scores
            history[learning_rate]['train_acc'].append(train_acc)
            history[learning_rate]['val_acc'].append(val_acc)
            history[learning_rate]['test_acc'].append(test_acc)
            history[learning_rate]['train_loss'].append(train_loss)
            history[learning_rate]['val_loss'].append(val_loss)
            history[learning_rate]['test_loss'].append(test_loss)

            # print(f"Epoch {epoch+1} : Val Acc {val_acc}")
            # # Print every 5 epochs to reduce clutter
            # if (epoch + 1) % 5 == 0:
            #     print(f"Epoch {epoch+1}/30 | Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")


    # Prepare data for the "Accuracy vs LR" plot
    final_val_accs = []
    final_test_accs = []
    str_lrs = [str(lr) for lr in learning_rates] # Convert to string for x-axis labels 

    print("\n--- RESULTS ---")
    for lr in learning_rates:
        # Get the accuracy from the LAST epoch
        v_acc = history[lr]['val_acc'][-1]
        t_acc = history[lr]['test_acc'][-1]
        
        final_val_accs.append(v_acc)
        final_test_accs.append(t_acc)
        print(f"LR {lr}: Val Acc: {v_acc:.4f}, Test Acc: {t_acc:.4f}")

    # Pick the Best Model (based on Validation)
    best_idx = np.argmax(final_val_accs)
    best_lr = learning_rates[best_idx]
    best_test_acc = final_test_accs[best_idx]
    
    print(f"\nWINNER: Best LR is {best_lr}")
    print(f"Test Accuracy of the Best Model: {best_test_acc:.4f}")

    # *** PLOT: Accuracy vs Learning Rate ***
    plt.figure(figsize=(8, 5))
    
    # Plot dots connected by lines
    plt.plot(str_lrs, final_val_accs, marker='o', label='Validation Accuracy')
    plt.plot(str_lrs, final_test_accs, marker='o', label='Test Accuracy')
    
    plt.title('Accuracy vs. Learning Rate')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plotting code
    epochs_range = range(1, 31)

    # Plot A: LOSSES over Epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history[best_lr]['train_loss'], label='Training Loss', marker='o')
    plt.plot(epochs_range, history[best_lr]['val_loss'], label='Validation Loss', marker='o')
    plt.plot(epochs_range, history[best_lr]['test_loss'], label='Test Loss', marker='o')
    plt.title(f'Losses for Best Model (LR={best_lr})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot B: ACCURACIES over Epochs
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history[best_lr]['train_acc'], label='Training Acc')
    plt.plot(epochs_range, history[best_lr]['val_acc'], label='Validation Acc')
    plt.plot(epochs_range, history[best_lr]['test_acc'], label='Test Acc')
    
    plt.title(f'Accuracies over Epochs (Best LR: {best_lr})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def run_decision_tree():

    #load the data
    print("Loading data...")
    X_train, y_train = load_data('C:\\Users\\hadar\\IML\\EX3\\train.csv')
    X_val, y_val = load_data('C:\\Users\\hadar\\IML\\EX3\\validation.csv')
    X_test, y_test = load_data('C:\\Users\\hadar\\IML\\EX3\\test.csv')

    dt_2 = DecisionTreeClassifier(max_depth=2, random_state=42)
    dt_2.fit(X_train, y_train)

    acc_test_2 = accuracy_score(y_test, dt_2.predict(X_test))
    print(f"Accuracy of the model with max depth 2 on the test set: {acc_test_2:.4f}")
    plot_decision_boundaries(dt_2, X_test, y_test, title='Decision Tree (Depth=2)')

    dt_10 = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt_10.fit(X_train, y_train)

    acc_test_10 = accuracy_score(y_test, dt_10.predict(X_test))
    print(f"Accuracy of the model with max depth 10 on the test set: {acc_test_10:.4f}")
    plot_decision_boundaries(dt_10, X_test, y_test, title='Decision Tree (Depth=10)')
    

if __name__ == "__main__":
    run_ridge_regression()
    run_gradient_descent()
    run_logistic_regression()
    multi_class_logistic_regression()
    run_decision_tree()