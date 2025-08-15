# cloud.py
import flwr as fl
import torch
import csv
import matplotlib.pyplot as plt
import time
import os
from model_utils import get_model, get_loaders, evaluate_model

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# Global result tracking
results = []
start_time = time.time()

# Evaluation function used by the strategy
def evaluate_fn(server_round, parameters, config):
    model = get_model()
    state_dict = model.state_dict()
    for k, v in zip(state_dict.keys(), parameters):
        state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict)
    _, test_loader = get_loaders()
    acc, loss = evaluate_model(model, test_loader)
    results.append((server_round, loss, acc))
    print(f"[SERVER] Round {server_round} evaluation: acc={acc:.2f} loss={loss:.4f}")
    return loss, {"accuracy": acc}

# FedAvg strategy with custom evaluation
def get_strategy():
    return fl.server.strategy.FedAvg(
        evaluate_fn=evaluate_fn,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )

# Save results to CSV and plot
def save_and_plot_results():
    training_time = (time.time() - start_time) / 60  # minutes
    with open("results/federated_train_logs.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "Loss", "Accuracy"])
        writer.writerows(results)

    rounds = [r[0] for r in results]
    losses = [r[1] for r in results]
    accs = [r[2] for r in results]
    final_acc = accs[-1] if accs else 0.0
    
    with open("results/federated_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Final Accuracy", f"{final_acc:.2f}"])
        writer.writerow(["Training Time (min)", f"{training_time:.2f}"])

    # Plot Loss
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, losses, label="Loss", color='blue')
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Loss vs Round")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/federated_loss_plot.png")
    plt.close()
    
    # Plot Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, accs, label="Accuracy", color='orange')
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Round")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/federated_accuracy_plot.png")
    plt.close()

# Save final model and training summary
def save_final_model():
    model = get_model()
    torch.save(model.state_dict(), "results/models/resnet18_fed_cifar10.pth")


if __name__ == "__main__":
    strategy = get_strategy()
    port = int(os.environ.get("PORT", 8080))
    fl.server.start_server(
        server_address=f"0.0.0.0:{port}",
        config=fl.server.ServerConfig(num_rounds=50),
        strategy=strategy
    )
    save_and_plot_results()
    save_final_model()
    