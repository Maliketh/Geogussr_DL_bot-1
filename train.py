import torch

def train_model(model, criterion, optimizer, scheduler,
                train_loader, val_loader, train_dataset, val_dataset,
                centroids, country_list, compute_geodesic_error, num_epochs, device):

    best_val_loss = float("inf")
    patience = 3  # stop if no improvement after 3 epochs
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_dataset)


        model.eval()
        val_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_dataset)
        geo_error = compute_geodesic_error(all_preds, all_labels, centroids, country_list)

        # Scheduler
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Avg Geodesic Error: {geo_error:.2f} km, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 🔑 Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("New best model saved.")
        else:
            counter += 1
            print(f" No improvement for {counter} epoch(s).")
            if counter >= patience:
                print("Early stopping triggered.")
                break
