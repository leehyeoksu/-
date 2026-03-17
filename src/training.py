import torch
from tqdm.auto import tqdm
import mlflow
import mlflow.pytorch
from .evaluating import evaluate
import os
import gc
def train(
    model,
    dataloader,
    test_loader,
    epochs,
    criterion,
    scaler,
    device,
    optimizer,
    experiment_name,
    path
):
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        model = model.to(device)
       
        min_delta = 0.001
        patience = 10

        best_acc = 0.0
        counter = 0

        # 파라미터 로깅
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", dataloader.batch_size)
        mlflow.log_param("optimizer", optimizer.__class__.__name__)
        mlflow.log_param("lr", optimizer.param_groups[0]['lr'])
        mlflow.log_param("patience", patience)
        mlflow.log_param("min_delta", min_delta)

        for i in tqdm(range(epochs)):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            # =====================
            # Train
            # =====================
            for x, y in dataloader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                with torch.autocast('cuda', dtype=torch.float16): # loss.backward() opimizer.step()
                    out = model(x)
                    loss = criterion(out, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                bn = x.size(0)
                total += bn
                total_loss += loss.item() * bn
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()

            train_loss = total_loss / total
            train_acc = correct / total

            # =====================
            # Test
            # =====================
            test_loss, test_acc = evaluate(
                model, test_loader, criterion, device
            )

            # =====================
            # MLflow Logging
            # =====================
            mlflow.log_metric("train_loss", train_loss, step=i)
            mlflow.log_metric("train_accuracy", train_acc, step=i)
            mlflow.log_metric("test_loss", test_loss, step=i)
            mlflow.log_metric("test_accuracy", test_acc, step=i)

            # 오타 수정 완료: train_loss -> test_acc
            print(
                f"Epoch {i} | "
                f"Train Acc={train_acc:.4f} | "
                f"Train Loss={train_loss:.4f}"
            )

            # =====================
            # Early Stopping
            # =====================
            if test_acc > best_acc + min_delta:
                best_acc = test_acc
                counter = 0
                torch.save(model.state_dict(), path)
            else:
                counter += 1

                if counter >= patience:
                    print(" Early Stopping Triggered")
                    mlflow.log_metric("stopped_epoch", i)
                    break

        # =====================
        # Best model 로드 후 저장 (안전장치 추가)
        # =====================
        if os.path.exists(path):
            model.load_state_dict(torch.load(path))
            mlflow.log_artifact(path)
            mlflow.pytorch.log_model(model, artifact_path="best_model")
            print("최고 성능 모델이 MLflow에 저장되었습니다.")
        else:
            print("저장된 모델 가중치가 없습니다. (성능 개선 없음)")

        mlflow.log_metric("best_test_accuracy", best_acc)
        gc.collect()
        torch.cuda.empty_cache()