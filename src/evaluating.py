
import torch





@torch.no_grad()
def evaluate(model,datloader,criterion,device):
    model.eval() # 학습모드
    model=model.to(device) # gpu 올리자
    total_loss=0 # 전체 반환 로스값
    correct=0 # 맞춘거 개수
    total=0 # batchsize
    for x,y in datloader:
        x=x.to(device) # x,y gpu
        y=y.to(device) #
        out=model(x) 
        loss=criterion(out,y)
        bn=x.size(0)
        total_loss+=loss.item()*bn
        total+=bn
        pred=out.argmax(dim=1)
        correct+=(pred==y).sum().item()
        
    print(f"Test Loss={total_loss/total:.4f}, Test Accuracy={correct/total:.4f}")
    return total_loss/total, correct/total